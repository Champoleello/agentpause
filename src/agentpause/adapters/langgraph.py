"""LangGraph adapter: predictive suspension inside any graph.

`LangGraph <https://github.com/langchain-ai/langgraph>`_ persists state with a
checkpointer and can pause a run with ``interrupt()`` — but it decides
*reactively*: it saves after each node and crashes (or retries) when the
provider returns 429. This adapter adds the missing *predictive* gate.

Usage — drop a guard into the node that calls the LLM::

    from agentpause.adapters.langgraph import AgentPauseGuard

    guard = AgentPauseGuard(telemetry=adapter.telemetry)  # e.g. from LiteLLMAdapter

    def agent_node(state):
        guard.check(state["messages"])          # ← may pause the graph HERE
        reply = llm.invoke(state["messages"])
        guard.record(state["messages"], reply.usage_metadata["total_tokens"])
        return {"messages": state["messages"] + [reply]}

When the estimated cost of the next call does not fit the remaining budget,
``check()`` calls LangGraph's ``interrupt()``: the run pauses cleanly and the
configured checkpointer persists it. Resume it later with
``graph.invoke(Command(resume=True), config)``.

Design notes
------------
* **Fresh telemetry on every pass.** On resume LangGraph re-executes the node
  from the top, so ``check()`` runs again and re-reads the budget — a value
  serialized before the pause is stale by definition. If the budget is
  *still* too low, the guard interrupts again rather than letting the call
  fail with a 429.
* ``interrupt_fn`` is injectable so the guard is fully testable offline;
  by default it lazily imports ``langgraph.types.interrupt``.
* The guard owns an :class:`~agentpause.estimator.Estimator`; feed it with
  ``record()`` after each real call so ε and σ adapt to the workload.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from ..estimator import Estimator
from ..risk import Budget, decide

__all__ = ["AgentPauseGuard"]

# text -> token count
TokenCounter = Callable[[str], int]


def _default_token_counter(text: str) -> int:
    """Rough offline token estimate (~4 chars/token). Replace via ``count_tokens``."""
    return max(1, len(text) // 4)


def _default_interrupt() -> Callable[[Any], Any]:
    try:
        from langgraph.types import interrupt
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "The LangGraph adapter needs the 'langgraph' package. "
            "Install it with:  pip install agentpause[langgraph]"
        ) from exc
    return interrupt


class AgentPauseGuard:
    """The predictive gate for a LangGraph node.

    Args:
        telemetry: callable ``() -> remaining_tokens`` reading the rate-limit
            window (e.g. ``LiteLLMAdapter(...).telemetry``).
        estimator: cost estimator (defaults to a fresh :class:`Estimator`).
        count_tokens: text->token counter (defaults to ~4 chars/token).
        safety_k: safety factor; higher suspends earlier.
        interrupt_fn: the pause primitive. Defaults to LangGraph's
            ``interrupt`` (imported lazily); tests inject a fake.
        wait_threshold_s: if the budget does not fit but the provider window
            resets within this many seconds, the guard *sleeps in place*
            instead of interrupting the graph — a short pause is cheaper than
            a suspend/resume cycle. Requires telemetry that reports
            ``reset_seconds`` (e.g. ``LiteLLMAdapter(...).budget``).
        sleep_fn: how to wait (defaults to ``time.sleep``; tests inject a fake).
        max_resumes: how many resume/wait passes with a still-too-low budget
            are tolerated per ``check()`` call — a loop bound, not a policy;
            each pass re-reads telemetry.
    """

    def __init__(
        self,
        telemetry: Callable[[], "int | Budget"],
        estimator: Optional[Estimator] = None,
        count_tokens: Optional[TokenCounter] = None,
        safety_k: float = 2.0,
        interrupt_fn: Optional[Callable[[Any], Any]] = None,
        wait_threshold_s: float = 15.0,
        sleep_fn: Callable[[float], None] = time.sleep,
        async_sleep_fn: Optional[Callable[[float], Any]] = None,
        chunk_s: float = 10.0,
        max_resumes: int = 100,
    ) -> None:
        self.telemetry = telemetry
        self.estimator = estimator if estimator is not None else Estimator()
        self.count_tokens = count_tokens if count_tokens is not None else _default_token_counter
        self.safety_k = safety_k
        self._interrupt = interrupt_fn
        self.wait_threshold_s = wait_threshold_s
        self.sleep_fn = sleep_fn
        self.async_sleep_fn = async_sleep_fn
        self.chunk_s = chunk_s
        self.max_resumes = max_resumes

    # -- internals -----------------------------------------------------------

    def _input_tokens(self, messages: List[Dict[str, str]]) -> int:
        return sum(self.count_tokens(m.get("content") or "") for m in messages)

    def _interrupt_call(self, payload: Any) -> Any:
        if self._interrupt is None:
            self._interrupt = _default_interrupt()
        return self._interrupt(payload)

    # -- the predictive gate ---------------------------------------------------

    def check(self, messages: List[Dict[str, str]]) -> None:
        """Pause the graph if the next LLM call would not fit the budget.

        Reads telemetry fresh on every pass (both the token/TPM and, when
        reported, the request/RPM dimension). Three outcomes:

        * ``continue`` — returns, the node proceeds to the LLM call.
        * ``wait`` — the window resets within ``wait_threshold_s``: the guard
          sleeps in place and re-checks, without pausing the whole graph.
        * ``checkpoint`` — calls ``interrupt()``: the node aborts and the
          checkpointer persists the run. On resume the node re-executes and
          only proceeds once the (freshly read) budget fits.
        """
        input_tokens = self._input_tokens(messages)
        for _ in range(self.max_resumes):
            raw = self.telemetry()
            budget = raw if isinstance(raw, Budget) else Budget(remaining_tokens=int(raw))
            estimated = self.estimator.estimate(input_tokens)
            sigma = self.estimator.sigma(estimated)
            d = decide(budget, estimated, sigma, self.safety_k, self.wait_threshold_s)
            if d.action == "continue":
                return
            if d.action == "wait":
                # reset is imminent: sleeping beats a suspend/resume cycle.
                # Sleep in chunks and re-read: if the bucket refills sooner,
                # resume sooner (and the samples feed regime detection).
                wait_s = d.wait_seconds or budget.reset_seconds or 1.0
                self.sleep_fn(min(wait_s, self.chunk_s) + 0.5)
                continue
            # interrupt() raises on the first pass (the node aborts here);
            # on a resume pass it RETURNS, and the loop re-reads telemetry.
            self._interrupt_call({
                "reason": "predicted_rate_limit",
                "remaining": budget.remaining_tokens,
                "remaining_requests": budget.remaining_requests,
                "reset_seconds": budget.reset_seconds,
                "estimated": estimated,
                "sigma": round(sigma, 1),
                "safety_k": self.safety_k,
            })

    async def acheck(self, messages: List[Dict[str, str]]) -> None:
        """Async twin of :meth:`check`, for async LangGraph nodes.

        Same three outcomes; the ``wait`` pause uses ``asyncio.sleep`` (or the
        injected ``async_sleep_fn``) so the event loop is never blocked.
        """
        import asyncio

        input_tokens = self._input_tokens(messages)
        for _ in range(self.max_resumes):
            raw = self.telemetry()
            budget = raw if isinstance(raw, Budget) else Budget(remaining_tokens=int(raw))
            estimated = self.estimator.estimate(input_tokens)
            sigma = self.estimator.sigma(estimated)
            d = decide(budget, estimated, sigma, self.safety_k, self.wait_threshold_s)
            if d.action == "continue":
                return
            if d.action == "wait":
                wait_s = d.wait_seconds or budget.reset_seconds or 1.0
                delay = min(wait_s, self.chunk_s) + 0.5
                if self.async_sleep_fn is not None:
                    await self.async_sleep_fn(delay)
                else:
                    await asyncio.sleep(delay)
                continue
            self._interrupt_call({
                "reason": "predicted_rate_limit",
                "remaining": budget.remaining_tokens,
                "remaining_requests": budget.remaining_requests,
                "reset_seconds": budget.reset_seconds,
                "estimated": estimated,
                "sigma": round(sigma, 1),
                "safety_k": self.safety_k,
            })

    # -- learning ---------------------------------------------------------------

    def record(self, messages: List[Dict[str, str]], used: int) -> None:
        """Feed the estimator with the real consumption of a completed call.

        Args:
            messages: the input that was sent to the LLM.
            used: total tokens the provider reports for the call.
        """
        self.estimator.record(self._input_tokens(messages), used)
