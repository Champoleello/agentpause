"""The high-level, provider-agnostic scheduler API.

This is the piece users actually touch. It ties the pure components
(:class:`~agentpause.estimator.Estimator`, the risk rule,
:class:`~agentpause.state.StateStore`) into a small workflow:

    sched = PredictiveScheduler(backend=..., telemetry=...)
    with sched.session("task-1") as s:      # resumes automatically if a checkpoint exists
        for question in questions[s.step:]:  # skip steps already done before a suspend
            s.add_user(question)
            if s.should_suspend():           # predictive check, before the call
                s.checkpoint()
                break
            reply = s.call()                 # perform the LLM call, record consumption
        else:
            s.complete()                     # task finished: drop the checkpoint

``backend`` and ``telemetry`` are injected callables, so the scheduler depends on
no particular provider. Adapters (LiteLLM, LangGraph, per-provider telemetry)
supply them; tests supply stubs.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .errors import BackendError, RateLimitHit
from .estimator import Estimator
from .forecast import Forecast, forecast_run
from .retry import RetryPolicy
from .risk import Budget, Decision, decide, should_checkpoint
from .state import Checkpoint, StateStore

# messages -> (reply_text, tokens_used)
Backend = Callable[[List[Dict[str, str]]], Tuple[str, int]]
# async variant: messages -> awaitable (reply_text, tokens_used)
AsyncBackend = Callable[[List[Dict[str, str]]], "Any"]
# () -> remaining tokens (int) or a full Budget reading
Telemetry = Callable[[], "int | Budget"]
# text -> token count
TokenCounter = Callable[[str], int]


def _default_token_counter(text: str) -> int:
    """Rough offline token estimate (~4 chars/token). Replace via ``count_tokens``."""
    return max(1, len(text) // 4)


class Session:
    """A single resumable agent run, bound to a ``session_id``."""

    def __init__(self, scheduler: "PredictiveScheduler", session_id: str) -> None:
        self._sched = scheduler
        self.session_id = session_id
        self._started_at = scheduler.clock()   # for the optional time budget
        cp = scheduler.store.load(session_id)
        self.resumed = cp is not None and cp.step > 0
        self._cp = cp if cp is not None else Checkpoint(session_id=session_id)
        # F9.3: restore the learned statistics saved with the checkpoint —
        # a resumed session starts calibrated instead of re-learning
        est_state = self._cp.extra.get("estimator")
        if self.resumed and est_state:
            scheduler.estimator.load(est_state)

    # -- read-only view -----------------------------------------------------

    @property
    def step(self) -> int:
        return self._cp.step

    @property
    def messages(self) -> List[Dict[str, str]]:
        return self._cp.messages

    @property
    def total_tokens_used(self) -> int:
        return self._cp.total_tokens_used

    # -- building the conversation -----------------------------------------

    def add_message(self, role: str, content: str) -> None:
        self._cp.messages.append({"role": role, "content": content})

    def add_user(self, content: str) -> None:
        self.add_message("user", content)

    def add_system(self, content: str) -> None:
        self.add_message("system", content)

    # -- the predictive decision -------------------------------------------

    def _input_tokens(self) -> int:
        return sum(self._sched.count_tokens(m["content"]) for m in self._cp.messages)

    def _read_budget(self) -> Budget:
        """Read telemetry fresh; accept a plain int (legacy) or a Budget."""
        raw = self._sched.telemetry()
        return raw if isinstance(raw, Budget) else Budget(remaining_tokens=int(raw))

    def should_suspend(self) -> bool:
        """True if the next call would not safely fit the remaining budget.

        Telemetry is read fresh on every call — never from the checkpoint —
        because a serialized budget value goes stale after a window reset.
        Considers both the token budget (TPM) and, when reported, the
        request budget (RPM).
        """
        return self.next_action().action != "continue"

    def next_action(self) -> Decision:
        """The full three-valued decision: ``continue`` / ``wait`` / ``checkpoint``.

        ``wait`` means the budget does not fit but the provider window resets
        within ``wait_threshold_s`` — sleeping briefly beats a suspend/resume
        cycle. The returned :class:`Decision` carries the budget reading and
        the estimate for logging.
        """
        budget = self._read_budget()
        input_tokens = self._input_tokens()
        # preventive compaction (F10.2): don't wait for the overflow wall —
        # when context pressure crosses the threshold, shrink old history
        # BEFORE deciding, so the decision sees the slimmer context
        cw = self._sched.context_window
        if cw is not None and input_tokens > self._sched.compact_at * cw:
            saved = self.compact(keep_last=self._sched.compact_keep_last,
                                 max_chars=self._sched.compact_max_chars)
            if saved:
                input_tokens = self._input_tokens()
                self._sched._emit("compacted", {
                    "chars_saved": saved, "input_tokens": input_tokens,
                    "session_id": self.session_id, "step": self.step,
                })
        estimated = self._sched.estimator.estimate(input_tokens)
        sigma = self._sched.estimator.sigma(estimated)
        if self._sched.quantile is not None:
            upper = self._sched.estimator.upper_estimate(input_tokens, self._sched.quantile)
            if upper is not None:
                # the quantile bound already covers the tail: no extra margin
                estimated, sigma = upper, 0.0
        # latency dimension: fold the task's time budget into the read budget,
        # and ask the estimator how long the next step will take (if it can)
        if self._sched.time_budget_s is not None and budget.remaining_seconds is None:
            elapsed = self._sched.clock() - self._started_at
            budget.remaining_seconds = self._sched.time_budget_s - elapsed
        est_latency = None
        predict_latency = getattr(self._sched.estimator, "estimate_latency", None)
        if callable(predict_latency):
            est_latency = predict_latency(input_tokens)
        d = decide(budget, estimated, sigma,
                   self._sched.safety_k, self._sched.wait_threshold_s,
                   estimated_input=input_tokens,
                   estimated_output=self._sched.estimator.max_tokens,
                   estimated_latency=est_latency,
                   rpm_margin=self._sched.rpm_margin)
        # monetary hard constraint: money does not reset by waiting,
        # so an overrun always means checkpoint — never wait
        price = self._sched.price_per_1k_tokens
        remaining_money = self._sched.money_remaining
        if price is not None and remaining_money is not None:
            estimated_cost = estimated / 1000.0 * price
            if estimated_cost > remaining_money:
                d = Decision(action="checkpoint", budget=budget,
                             estimated=estimated, sigma=sigma)
        self._sched._emit("decision", {
            "action": d.action, "estimated": estimated,
            "remaining_tokens": budget.remaining_tokens,
            "session_id": self.session_id, "step": self.step,
        })
        return d

    def forecast(self, steps_remaining: int) -> Forecast:
        """Project the cost of the next ``steps_remaining`` steps (F11.1).

        A pure read: the live budget comes through the same telemetry path
        ``next_action()`` uses, the per-step cost from the estimator's current
        beliefs, and everything else from a simulation against the window's
        refill rate (see :func:`~agentpause.forecast.forecast_run`). Nothing
        is consumed, no backend call is made, and no session or estimator
        state is mutated — calling it twice yields the same answer.
        """
        budget = self._read_budget()
        input_tokens = self._input_tokens()
        estimated = self._sched.estimator.estimate(input_tokens)
        sigma = self._sched.estimator.sigma(estimated)
        est_latency = None
        predict_latency = getattr(self._sched.estimator, "estimate_latency", None)
        if callable(predict_latency):
            est_latency = predict_latency(input_tokens)
        # remaining time budget: prefer telemetry's own deadline, else fold in
        # the scheduler's run deadline minus elapsed (same rule as decide())
        time_left = budget.remaining_seconds
        if time_left is None and self._sched.time_budget_s is not None:
            elapsed = self._sched.clock() - self._started_at
            time_left = self._sched.time_budget_s - elapsed
        return forecast_run(
            budget, estimated, sigma, steps_remaining,
            k=self._sched.safety_k,
            latency_per_step=est_latency,
            price_per_1k_tokens=self._sched.price_per_1k_tokens,
            wait_threshold_s=self._sched.wait_threshold_s,
            time_budget_s=time_left,
        )

    # -- performing a step --------------------------------------------------

    def call(self) -> str:
        """Perform the LLM call, record consumption, advance the step.

        An unexpected 429 (:class:`~agentpause.errors.RateLimitHit`) is
        retried per the scheduler's :class:`~agentpause.retry.RetryPolicy`,
        and each hit adapts the safety factor upward (the prediction was too
        optimistic). If retries run out the error propagates — and the session
        state is left UNTOUCHED, so a checkpoint/resume stays clean.
        """
        input_tokens = self._input_tokens()
        retry = self._sched.retry
        attempt = 0
        started = self._sched.clock()
        while True:
            try:
                reply, used = self._sched.backend(self._cp.messages)
                break
            except RateLimitHit as hit:
                self._sched._register_rate_limit_hit()
                if attempt >= retry.max_retries:
                    raise
                delay = hit.retry_after if hit.retry_after is not None else retry.delay(attempt)
                self._sched._emit("retry", {"attempt": attempt + 1, "delay_s": delay})
                retry.sleep_fn(delay)
                attempt += 1
            except BackendError as err:
                # transient infrastructure failures (5xx) deserve a retry;
                # request problems (4xx) do not — retrying wastes budget
                if not err.retriable or attempt >= retry.max_retries:
                    raise
                delay = retry.delay(attempt)
                self._sched._emit("retry", {"attempt": attempt + 1, "delay_s": delay})
                retry.sleep_fn(delay)
                attempt += 1
        # state mutations happen only after a successful call
        self._record_success(input_tokens, used, reply,
                             latency=self._sched.clock() - started)
        return reply

    def _record_success(self, input_tokens: int, used: int, reply: str,
                        latency: Optional[float] = None) -> None:
        # pass latency only to estimators that learn it (e.g. FeatureEstimator)
        try:
            self._sched.estimator.record(input_tokens, used, latency=latency)
        except TypeError:
            self._sched.estimator.record(input_tokens, used)
        self._cp.total_tokens_used += used
        self._cp.step += 1
        self.add_message("assistant", reply)
        if self._sched.price_per_1k_tokens is not None:
            self._sched.money_spent += used / 1000.0 * self._sched.price_per_1k_tokens
        self._sched._emit("step_completed", {
            "session_id": self.session_id, "step": self._cp.step,
            "tokens_used": used, "money_spent": self._sched.money_spent,
        })

    async def acall(self) -> str:
        """Async twin of :meth:`call`: same rule, same retry, same guarantees.

        Requires the scheduler to be built with ``async_backend=...``
        (e.g. ``LiteLLMAdapter(...).abackend``).
        """
        if self._sched.async_backend is None:
            raise RuntimeError(
                "acall() needs an async backend: "
                "PredictiveScheduler(..., async_backend=adapter.abackend)"
            )
        input_tokens = self._input_tokens()
        retry = self._sched.retry
        attempt = 0
        started = self._sched.clock()
        while True:
            try:
                reply, used = await self._sched.async_backend(self._cp.messages)
                break
            except RateLimitHit as hit:
                self._sched._register_rate_limit_hit()
                if attempt >= retry.max_retries:
                    raise
                delay = hit.retry_after if hit.retry_after is not None else retry.delay(attempt)
                await retry.asleep(delay)
                attempt += 1
            except BackendError as err:
                if not err.retriable or attempt >= retry.max_retries:
                    raise
                await retry.asleep(retry.delay(attempt))
                attempt += 1
        self._record_success(input_tokens, used, reply,
                             latency=self._sched.clock() - started)
        return reply

    # -- persistence --------------------------------------------------------

    def compact(self, keep_last: int = 4, max_chars: int = 200) -> int:
        """Shrink old history (overflow policy, §8.6): see
        :meth:`agentpause.state.Checkpoint.compact`."""
        return self._cp.compact(keep_last=keep_last, max_chars=max_chars)

    def summarize_with(self, summarizer, keep_last: int = 4) -> int:
        """Semantic compression via an injected cheap model: see
        :meth:`agentpause.state.Checkpoint.summarize_with`."""
        return self._cp.summarize_with(summarizer, keep_last=keep_last)

    def checkpoint(self) -> str:
        """Serialize the logical state so the run can resume later.

        The estimator's learned statistics ride along (F9.3): unlike the
        budget — which is re-read fresh on resume, never trusted — the
        statistics stay valid across a suspension.
        """
        self._cp.extra["estimator"] = self._sched.estimator.to_dict()
        return self._sched.store.save(self._cp)

    def complete(self) -> None:
        """Mark the task finished and drop its checkpoint."""
        self._sched.store.clear(self.session_id)

    # -- context manager ----------------------------------------------------

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False  # never suppress exceptions


class PredictiveScheduler:
    """Factory and configuration holder for :class:`Session` runs.

    Args:
        backend: callable ``messages -> (reply, tokens_used)`` performing the LLM call.
        telemetry: callable ``() -> remaining_tokens`` reading the rate-limit window.
        store: where checkpoints live (defaults to ``.agentpause/`` on disk).
        estimator: cost estimator (defaults to a fresh :class:`Estimator`).
        count_tokens: text->token counter (defaults to a ~4 chars/token heuristic).
        safety_k: safety factor; higher suspends earlier.
        wait_threshold_s: if the window resets within this many seconds,
            ``next_action()`` returns ``wait`` instead of ``checkpoint``.
        retry: how to wait-and-retry on unexpected 429s (see
            :class:`~agentpause.retry.RetryPolicy`).
        k_bump: how much ``safety_k`` grows after each unexpected 429 —
            the hit is feedback that the estimates run optimistic.
        k_max: ceiling for the adaptive safety factor.
        price_per_1k_tokens: model price, for the monetary hard constraint.
        money_budget: spending cap for this scheduler. When the estimated
            next step would overrun it, the decision is ``checkpoint`` —
            never ``wait``, because money does not reset with the window.
        on_event: observability hook ``(event_name, info_dict) -> None``.
            Events: ``decision``, ``step_completed``, ``rate_limit_hit``,
            ``retry``. Exceptions inside the hook are swallowed — telemetry
            must never take down the agent.
        quantile: when set (e.g. ``0.95``) and enough history exists, the
            decision uses the empirical q-quantile of past estimation errors
            instead of the ``k·sigma`` margin — statistically honest with
            heavy-tailed consumption. Falls back to ``k·sigma`` until
            8 steps are recorded.
        time_budget_s: optional wall-clock deadline for the whole run. When
            set, the time left (``time_budget_s`` minus elapsed) becomes a
            budget dimension: if the estimator predicts a step can't finish in
            it, the decision is ``checkpoint`` (time never refills by waiting).
        clock: monotonic time source for ``time_budget_s`` (injectable for tests).
    """

    def __init__(
        self,
        backend: Optional[Backend],
        telemetry: Telemetry,
        store: Optional[StateStore] = None,
        estimator: Optional[Estimator] = None,
        count_tokens: Optional[TokenCounter] = None,
        safety_k: float = 2.0,
        wait_threshold_s: float = 15.0,
        retry: Optional[RetryPolicy] = None,
        k_bump: float = 0.25,
        k_max: float = 4.0,
        async_backend: Optional[AsyncBackend] = None,
        price_per_1k_tokens: Optional[float] = None,
        money_budget: Optional[float] = None,
        on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        quantile: Optional[float] = None,
        context_window: Optional[int] = None,
        compact_at: float = 0.6,
        compact_keep_last: int = 4,
        compact_max_chars: int = 200,
        time_budget_s: Optional[float] = None,
        clock: Callable[[], float] = time.monotonic,
        rpm_margin: int = 0,
    ) -> None:
        if backend is None and async_backend is None:
            raise ValueError("Provide backend=, async_backend=, or both.")
        self.backend = backend
        self.async_backend = async_backend
        self.telemetry = telemetry
        self.store = store if store is not None else StateStore()
        self.estimator = estimator if estimator is not None else Estimator()
        self.count_tokens = count_tokens if count_tokens is not None else _default_token_counter
        self.safety_k = safety_k
        self.wait_threshold_s = wait_threshold_s
        self.retry = retry if retry is not None else RetryPolicy()
        self.k_bump = k_bump
        self.k_max = k_max
        self.rate_limit_hits = 0
        self.price_per_1k_tokens = price_per_1k_tokens
        self.money_budget = money_budget
        self.money_spent = 0.0
        self.on_event = on_event
        self.quantile = quantile
        self.context_window = context_window
        self.compact_at = compact_at
        self.compact_keep_last = compact_keep_last
        self.compact_max_chars = compact_max_chars
        self.time_budget_s = time_budget_s
        self.clock = clock
        # safety slack on the request budget (see risk.decide); 0 keeps the
        # historical behavior, field practice elsewhere uses 1-2
        self.rpm_margin = rpm_margin

    @property
    def money_remaining(self) -> Optional[float]:
        if self.money_budget is None:
            return None
        return self.money_budget - self.money_spent

    def _emit(self, name: str, info: Dict[str, Any]) -> None:
        """Fire the observability hook; never let it break the run."""
        if self.on_event is not None:
            try:
                self.on_event(name, info)
            except Exception:
                pass

    def _register_rate_limit_hit(self) -> None:
        """A 429 slipped through: count it and grow the safety margin."""
        self.rate_limit_hits += 1
        self.safety_k = min(self.safety_k + self.k_bump, self.k_max)
        self._emit("rate_limit_hit", {"total_hits": self.rate_limit_hits,
                                      "safety_k": self.safety_k})

    def session(self, session_id: str) -> Session:
        """Start (or resume) a run identified by ``session_id``."""
        return Session(self, session_id)
