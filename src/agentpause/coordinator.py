"""Shared-budget coordination for multiple agents (D5).

One rate-limit window, many agents. A single TPM/RPM budget is a shared pool:
if three agents each independently think "I have 9k tokens of headroom" and all
fire, they collectively blow the window and two of them eat a 429 — the exact
cross-agent overcommit the single-agent scheduler can't see.

The coordinator makes the shared pool visible. Each agent asks before calling;
a granted call **reserves** its predicted cost (estimate + k·σ margin) out of
the pool, so every other agent's next decision is made against the budget that
actually remains. Releasing on completion reconciles the reservation with the
tokens truly spent.

This is also where priority / deadline / fairness finally mean something: with
one agent there is nothing to arbitrate, but when several contend for a pool too
small for all of them, :meth:`arbitrate` allocates it — highest priority first,
then longest-waiting — granting until the window is spent and telling the rest
to wait or checkpoint.

    from agentpause import MultiAgentCoordinator
    from agentpause.adapters.openai_compat import OpenAICompatAdapter

    shared = OpenAICompatAdapter.for_model("groq/llama-3.1-8b-instant")
    coord = MultiAgentCoordinator(telemetry=shared.budget)
    coord.register("researcher", priority=1)
    coord.register("summarizer", priority=0)

    d = coord.request("researcher", estimated=1200, sigma=90)
    if d.action == "continue":
        reply, used = shared.backend(messages)
        coord.complete("researcher", actual_tokens=used)

The telemetry source can be a single adapter's ``budget`` or a
:class:`~agentpause.router.BudgetRouter` — routing across providers (F9.1) and
sharing one provider across agents (D5) compose.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Dict, List, Optional, Tuple

from .risk import Budget, Decision, decide

__all__ = ["MultiAgentCoordinator"]


class MultiAgentCoordinator:
    """Distribute one shared rate-limit window across several agents.

    Args:
        telemetry: ``() -> Budget`` for the SHARED window (one adapter's
            ``budget`` or a ``BudgetRouter.budget``).
        k: safety factor for the reserved margin (matches the scheduler's).
        wait_threshold_s: passed through to :func:`~agentpause.risk.decide`.
        clock: monotonic time source (injectable for tests).
    """

    def __init__(
        self,
        telemetry: Callable[[], Budget],
        k: float = 2.0,
        wait_threshold_s: float = 15.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.telemetry = telemetry
        self.k = k
        self.wait_threshold_s = wait_threshold_s
        self._clock = clock
        self._priority: Dict[str, int] = {}
        self._order: Dict[str, int] = {}            # registration order (tiebreak)
        self._reservations: Dict[str, int] = {}     # agent -> tokens held
        self._waiting_since: Dict[str, float] = {}   # agent -> when it first waited

    # -- registration -----------------------------------------------------------

    def register(self, agent_id: str, priority: int = 0) -> None:
        """Announce an agent (higher ``priority`` wins contention)."""
        if agent_id not in self._order:
            self._order[agent_id] = len(self._order)
        self._priority[agent_id] = priority

    def _ensure(self, agent_id: str) -> None:
        if agent_id not in self._order:
            self.register(agent_id)

    # -- shared-pool accounting -------------------------------------------------

    @property
    def reserved(self) -> int:
        """Tokens currently held by in-flight agents."""
        return sum(self._reservations.values())

    def _margin(self, estimated: int, sigma: float) -> int:
        return estimated + math.ceil(self.k * sigma)

    def _shadow(self, extra_reserved: int = 0) -> Budget:
        """The shared budget as it looks after outstanding reservations.

        Other agents' holds are subtracted from ``remaining_tokens`` so this
        agent decides against what is genuinely free, not the raw header count.
        """
        b = self.telemetry()
        held = self.reserved + extra_reserved
        return Budget(
            remaining_tokens=max(0, b.remaining_tokens - held),
            remaining_requests=b.remaining_requests,
            reset_seconds=b.reset_seconds,
            reset_requests_seconds=b.reset_requests_seconds,
            remaining_input_tokens=b.remaining_input_tokens,
            remaining_output_tokens=b.remaining_output_tokens,
            limit_tokens=b.limit_tokens,
            refill_regime=b.refill_regime,
        )

    # -- streaming API: one agent at a time -------------------------------------

    def request(self, agent_id: str, estimated: int, sigma: float = 0.0) -> Decision:
        """Ask to call now. On ``continue`` the cost is reserved from the pool.

        The decision is made against the shared budget minus what other agents
        already hold, so two agents can't both be told "go" into a window that
        only fits one. A non-``continue`` decision reserves nothing and records
        that the agent is waiting (used to break ties fairly in
        :meth:`arbitrate`).
        """
        self._ensure(agent_id)
        # don't let this agent's own stale hold block its fresh request
        own = self._reservations.get(agent_id, 0)
        d = decide(self._shadow(extra_reserved=-own), estimated, sigma,
                   k=self.k, wait_threshold_s=self.wait_threshold_s)
        if d.action == "continue":
            self._reservations[agent_id] = self._margin(estimated, sigma)
            self._waiting_since.pop(agent_id, None)
        else:
            self._waiting_since.setdefault(agent_id, self._clock())
        return d

    def complete(self, agent_id: str, actual_tokens: Optional[int] = None) -> None:
        """Release the agent's reservation once its call has returned.

        ``actual_tokens`` is accepted for symmetry and future reconciliation;
        the live token count comes from the provider headers on the next read,
        so the reservation is simply dropped.
        """
        self._reservations.pop(agent_id, None)
        self._waiting_since.pop(agent_id, None)

    # -- batch API: arbitrate contention by priority then fairness --------------

    def arbitrate(
        self, requests: List[Tuple[str, int, float]]
    ) -> Dict[str, Decision]:
        """Allocate the shared window across simultaneous requests.

        ``requests`` is a list of ``(agent_id, estimated, sigma)``. Agents are
        served highest-priority first, then longest-waiting, then registration
        order; each grant reserves its cost so later agents in the same round
        decide against the shrinking pool. Everyone who doesn't fit gets a
        ``wait``/``checkpoint`` decision instead of a false ``continue``.

        Returns a ``{agent_id: Decision}`` map. Granted agents hold reservations
        afterward — call :meth:`complete` when each finishes.
        """
        for agent_id, _, _ in requests:
            self._ensure(agent_id)

        def sort_key(req: Tuple[str, int, float]) -> Tuple[int, float, int]:
            agent_id = req[0]
            waited = self._waiting_since.get(agent_id, self._clock())
            # priority desc, then earlier-waiting first, then registration order
            return (-self._priority[agent_id], waited, self._order[agent_id])

        results: Dict[str, Decision] = {}
        for agent_id, estimated, sigma in sorted(requests, key=sort_key):
            d = decide(self._shadow(), estimated, sigma,
                       k=self.k, wait_threshold_s=self.wait_threshold_s)
            if d.action == "continue":
                self._reservations[agent_id] = self._margin(estimated, sigma)
                self._waiting_since.pop(agent_id, None)
            else:
                self._waiting_since.setdefault(agent_id, self._clock())
            results[agent_id] = d
        return results
