"""One-call facade over BudgetRouter + MultiAgentCoordinator + Estimator.

The README's "Scaling up: many providers, many agents, smarter estimates"
section shows the only way to combine these three pieces today: build a
router, build a coordinator around ``router.budget``, register each agent,
build a separate estimator, compute ``estimated``/``sigma`` from it, call
``coordinator.request(...)``, check ``.action == "continue"``, call
``router.backend(...)``, call ``coordinator.complete(...)``, and call
``est.record(...)`` — six manual pieces, wired by hand, for every single
call any agent makes. That fragmentation is exactly what
:class:`~agentpause.adapters.local_resources.LocalResourceBudget` already
collapsed for local-resource budgets. :class:`AgentFleet` does the same job
here: one constructor call builds the router and the coordinator together
(same clock, the coordinator's telemetry wired to the router's budget), one
``.register()`` per agent (optional — unregistered agents auto-register,
mirroring :class:`~agentpause.coordinator.MultiAgentCoordinator`'s own
behavior), and one ``.call()`` per step replaces the request → check →
call → complete → record dance.

It composes only :mod:`agentpause.router`, :mod:`agentpause.coordinator`,
and :mod:`agentpause.estimator` — all already dependency-free core modules
— so it lives at the same level as they do, not under ``adapters/``.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Tuple

from .coordinator import MultiAgentCoordinator
from .estimator import Estimator
from .risk import Decision
from .router import BudgetRouter, ProviderArg

__all__ = ["AgentFleet"]

AgentArg = "str | Tuple[str, int]"


def _default_token_counter(text: str) -> int:
    """Rough offline token estimate (~4 chars/token).

    A private copy of ``scheduler.py``'s ``_default_token_counter`` — that
    function is a one-line pure heuristic, not a shared contract, so it is
    duplicated here rather than imported across a leading-underscore name.
    """
    return max(1, len(text) // 4)


class AgentFleet:
    """Construction + per-step facade for many agents sharing many providers.

    What's automatic
    -----------------
    * Building :class:`~agentpause.router.BudgetRouter` and
      :class:`~agentpause.coordinator.MultiAgentCoordinator` together, wired
      to the SAME clock and to each other (the coordinator's ``telemetry``
      is the router's ``budget``, so the shared pool it reasons about is
      whichever provider the router would actually route to next).
    * A fresh, ISOLATED estimator per ``agent_id`` (via ``estimator_factory``,
      default :class:`~agentpause.estimator.Estimator`) — one agent's
      learned history never leaks into another's.
    * The request → check action → call → complete → record dance, collapsed
      into one :meth:`call`.
    * Auto-registration: calling with an ``agent_id`` you never
      ``.register()``-ed just works, at ``priority=0``, mirroring
      :class:`~agentpause.coordinator.MultiAgentCoordinator`'s own
      ``_ensure`` behavior.

    What you must still provide
    ----------------------------
    * The list of ``providers`` — same shape ``BudgetRouter(*providers)``
      accepts (adapter-like objects, or ``(name, provider)`` pairs). At
      least one is required; zero is a clear failure, not a silently broken
      object.
    * Agent ids, if you want explicit priorities — pass ``agents=`` or call
      ``.register(agent_id, priority=...)`` yourself.

    Scope boundary — what this deliberately does NOT do
    -----------------------------------------------------
    This facade simplifies CONSTRUCTION and the PER-STEP DANCE. It does
    **not** give fleet agents a resumable, checkpointable
    :class:`~agentpause.scheduler.Session`-like loop — no ``.checkpoint()``,
    no ``.should_suspend()`` context manager. ``Session.next_action()``
    computes its ``estimated``/``sigma`` from its own private estimator and
    calls :func:`~agentpause.risk.decide` directly; there is no existing
    seam to redirect that decision through
    :meth:`~agentpause.coordinator.MultiAgentCoordinator.request` instead,
    and the coordinator/router were deliberately built as separate,
    adapter-level constructs, not wired into ``Session``/
    ``PredictiveScheduler`` — the core stays dependency-free, adapters (and
    facades like this one) compose it from outside. If you need per-agent
    checkpoint/resume, compose your own
    :class:`~agentpause.state.StateStore` per ``agent_id``, exactly as
    single-agent usage already does elsewhere.

    Power-user access
    -------------------
    ``self.router`` and ``self.coordinator`` are the real, live objects used
    internally (not copies) — drop to ``fleet.coordinator.arbitrate(...)``
    for the batch/contention case, or read ``fleet.router.last_route``,
    exactly as :class:`~agentpause.adapters.local_resources.LocalResourceBudget`
    still exposes its underlying pieces for fine control.

    Args:
        providers: passed through as ``*providers`` to
            :class:`~agentpause.router.BudgetRouter`.
        agents: optional list of agent id strings, or ``(agent_id,
            priority)`` tuples, registered immediately via :meth:`register`.
        estimator_factory: zero-arg callable returning a fresh
            estimator-shaped object (``.estimate``, ``.sigma``, ``.record``).
            Defaults to :class:`~agentpause.estimator.Estimator`;
            :class:`~agentpause.regression.FeatureEstimator` (or any
            equivalent shape) also works — nothing here assumes the default.
        count_tokens: ``text -> int``. Defaults to the same ~4-chars/token
            heuristic ``scheduler.py`` uses.
        router_key: forwarded to ``BudgetRouter(key=...)``.
        on_route: forwarded to ``BudgetRouter(on_route=...)``.
        k: forwarded to ``MultiAgentCoordinator(k=...)``.
        wait_threshold_s: forwarded to ``MultiAgentCoordinator(wait_threshold_s=...)``.
        clock: monotonic time source, forwarded to BOTH the router and the
            coordinator — the same clock for both, injectable for tests.
        default_cooldown_s: forwarded to ``BudgetRouter(default_cooldown_s=...)``.
    """

    def __init__(
        self,
        providers: List[ProviderArg],
        agents: Optional[List] = None,
        *,
        estimator_factory: Callable[[], Estimator] = Estimator,
        count_tokens: Optional[Callable[[str], int]] = None,
        router_key: Optional[Callable] = None,
        on_route: Optional[Callable[[str, object], None]] = None,
        k: float = 2.0,
        wait_threshold_s: float = 15.0,
        clock: Callable[[], float] = time.monotonic,
        default_cooldown_s: float = 5.0,
    ) -> None:
        if not providers:
            raise ValueError("AgentFleet needs at least one provider")
        self.router = BudgetRouter(
            *providers,
            key=router_key,
            on_route=on_route,
            clock=clock,
            default_cooldown_s=default_cooldown_s,
        )
        self.coordinator = MultiAgentCoordinator(
            telemetry=self.router.budget,
            k=k,
            wait_threshold_s=wait_threshold_s,
            clock=clock,
        )
        self._estimator_factory = estimator_factory
        self.count_tokens = count_tokens if count_tokens is not None else _default_token_counter
        self._estimators: Dict[str, Estimator] = {}

        for a in agents or []:
            if isinstance(a, tuple) and len(a) == 2:
                agent_id, priority = a
                self.register(agent_id, priority=priority)
            else:
                self.register(a)

    # -- registration -------------------------------------------------------

    def register(self, agent_id: str, priority: int = 0) -> None:
        """Announce an agent and, if new, give it a fresh estimator.

        Re-registering an already-known ``agent_id`` only updates its
        priority — it never resets that agent's learned estimator history.
        """
        self.coordinator.register(agent_id, priority=priority)
        if agent_id not in self._estimators:
            self._estimators[agent_id] = self._estimator_factory()

    def estimator_for(self, agent_id: str) -> Estimator:
        """This agent's estimator, auto-registering (priority=0) if unknown.

        Mirrors :class:`~agentpause.coordinator.MultiAgentCoordinator`'s own
        ``_ensure`` auto-registration: :meth:`call` on a never-registered
        ``agent_id`` just works.
        """
        if agent_id not in self._estimators:
            self.register(agent_id)
        return self._estimators[agent_id]

    # -- the per-step dance, collapsed ---------------------------------------

    def call(
        self, agent_id: str, messages: List[Dict[str, str]]
    ) -> Tuple[Decision, Optional[str], Optional[int]]:
        """Estimate, ask the coordinator, call the router, and record — in one step.

        On a non-``continue`` decision, nothing further happens: no provider
        is called, and this agent's estimator is left untouched. The caller
        decides wait vs. checkpoint from ``d.action``, exactly like
        single-agent ``Session.next_action()`` callers already do.

        Returns ``(decision, reply, used_tokens)``; ``reply``/``used_tokens``
        are ``None`` unless ``decision.action == "continue"``.
        """
        est = self.estimator_for(agent_id)
        input_tokens = sum(self.count_tokens(m["content"]) for m in messages)
        estimated = est.estimate(input_tokens)
        sigma = est.sigma(estimated)
        d = self.coordinator.request(agent_id, estimated=estimated, sigma=sigma)
        if d.action != "continue":
            return d, None, None
        reply, used = self.router.backend(messages)
        self.coordinator.complete(agent_id, actual_tokens=used)
        est.record(input_tokens, used)
        return d, reply, used
