"""Budget-aware multi-provider routing (F9.1).

Where :class:`~agentpause.fallback.FallbackBackend` is *reactive* — it tries a
fixed order and only switches after a call fails — the router is *predictive*:
it reads every provider's telemetry FIRST and sends the next call to the one
with the most headroom, before anyone hits a 429. Same philosophy as the
scheduler itself (act on the budget you can see, don't wait for the error).

It duck-types the adapter shape, so any adapter (or anything exposing
``backend(messages) -> (reply, tokens)`` and ``budget() -> Budget``) is a valid
provider, and the router itself exposes the same two callables — so it drops
straight into the scheduler in place of a single adapter::

    from agentpause import BudgetRouter, PredictiveScheduler
    from agentpause.adapters.openai_compat import OpenAICompatAdapter
    from agentpause.adapters.anthropic import AnthropicAdapter

    router = BudgetRouter(
        ("groq",   OpenAICompatAdapter.for_model("groq/llama-3.1-8b-instant")),
        ("claude", AnthropicAdapter("claude-haiku-4-5")),
    )
    sched = PredictiveScheduler(backend=router.backend, telemetry=router.budget)

Selection metric: by default the provider with the most ``remaining_tokens``.
Providers differ in window size, so a fraction-of-window key is often fairer —
pass ``key=`` to override (e.g. ``key=lambda b: b.remaining_tokens / (b.limit_tokens or 1)``).

Cooldown: a provider that just returned a 429 (or a retriable backend error)
is parked until its window resets, so ``budget()`` and the next pick skip it
instead of steering back into the wall.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .errors import BackendError, RateLimitHit, TelemetryError
from .risk import Budget

__all__ = ["BudgetRouter"]

Backend = Callable[[List[Dict[str, str]]], Tuple[str, int]]

# a provider is anything with .backend and .budget (an adapter); accepted bare
# or as a (name, provider) pair for readable diagnostics
Provider = Any
ProviderArg = Union[Provider, Tuple[str, Provider]]

_DEFAULT_COOLDOWN_S = 5.0     # used when a 429 carries no reset/retry-after hint


class _Route:
    """One provider plus the bookkeeping the router keeps about it."""

    __slots__ = ("name", "provider", "cooldown_until")

    def __init__(self, name: str, provider: Provider) -> None:
        self.name = name
        self.provider = provider
        self.cooldown_until: float = 0.0

    def backend(self, messages: List[Dict[str, str]]) -> Tuple[str, int]:
        return self.provider.backend(messages)

    def budget(self) -> Budget:
        return self.provider.budget()


class BudgetRouter:
    """Pick the provider with the most headroom BEFORE calling.

    Args:
        *providers: two or more providers, each an adapter-like object with
            ``backend`` and ``budget``, or a ``(name, provider)`` pair.
        key: score a :class:`Budget`; the highest score wins. Defaults to
            ``remaining_tokens``.
        on_route: optional callback ``(name, budget) -> None`` fired when a
            provider is chosen — for logging which window a call landed in.
        clock: monotonic time source (injectable for tests).
        default_cooldown_s: fallback park time when a 429 gives no reset hint.
    """

    def __init__(
        self,
        *providers: ProviderArg,
        key: Optional[Callable[[Budget], float]] = None,
        on_route: Optional[Callable[[str, Budget], None]] = None,
        clock: Callable[[], float] = time.monotonic,
        default_cooldown_s: float = _DEFAULT_COOLDOWN_S,
    ) -> None:
        if len(providers) < 1:
            raise ValueError("BudgetRouter needs at least one provider")
        self.routes: List[_Route] = []
        for i, p in enumerate(providers):
            if isinstance(p, tuple) and len(p) == 2 and isinstance(p[0], str):
                name, provider = p
            else:
                name, provider = getattr(p, "model", f"provider{i}"), p
            self.routes.append(_Route(str(name), provider))
        self.key = key if key is not None else (lambda b: float(b.remaining_tokens))
        self.on_route = on_route
        self._clock = clock
        self.default_cooldown_s = default_cooldown_s
        self.last_route: Optional[str] = None   # who served the last call

    # -- ranking ----------------------------------------------------------------

    def _ranked(self) -> List[Tuple[float, _Route, Budget]]:
        """Providers not in cooldown and with a readable budget, best first.

        A provider whose ``budget()`` raises :class:`TelemetryError` (no
        headers, no fallback set) is simply skipped — the router can't reason
        about a window it can't see.
        """
        now = self._clock()
        scored: List[Tuple[float, _Route, Budget]] = []
        for r in self.routes:
            if r.cooldown_until > now:
                continue
            try:
                b = r.budget()
            except TelemetryError:
                continue
            scored.append((self.key(b), r, b))
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored

    def _park(self, route: _Route, retry_after: Optional[float]) -> None:
        wait = retry_after
        if wait is None:
            try:
                wait = route.budget().reset_seconds
            except TelemetryError:
                wait = None
        route.cooldown_until = self._clock() + (wait or self.default_cooldown_s)

    # -- the adapter-shaped callables the scheduler wants -----------------------

    def budget(self) -> Budget:
        """The budget of the provider the NEXT call would go to.

        The scheduler makes its predictive decision against this — so it plans
        around the window with the most room, exactly the one ``backend`` will
        use. Raises :class:`TelemetryError` if no provider is currently
        readable and available.
        """
        ranked = self._ranked()
        if not ranked:
            raise TelemetryError(
                "No provider is available: all are in cooldown or reporting "
                "no telemetry. Set fallback_remaining on an adapter to proceed."
            )
        return ranked[0][2]

    def backend(self, messages: List[Dict[str, str]]) -> Tuple[str, int]:
        """Route the call to the highest-headroom provider, descending on failure.

        Providers are tried best-margin first; a rate-limited or transiently
        failing one is parked (cooldown) and the call falls through to the next
        best. A non-retriable :class:`BackendError` (a 4xx — a broken request)
        is raised immediately: no other provider will fix it.
        """
        last_exc: Optional[Exception] = None
        for _score, route, b in self._ranked():
            try:
                reply, used = route.backend(messages)
                self.last_route = route.name
                if self.on_route is not None:
                    self.on_route(route.name, b)
                return reply, used
            except RateLimitHit as exc:
                self._park(route, exc.retry_after)
                last_exc = exc
            except BackendError as exc:
                if not exc.retriable:
                    raise               # broken request: switching won't help
                self._park(route, None)
                last_exc = exc
        if last_exc is not None:
            raise last_exc
        raise TelemetryError(
            "No provider available to route to (all in cooldown or blind to "
            "telemetry)."
        )
