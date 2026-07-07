"""Model fallback chain: route around a rate-limited or failing model.

The escalation ladder of the accompanying research (§5.3, "model switch")
as a composable piece: wrap any ordered list of backends and the chain tries
them in sequence when the current one is rate-limited or fails transiently.

    from agentpause import FallbackBackend, PredictiveScheduler

    chain = FallbackBackend(
        adapter_opus.backend,      # primary
        adapter_haiku.backend,     # cheaper fallback
        on_fallback=lambda i, exc: log.warning("switched to backend %d: %s", i, exc),
    )
    sched = PredictiveScheduler(backend=chain, telemetry=adapter_opus.budget)

Only rate limits and *retriable* failures trigger the switch — a 400-style
error means the request itself is broken, and no other model will fix it.

Behavioral-consistency caveat (the "Sierra lesson"): different models produce
differently-shaped outputs. The chain exposes ``last_index`` and the
``on_fallback`` callback so downstream code can validate outputs that came
from a fallback model instead of trusting them blindly.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from .errors import BackendError, RateLimitHit

__all__ = ["FallbackBackend"]

Backend = Callable[[List[Dict[str, str]]], Tuple[str, int]]


class FallbackBackend:
    """An ordered chain of backends, tried in sequence on transient failure.

    Args:
        *backends: two or more ``messages -> (reply, tokens_used)`` callables,
            primary first, cheapest last.
        on_fallback: optional callback ``(index, exception) -> None`` invoked
            when the chain moves past a failing backend — use it for logging
            or to flag outputs for extra validation.
    """

    def __init__(
        self,
        *backends: Backend,
        on_fallback: Optional[Callable[[int, Exception], None]] = None,
    ) -> None:
        if len(backends) < 1:
            raise ValueError("FallbackBackend needs at least one backend")
        self.backends = list(backends)
        self.on_fallback = on_fallback
        self.last_index: Optional[int] = None   # which backend served the last call

    def __call__(self, messages: List[Dict[str, str]]) -> Tuple[str, int]:
        last_exc: Optional[Exception] = None
        for i, backend in enumerate(self.backends):
            try:
                result = backend(messages)
                self.last_index = i
                return result
            except RateLimitHit as exc:
                last_exc = exc
            except BackendError as exc:
                if not exc.retriable:
                    raise               # request problem: no model will fix it
                last_exc = exc
            if self.on_fallback is not None and i + 1 < len(self.backends):
                self.on_fallback(i + 1, last_exc)
        assert last_exc is not None
        raise last_exc
