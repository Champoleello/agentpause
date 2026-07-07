"""Lightweight circuit breaker: stop hammering a failing provider.

Retries handle *transient* glitches; the breaker handles *persistent* ones.
After ``failure_threshold`` consecutive infrastructure failures the circuit
opens: calls fail fast (raising :class:`CircuitOpenError`) without touching
the provider, until a cooldown expires and a single probe call is allowed
through (half-open). A successful probe closes the circuit again.

Wrap any backend — it stays a plain ``messages -> (reply, tokens)`` callable:

    from agentpause import CircuitBreaker

    guarded = CircuitBreaker(adapter.backend, failure_threshold=5, cooldown_s=300)
    sched = PredictiveScheduler(backend=guarded, telemetry=adapter.budget)

Only infrastructure failures count (rate limits, retriable 5xx): a 4xx means
the request is broken — no amount of cooling down will fix it, so it never
trips the breaker.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Tuple

from .errors import AgentPauseError, BackendError, RateLimitHit

__all__ = ["CircuitBreaker", "CircuitOpenError"]

Backend = Callable[[List[Dict[str, str]]], Tuple[str, int]]


class CircuitOpenError(AgentPauseError):
    """The circuit is open: the provider is presumed down, call not attempted."""


class CircuitBreaker:
    """Wrap a backend with a CLOSED → OPEN → HALF_OPEN state machine.

    Args:
        backend: the callable to protect.
        failure_threshold: consecutive infrastructure failures that open
            the circuit.
        cooldown_s: how long the circuit stays open before allowing a probe.
        clock: time source (injectable for tests).
    """

    def __init__(
        self,
        backend: Backend,
        failure_threshold: int = 5,
        cooldown_s: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.backend = backend
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self._clock = clock
        self._failures = 0
        self._opened_at: Optional[float] = None

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "closed"
        if self._clock() - self._opened_at >= self.cooldown_s:
            return "half_open"
        return "open"

    def __call__(self, messages: List[Dict[str, str]]) -> Tuple[str, int]:
        state = self.state
        if state == "open":
            raise CircuitOpenError(
                f"circuit open after {self._failures} consecutive failures; "
                f"retry after the cooldown ({self.cooldown_s:.0f}s)"
            )
        try:
            result = self.backend(messages)
        except (RateLimitHit, BackendError) as exc:
            if isinstance(exc, BackendError) and not exc.retriable:
                raise               # request problem: never trips the breaker
            self._failures += 1
            if state == "half_open" or self._failures >= self.failure_threshold:
                self._opened_at = self._clock()   # (re)open, cooldown restarts
            raise
        self._failures = 0
        self._opened_at = None      # any success closes the circuit
        return result
