"""Behavioral contract for the model fallback chain (B1).

Pattern from production agent systems (and §5.3 of the research): when the
primary model is rate-limited or its infrastructure fails transiently, route
the call to the next backend in the chain (typically a cheaper model) instead
of failing the step.

Care points encoded here:
* only RATE-LIMIT and RETRIABLE failures trigger fallback — a 400 means the
  request is broken and no other model will fix it;
* the chain must be observable (which backend actually served?) because
  different models behave differently (the Sierra lesson): downstream code
  may want to validate outputs produced by a fallback model.
"""

import pytest

from agentpause import (
    BackendError,
    FallbackBackend,
    PredictiveScheduler,
    RateLimitHit,
    StateStore,
)


def ok_backend(reply="ok", tokens=100):
    def backend(messages):
        return reply, tokens
    return backend


def limited_backend():
    def backend(messages):
        raise RateLimitHit()
    return backend


def broken_backend(retriable):
    def backend(messages):
        raise BackendError("boom", retriable=retriable)
    return backend


def test_uses_primary_when_healthy():
    chain = FallbackBackend(ok_backend("primary"), ok_backend("cheap"))
    reply, _ = chain([])
    assert reply == "primary"
    assert chain.last_index == 0


def test_falls_back_on_rate_limit():
    chain = FallbackBackend(limited_backend(), ok_backend("cheap"))
    reply, _ = chain([])
    assert reply == "cheap"
    assert chain.last_index == 1


def test_falls_back_on_retriable_failure():
    chain = FallbackBackend(broken_backend(retriable=True), ok_backend("cheap"))
    assert chain([])[0] == "cheap"


def test_non_retriable_error_does_not_fall_back():
    chain = FallbackBackend(broken_backend(retriable=False), ok_backend("cheap"))
    with pytest.raises(BackendError):
        chain([])


def test_raises_last_error_when_all_exhausted():
    chain = FallbackBackend(limited_backend(), limited_backend())
    with pytest.raises(RateLimitHit):
        chain([])


def test_on_fallback_callback_reports_the_switch():
    events = []
    chain = FallbackBackend(limited_backend(), ok_backend("cheap"),
                            on_fallback=lambda i, exc: events.append((i, type(exc).__name__)))
    chain([])
    assert events == [(1, "RateLimitHit")]


def test_scheduler_integration(tmp_path):
    """The chain is a plain Backend: drop it into the scheduler as-is."""
    chain = FallbackBackend(limited_backend(), ok_backend("cheap", tokens=50))
    sched = PredictiveScheduler(backend=chain, telemetry=lambda: 100_000,
                                store=StateStore(str(tmp_path)))
    with sched.session("t") as s:
        s.add_user("q")
        assert s.call() == "cheap"
        assert s.total_tokens_used == 50
