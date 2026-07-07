"""Behavioral contract for the lightweight circuit breaker (B4).

When a backend keeps failing, hammering it wastes budget and worsens the
outage (retry storm). The breaker is a small state machine:

    CLOSED --(N consecutive failures)--> OPEN --(cooldown)--> HALF_OPEN
    HALF_OPEN --(probe succeeds)--> CLOSED
    HALF_OPEN --(probe fails)-----> OPEN (cooldown restarts)

Care points: only *infrastructure* failures count (429, retriable 5xx) —
request errors (4xx) never trip the breaker; while OPEN, calls fail fast
with the stored error instead of touching the provider.
"""

import pytest

from agentpause import (
    BackendError,
    CircuitBreaker,
    CircuitOpenError,
    RateLimitHit,
)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def failing(exc_factory):
    def backend(messages):
        raise exc_factory()
    return backend


def ok(messages):
    return "ok", 100


def test_stays_closed_on_success():
    br = CircuitBreaker(ok, failure_threshold=3)
    assert br([])[0] == "ok"
    assert br.state == "closed"


def test_opens_after_consecutive_failures():
    br = CircuitBreaker(failing(RateLimitHit), failure_threshold=3,
                        clock=FakeClock())
    for _ in range(3):
        with pytest.raises(RateLimitHit):
            br([])
    assert br.state == "open"
    with pytest.raises(CircuitOpenError):
        br([])                     # fails fast: the provider is NOT touched


def test_request_errors_never_trip_the_breaker():
    br = CircuitBreaker(failing(lambda: BackendError("400", retriable=False)),
                        failure_threshold=2)
    for _ in range(5):
        with pytest.raises(BackendError):
            br([])
    assert br.state == "closed"


def test_half_open_probe_recovers():
    clock = FakeClock()
    calls = {"fail": True}

    def flaky(messages):
        if calls["fail"]:
            raise RateLimitHit()
        return "ok", 100

    br = CircuitBreaker(flaky, failure_threshold=2, cooldown_s=30.0, clock=clock)
    for _ in range(2):
        with pytest.raises(RateLimitHit):
            br([])
    assert br.state == "open"

    clock.now = 31.0               # cooldown expired → HALF_OPEN probe allowed
    calls["fail"] = False
    assert br([])[0] == "ok"       # probe succeeds
    assert br.state == "closed"


def test_half_open_probe_failure_reopens():
    clock = FakeClock()
    br = CircuitBreaker(failing(RateLimitHit), failure_threshold=1,
                        cooldown_s=10.0, clock=clock)
    with pytest.raises(RateLimitHit):
        br([])
    assert br.state == "open"
    clock.now = 11.0
    with pytest.raises(RateLimitHit):
        br([])                     # the probe itself fails
    assert br.state == "open"
    clock.now = 15.0               # cooldown restarted at t=11 → still open
    with pytest.raises(CircuitOpenError):
        br([])


def test_success_resets_the_failure_count():
    calls = {"n": 0}

    def alternating(messages):
        calls["n"] += 1
        if calls["n"] % 2 == 1:
            raise RateLimitHit()
        return "ok", 100

    br = CircuitBreaker(alternating, failure_threshold=3)
    for _ in range(4):             # fail, ok, fail, ok — never 3 in a row
        try:
            br([])
        except RateLimitHit:
            pass
    assert br.state == "closed"
