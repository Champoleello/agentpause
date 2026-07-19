"""Two adoptions from field practice (Hermes agent #7489/#7490, July 2026).

1. Age-adjusted reset: the telemetry cache may serve a reading up to
   ``max_age_s`` old; its reset countdowns must be advanced by the reading's
   age, or callers wait on seconds that already elapsed.
2. ``rpm_margin``: optional safety slack on the request budget (Hermes pauses
   at 2 remaining; our default stays 0 = historical behavior).
"""

import pytest

from agentpause import PredictiveScheduler
from agentpause.adapters.openai_compat import OpenAICompatAdapter
from agentpause.risk import Budget, decide


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


HEADERS = {
    "x-ratelimit-remaining-tokens": "5000",
    "x-ratelimit-remaining-requests": "12",
    "x-ratelimit-reset-tokens": "30s",
    "x-ratelimit-reset-requests": "8s",
}
BODY = {"choices": [{"message": {"content": "hi"}}], "usage": {"total_tokens": 5}}


def make_adapter(clock):
    def post(url, headers, payload):
        return 200, HEADERS, BODY
    return OpenAICompatAdapter("m", base_url="https://api.example.com/v1",
                               api_key="k", post_fn=post, clock=clock)


# ---------------------------------------------------------------- aging

def test_cached_reset_ages_with_the_clock():
    clock = FakeClock()
    a = make_adapter(clock)
    b0 = a.budget()                      # fresh read at t=100
    assert b0.reset_seconds == pytest.approx(30.0)
    clock.t += 8.0                       # still fresh (max_age_s=10), no ping
    b1 = a.budget()
    assert b1.reset_seconds == pytest.approx(22.0)
    assert b1.reset_requests_seconds == pytest.approx(0.0)   # 8s - 8s, floored


def test_aging_never_goes_negative_and_cache_not_mutated():
    clock = FakeClock()
    a = make_adapter(clock)
    a.budget()
    clock.t += 9.9
    b = a.budget()
    assert b.reset_seconds >= 0.0
    assert b.reset_requests_seconds == 0.0
    # the CACHE keeps the raw capture: a later reader adjusts independently
    assert a._budget.reset_seconds == pytest.approx(30.0)


# ---------------------------------------------------------------- rpm_margin

def _budget(requests):
    return Budget(remaining_tokens=50_000, remaining_requests=requests,
                  reset_seconds=30.0, reset_requests_seconds=5.0)


def test_default_margin_keeps_historical_behavior():
    d = decide(_budget(requests=1), estimated=100, sigma=0.0)
    assert d.action == "continue"


def test_margin_pauses_before_the_last_request():
    d = decide(_budget(requests=2), estimated=100, sigma=0.0, rpm_margin=2)
    assert d.action in ("wait", "checkpoint")
    if d.action == "wait":
        # the binding constraint is the REQUEST clock, not the token clock
        assert d.wait_seconds is not None and d.wait_seconds <= 10.0


def test_margin_scheduler_wiring():
    sched = PredictiveScheduler(backend=lambda m: ("ok", 10),
                                telemetry=lambda: 50_000, rpm_margin=3)
    assert sched.rpm_margin == 3


def test_unknown_rpm_is_never_blocked_by_margin():
    b = Budget(remaining_tokens=50_000, remaining_requests=None,
               reset_seconds=30.0)
    d = decide(b, estimated=100, sigma=0.0, rpm_margin=5)
    assert d.action == "continue"
