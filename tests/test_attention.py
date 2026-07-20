"""Behavioral contract for F11.3: the human in the loop as a budgeted resource.

An agent using the "ask" overflow policy (§8.6) must treat a human exactly
like any other rate-limited channel: only so many interruptions per window,
manual presence/absence overrides, and -- via `as_budget()` -- the same
three-valued `decide()` rule (continue / wait / checkpoint) that governs
provider TPM/RPM.
"""

import pytest

from agentpause.attention import HumanAttentionBudget
from agentpause.risk import decide


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


# -- window refill math ----------------------------------------------------


def test_ready_under_the_limit_then_exhausted():
    clock = FakeClock()
    a = HumanAttentionBudget(max_asks=3, window_s=3600, clock=clock)
    for _ in range(3):
        assert a.ready()
        assert a.wait_seconds() == 0.0
        a.record_ask()
    assert not a.ready()
    assert a.remaining() == 0


def test_wait_seconds_until_the_oldest_ask_leaves_the_window():
    clock = FakeClock()
    a = HumanAttentionBudget(max_asks=2, window_s=3600, clock=clock)
    a.record_ask()                # t=0
    clock.now = 100.0
    a.record_ask()                # t=100
    clock.now = 500.0
    assert not a.ready()
    assert a.wait_seconds() == pytest.approx(3100.0)   # frees at t=3600
    clock.now = 3601.0
    assert a.ready()
    assert a.remaining() == 1


# -- manual override -------------------------------------------------------


def test_away_zeroes_the_budget_regardless_of_window():
    clock = FakeClock()
    a = HumanAttentionBudget(max_asks=5, window_s=3600, clock=clock)
    assert a.ready()             # window is wide open
    a.away()
    assert a.is_away
    assert not a.ready()
    assert a.wait_seconds() is None   # no ETA: unknowable


def test_available_clears_the_override():
    clock = FakeClock()
    a = HumanAttentionBudget(max_asks=5, window_s=3600, clock=clock)
    a.away()
    assert not a.ready()
    a.available()
    assert not a.is_away
    assert a.ready()


def test_away_with_until_reports_a_bounded_wait():
    clock = FakeClock()
    a = HumanAttentionBudget(max_asks=5, window_s=3600, clock=clock)
    a.away(until_s=120.0)
    assert a.is_away
    assert a.wait_seconds() == pytest.approx(120.0)
    clock.now = 60.0
    assert a.wait_seconds() == pytest.approx(60.0)


def test_away_until_auto_expires_on_the_injected_clock():
    clock = FakeClock()
    a = HumanAttentionBudget(max_asks=5, window_s=3600, clock=clock)
    a.away(until_s=100.0)
    assert not a.ready()
    clock.now = 99.0
    assert a.is_away
    assert not a.ready()
    clock.now = 100.0
    assert not a.is_away             # expired exactly at the deadline
    assert a.ready()


# -- as_budget() composing with decide() -----------------------------------


def test_as_budget_ready_continues_when_tokens_abound():
    clock = FakeClock()
    a = HumanAttentionBudget(max_asks=3, window_s=3600, clock=clock)
    decision = decide(a.as_budget(), estimated=1, sigma=0.0)
    assert decision.action == "continue"


def test_as_budget_exhausted_with_known_next_slot_waits():
    clock = FakeClock()
    a = HumanAttentionBudget(max_asks=1, window_s=30, clock=clock)
    a.record_ask()                # t=0, window exhausted, frees at t=30
    clock.now = 20.0               # 10s left until the slot frees
    decision = decide(a.as_budget(), estimated=1, sigma=0.0, wait_threshold_s=15.0)
    assert decision.action == "wait"
    assert decision.wait_seconds == pytest.approx(10.0)


def test_as_budget_away_without_expiry_checkpoints():
    clock = FakeClock()
    a = HumanAttentionBudget(max_asks=3, window_s=3600, clock=clock)
    a.away()                       # gone, no ETA
    decision = decide(a.as_budget(), estimated=1, sigma=0.0)
    assert decision.action == "checkpoint"


def test_as_budget_away_with_expiry_waits_when_within_threshold():
    clock = FakeClock()
    a = HumanAttentionBudget(max_asks=3, window_s=3600, clock=clock)
    a.away(until_s=5.0)
    decision = decide(a.as_budget(), estimated=1, sigma=0.0, wait_threshold_s=15.0)
    assert decision.action == "wait"
    assert decision.wait_seconds == pytest.approx(5.0)
