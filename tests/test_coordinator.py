"""Behavioral contract for MultiAgentCoordinator (D5): one shared window, many agents.

The value is preventing cross-agent overcommit: agents can't each independently
believe they own the whole window and collectively blow it. Reservations make
the shared pool visible; arbitration resolves contention by priority then
fairness. All offline: telemetry is a fake returning a fixed shared Budget.
"""

import pytest

from agentpause import MultiAgentCoordinator, Budget


class SharedWindow:
    """A mutable shared budget the coordinator reads through telemetry()."""

    def __init__(self, remaining, limit=None, reset_seconds=60.0):
        self.b = Budget(remaining_tokens=remaining, limit_tokens=limit,
                        reset_seconds=reset_seconds)

    def __call__(self):
        return self.b


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_reservation_hides_headroom_from_the_other_agent():
    win = SharedWindow(remaining=2000)
    coord = MultiAgentCoordinator(telemetry=win, k=0.0)
    d1 = coord.request("a", estimated=1500)
    assert d1.action == "continue"              # a fits and reserves 1500
    # only 500 left in the shared pool → b's 1500 no longer fits
    d2 = coord.request("b", estimated=1500)
    assert d2.action != "continue"
    assert coord.reserved == 1500               # only a holds


def test_completing_frees_the_pool_for_others():
    win = SharedWindow(remaining=2000)
    coord = MultiAgentCoordinator(telemetry=win, k=0.0)
    coord.request("a", estimated=1500)
    coord.complete("a")
    assert coord.reserved == 0
    d = coord.request("b", estimated=1500)
    assert d.action == "continue"               # pool freed → b now fits


def test_margin_uses_k_sigma():
    win = SharedWindow(remaining=1000)
    coord = MultiAgentCoordinator(telemetry=win, k=2.0)
    d = coord.request("a", estimated=800, sigma=150)   # needs 800 + 2*150 = 1100
    assert d.action != "continue"                       # 1100 > 1000


def test_single_agent_rerequest_not_blocked_by_its_own_hold():
    win = SharedWindow(remaining=2000)
    coord = MultiAgentCoordinator(telemetry=win, k=0.0)
    coord.request("a", estimated=1500)          # holds 1500
    # a asks again (e.g. re-decides before calling): its OWN hold must not
    # count against it, else it could never proceed
    d = coord.request("a", estimated=1500)
    assert d.action == "continue"


def test_arbitrate_serves_higher_priority_first():
    win = SharedWindow(remaining=1500)          # fits only one 1200-token call
    coord = MultiAgentCoordinator(telemetry=win, k=0.0)
    coord.register("low", priority=0)
    coord.register("high", priority=5)
    out = coord.arbitrate([("low", 1200, 0.0), ("high", 1200, 0.0)])
    assert out["high"].action == "continue"
    assert out["low"].action != "continue"      # window spent by high
    assert coord.reserved == 1200               # only high holds


def test_arbitrate_breaks_priority_ties_by_longest_waiting():
    clock = Clock()
    win = SharedWindow(remaining=100, reset_seconds=60.0)   # too small for anyone
    coord = MultiAgentCoordinator(telemetry=win, k=0.0, clock=clock)
    coord.register("x", priority=1)
    coord.register("y", priority=1)
    # both get denied (pool too small) but at different times → distinct waits
    coord.request("y", estimated=1200)          # denied at t=1000 → waiting_since y
    assert coord.reserved == 0                   # nothing held
    clock.t = 1005.0
    coord.request("x", estimated=1200)          # denied at t=1005 → waiting_since x
    # window resets: now it fits exactly one 1200-token call
    win.b = Budget(remaining_tokens=1500, reset_seconds=60.0)
    out = coord.arbitrate([("x", 1200, 0.0), ("y", 1200, 0.0)])
    assert out["y"].action == "continue"        # same priority, waited longest → first
    assert out["x"].action != "continue"


def test_arbitrate_grants_everyone_when_the_pool_is_big_enough():
    win = SharedWindow(remaining=10000)
    coord = MultiAgentCoordinator(telemetry=win, k=0.0)
    out = coord.arbitrate([("a", 1000, 0.0), ("b", 1000, 0.0), ("c", 1000, 0.0)])
    assert all(d.action == "continue" for d in out.values())
    assert coord.reserved == 3000


def test_unregistered_agent_is_auto_registered():
    win = SharedWindow(remaining=5000)
    coord = MultiAgentCoordinator(telemetry=win, k=0.0)
    d = coord.request("newcomer", estimated=1000)   # never register()ed
    assert d.action == "continue"


def test_wait_decision_when_window_resets_soon():
    # pool too small now, but the shared window resets within the threshold
    win = SharedWindow(remaining=100, limit=2000, reset_seconds=5.0)
    coord = MultiAgentCoordinator(telemetry=win, k=0.0, wait_threshold_s=15.0)
    d = coord.request("a", estimated=1500)
    assert d.action == "wait"
    assert d.wait_seconds is not None
