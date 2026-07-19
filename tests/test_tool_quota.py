"""Behavioral contract for F9.4: tools as budgeted resources.

LLM calls are not the only rate-limited resource an agent consumes: search
APIs, scrapers, geocoders all have quotas of their own — and they rarely
send telemetry headers. `ToolQuota` is a client-side sliding-window budget:
the agent asks BEFORE calling (predictive, like everything here), learns how
long until a slot frees up, and never slams a tool into its limit.
"""

import pytest

from agentpause.tools import ToolQuota


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_ready_under_the_limit():
    clock = FakeClock()
    q = ToolQuota(calls_per_window=3, window_s=60, clock=clock)
    for _ in range(3):
        assert q.ready()
        assert q.wait_seconds() == 0.0
        q.record()
    assert not q.ready()


def test_wait_until_the_oldest_call_leaves_the_window():
    clock = FakeClock()
    q = ToolQuota(calls_per_window=2, window_s=60, clock=clock)
    q.record()               # t=0
    clock.now = 10.0
    q.record()               # t=10
    clock.now = 25.0
    assert not q.ready()
    assert q.wait_seconds() == pytest.approx(35.0)   # slot frees at t=60
    clock.now = 61.0
    assert q.ready()         # the t=0 call has left the window


def test_acquire_sleeps_exactly_what_is_needed_then_records():
    clock = FakeClock()
    sleeps = []

    def fake_sleep(s):
        sleeps.append(s)
        clock.now += s

    q = ToolQuota(calls_per_window=1, window_s=30, clock=clock)
    q.acquire(sleep_fn=fake_sleep)       # first call: no wait
    q.acquire(sleep_fn=fake_sleep)       # second: must wait out the window
    assert sleeps == [pytest.approx(30.0)]
    assert not q.ready()                 # the second call is now recorded


def test_guard_a_tool_function():
    clock = FakeClock()
    calls = []
    q = ToolQuota(calls_per_window=2, window_s=60, clock=clock)

    def search(query):
        calls.append(query)
        return f"results for {query}"

    guarded = q.guard(search, sleep_fn=lambda s: clock.__setattr__("now", clock.now + s))
    assert guarded("a") == "results for a"
    assert guarded("b") == "results for b"
    guarded("c")                          # third call waits for a slot, then runs
    assert calls == ["a", "b", "c"]
