"""Behavioral contract for BudgetRouter (F9.1): predictive multi-provider routing.

Unlike FallbackBackend (react to a failure, then switch), the router reads
every provider's budget FIRST and sends the call to the one with the most
headroom — before anyone hits a 429. All offline: providers are tiny fakes
exposing the adapter shape (.backend + .budget), no network, injected clock.
"""

import pytest

from agentpause import BudgetRouter, Budget, RateLimitHit, TelemetryError
from agentpause.errors import BackendError


class FakeProvider:
    """Minimal adapter-shaped stand-in: a fixed budget and a scripted backend."""

    def __init__(self, name, remaining, reset_seconds=30.0, raises=None, reply=None):
        self.model = name
        self._budget = Budget(remaining_tokens=remaining, reset_seconds=reset_seconds)
        self.raises = raises          # exception to throw on backend(), or None
        self.reply = reply or f"reply from {name}"
        self.calls = 0

    def budget(self):
        if self._budget is None:
            raise TelemetryError("blind")
        return self._budget

    def backend(self, messages):
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        return self.reply, 10


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


MSG = [{"role": "user", "content": "hi"}]


def test_routes_to_the_provider_with_the_most_headroom():
    lean = FakeProvider("lean", remaining=500)
    rich = FakeProvider("rich", remaining=9000)
    router = BudgetRouter(lean, rich)
    reply, used = router.backend(MSG)
    assert reply == "reply from rich"       # picked BEFORE any failure
    assert router.last_route == "rich"
    assert rich.calls == 1 and lean.calls == 0


def test_budget_reports_the_window_the_next_call_will_use():
    lean = FakeProvider("lean", remaining=500)
    rich = FakeProvider("rich", remaining=9000)
    router = BudgetRouter(lean, rich)
    # the scheduler plans against the best available window, not a sum
    assert router.budget().remaining_tokens == 9000


def test_falls_through_to_next_best_on_rate_limit():
    clock = Clock()
    hot = FakeProvider("hot", remaining=9000, raises=RateLimitHit(retry_after=30.0))
    cool = FakeProvider("cool", remaining=4000)
    router = BudgetRouter(hot, cool, clock=clock)
    reply, _ = router.backend(MSG)
    assert reply == "reply from cool"       # hot was best but 429'd → next best
    assert hot.calls == 1 and cool.calls == 1
    assert router.last_route == "cool"


def test_rate_limited_provider_is_parked_until_reset():
    clock = Clock()
    hot = FakeProvider("hot", remaining=9000, raises=RateLimitHit(retry_after=30.0))
    cool = FakeProvider("cool", remaining=4000)
    router = BudgetRouter(hot, cool, clock=clock)
    router.backend(MSG)                       # hot parked for 30s
    # while parked, hot is invisible even though its budget looks bigger
    assert router.budget().remaining_tokens == 4000
    clock.t += 31.0                           # cooldown elapsed
    assert router.budget().remaining_tokens == 9000


def test_non_retriable_error_is_not_rerouted():
    bad = FakeProvider("bad", remaining=9000,
                       raises=BackendError("400 bad request", retriable=False))
    other = FakeProvider("other", remaining=100)
    router = BudgetRouter(bad, other)
    with pytest.raises(BackendError):
        router.backend(MSG)
    assert other.calls == 0                   # a broken request no model fixes


def test_blind_provider_is_skipped():
    blind = FakeProvider("blind", remaining=0)
    blind._budget = None                      # budget() will raise TelemetryError
    seeing = FakeProvider("seeing", remaining=1000)
    router = BudgetRouter(blind, seeing)
    reply, _ = router.backend(MSG)
    assert reply == "reply from seeing"


def test_all_exhausted_raises_last_rate_limit():
    clock = Clock()
    a = FakeProvider("a", remaining=9000, raises=RateLimitHit(retry_after=5.0))
    b = FakeProvider("b", remaining=8000, raises=RateLimitHit(retry_after=5.0))
    router = BudgetRouter(a, b, clock=clock)
    with pytest.raises(RateLimitHit):
        router.backend(MSG)


def test_custom_key_selects_by_fraction_of_window():
    # big absolute budget but nearly drained window vs. small but fresh window
    big = FakeProvider("big", remaining=1000)
    big._budget = Budget(remaining_tokens=1000, limit_tokens=100000)   # 1%
    small = FakeProvider("small", remaining=800)
    small._budget = Budget(remaining_tokens=800, limit_tokens=1000)    # 80%
    router = BudgetRouter(big, small,
                          key=lambda b: b.remaining_tokens / (b.limit_tokens or 1))
    reply, _ = router.backend(MSG)
    assert reply == "reply from small"        # fuller window wins on fraction


def test_on_route_callback_reports_choice():
    seen = []
    rich = FakeProvider("rich", remaining=9000)
    router = BudgetRouter(FakeProvider("lean", remaining=100), rich,
                          on_route=lambda name, b: seen.append((name, b.remaining_tokens)))
    router.backend(MSG)
    assert seen == [("rich", 9000)]


def test_needs_at_least_one_provider():
    with pytest.raises(ValueError):
        BudgetRouter()
