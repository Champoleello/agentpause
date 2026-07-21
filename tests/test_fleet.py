"""Behavioral contract for AgentFleet: the router+coordinator+estimator facade.

The README's "Scaling up" section shows the six manual pieces required to
combine BudgetRouter, MultiAgentCoordinator, and an Estimator today. AgentFleet
collapses construction to one constructor call (plus an optional .register())
and the per-step request/check/call/complete/record dance to one .call().
All offline: providers are the same tiny FakeProvider fakes test_router.py
uses (.backend + .budget, no network), injected clock.
"""

import pytest

from agentpause import AgentFleet, Estimator, TelemetryError
from agentpause.risk import Budget


class FakeProvider:
    """Minimal adapter-shaped stand-in: a fixed budget and a scripted backend.

    Same shape as tests/test_router.py's FakeProvider, copied here so this
    test file reads standalone.
    """

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


class FakeEstimator:
    """Bare-bones estimator-shaped fake for the estimator_factory override test."""

    def __init__(self):
        self.samples = 0

    def estimate(self, input_tokens):
        return 500

    def sigma(self, fallback_estimate):
        return 0.0

    def record(self, input_tokens, realized):
        self.samples += 1


MSG = [{"role": "user", "content": "hi"}]


def test_needs_at_least_one_provider():
    with pytest.raises(Exception):
        AgentFleet([])


def test_call_routes_to_the_richer_provider_and_returns_continue():
    lean = FakeProvider("lean", remaining=500)
    rich = FakeProvider("rich", remaining=9000)
    fleet = AgentFleet([lean, rich])
    fleet.register("agent1")
    d, reply, used = fleet.call("agent1", MSG)
    assert d.action == "continue"
    assert reply == "reply from rich"
    assert used == 10
    assert rich.calls == 1 and lean.calls == 0
    assert fleet.router.last_route == "rich"


def test_second_agent_blocked_by_first_agents_inflight_reservation():
    provider = FakeProvider("solo", remaining=2000)
    fleet = AgentFleet([provider])
    # "A" is mid-flight: it already reserved a big chunk of the shared pool
    # directly through the coordinator, without calling .complete() yet.
    d1 = fleet.coordinator.request("A", estimated=1500, sigma=0.0)
    assert d1.action == "continue"
    # only ~500 tokens are genuinely free now; B's estimate (~931 base + tool
    # overhead, plus k*sigma margin) does not fit in what's left
    d2, reply, used = fleet.call("B", MSG)
    assert d2.action != "continue"
    assert reply is None and used is None
    assert provider.calls == 0     # B's call never reached the provider


def test_estimators_stay_isolated_per_agent():
    a_provider = FakeProvider("a-provider", remaining=9000)
    b_provider = FakeProvider("b-provider", remaining=9000)
    fleet = AgentFleet([a_provider, b_provider])
    for _ in range(3):
        d, reply, used = fleet.call("A", MSG)
        assert d.action == "continue"
    assert fleet.estimator_for("A").samples == 3
    assert fleet.estimator_for("B").samples == 0   # no trace of A's history


def test_unregistered_agent_auto_registers_and_works():
    rich = FakeProvider("rich", remaining=9000)
    fleet = AgentFleet([rich])
    d, reply, used = fleet.call("newcomer", MSG)   # never .register()-ed
    assert d.action == "continue"
    assert reply == "reply from rich"


def test_reregistering_does_not_reset_learned_estimator_history():
    rich = FakeProvider("rich", remaining=9000)
    fleet = AgentFleet([rich], agents=["A"])
    for _ in range(2):
        fleet.call("A", MSG)
    samples_before = fleet.estimator_for("A").samples
    assert samples_before > 0
    fleet.register("A", priority=5)                # re-register, new priority
    assert fleet.estimator_for("A").samples == samples_before


def test_estimator_factory_override_is_used_per_agent():
    rich = FakeProvider("rich", remaining=9000)
    fleet = AgentFleet([rich], estimator_factory=FakeEstimator)
    est_a = fleet.estimator_for("A")
    est_b = fleet.estimator_for("B")
    assert isinstance(est_a, FakeEstimator)
    assert isinstance(est_b, FakeEstimator)
    assert est_a is not est_b                      # still isolated per agent
    assert not isinstance(est_a, Estimator)


def test_router_and_coordinator_are_the_live_objects_not_copies():
    lean = FakeProvider("lean", remaining=500)
    rich = FakeProvider("rich", remaining=9000)
    fleet = AgentFleet([lean, rich])
    assert fleet.router.last_route is None
    fleet.call("agent1", MSG)
    assert fleet.router.last_route == "rich"        # the real router updated
    # power-user access: arbitrate is the real coordinator's method
    out = fleet.coordinator.arbitrate([("agent2", 100, 0.0)])
    assert "agent2" in out


def test_non_continue_decision_calls_no_provider_and_records_nothing():
    lean = FakeProvider("lean", remaining=50)
    rich = FakeProvider("rich", remaining=50)
    fleet = AgentFleet([lean, rich])
    d, reply, used = fleet.call("agent1", MSG)
    assert d.action != "continue"
    assert reply is None and used is None
    assert lean.calls == 0 and rich.calls == 0
    assert fleet.estimator_for("agent1").samples == 0
