"""Latency as a budget dimension: a deadline the decision must respect.

Time is unlike tokens or requests: it never refills by waiting. So when a step
can't finish before the task's deadline, the rule is checkpoint (save state),
never wait. Exercised both at the decide() level and end-to-end through the
scheduler's optional time_budget_s.
"""

import pytest

from agentpause import Budget, PredictiveScheduler, FeatureEstimator
from agentpause.risk import decide


def test_decide_checkpoints_when_step_cannot_meet_deadline():
    b = Budget(remaining_tokens=100000, remaining_seconds=1.0)
    d = decide(b, estimated=1000, sigma=0.0, estimated_latency=3.0)
    assert d.action == "checkpoint"          # 3s step, 1s left → can't finish


def test_decide_continues_when_time_is_enough():
    b = Budget(remaining_tokens=100000, remaining_seconds=10.0)
    d = decide(b, estimated=1000, sigma=0.0, estimated_latency=3.0)
    assert d.action == "continue"


def test_unknown_time_never_blocks():
    # no deadline and/or no latency estimate → dimension is inert
    b = Budget(remaining_tokens=100000)
    assert decide(b, 1000, 0.0, estimated_latency=99.0).action == "continue"
    b2 = Budget(remaining_tokens=100000, remaining_seconds=0.1)
    assert decide(b2, 1000, 0.0, estimated_latency=None).action == "continue"


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_scheduler_time_budget_folds_into_the_decision():
    # a slow-per-step estimator + a tight run deadline → suspend
    clock = FakeClock()
    est = FeatureEstimator(min_samples=3)
    # teach it that steps take ~5 seconds regardless of context
    for _ in range(6):
        est.record(1000, 1500, latency=5.0)

    sched = PredictiveScheduler(
        backend=lambda msgs: ("ok", 10),
        telemetry=lambda: Budget(remaining_tokens=100000),
        estimator=est,
        time_budget_s=3.0,      # only 3s for the whole run
        clock=clock,
    )
    with sched.session("s1") as s:
        s.add_user("hello")
        d = s.next_action()
    # predicted 5s step, 3s budget → the time dimension forces checkpoint
    assert d.action == "checkpoint"


def test_scheduler_without_time_budget_is_unaffected():
    est = FeatureEstimator(min_samples=3)
    for _ in range(6):
        est.record(1000, 1500, latency=5.0)
    sched = PredictiveScheduler(
        backend=lambda msgs: ("ok", 10),
        telemetry=lambda: Budget(remaining_tokens=100000),
        estimator=est,
        # no time_budget_s
    )
    with sched.session("s2") as s:
        s.add_user("hello")
        assert s.next_action().action == "continue"
