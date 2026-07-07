"""Behavioral contract for B2 (money budget as a hard rule) and B3 (hooks).

B2 — the monetary dimension graduates from diagnostic (RiskModel w3) to hard
constraint: if the estimated next step would overrun the remaining money
budget, the decision is ``checkpoint`` — and never ``wait``, because unlike a
rate-limit window, money does not reset by waiting.

B3 — observability hooks: tiny injectable callbacks so users can wire their
own logging/metrics (Prometheus, Langfuse, ...) without agentpause taking on
any dependency.
"""

import pytest

from agentpause import PredictiveScheduler, RateLimitHit, RetryPolicy, StateStore


def make_backend(tokens=1000):
    def backend(messages):
        return "ok", tokens
    return backend


# -- B2: money as a hard constraint ---------------------------------------------

def test_money_budget_blocks_when_step_would_overrun(tmp_path):
    sched = PredictiveScheduler(
        backend=make_backend(),
        telemetry=lambda: 10_000_000,        # tokens are NOT the problem
        store=StateStore(str(tmp_path)),
        price_per_1k_tokens=10.0,            # expensive model
        money_budget=0.50,                   # ...and 50 cents left
    )
    with sched.session("t") as s:
        s.add_user("a question " * 50)
        d = s.next_action()
        assert d.action == "checkpoint"      # est. >> 50 tokens → > 0.50 €
        assert s.should_suspend() is True


def test_money_never_waits(tmp_path):
    """Even with an imminent window reset, no money → checkpoint, not wait."""
    from agentpause import Budget
    sched = PredictiveScheduler(
        backend=make_backend(),
        telemetry=lambda: Budget(remaining_tokens=10_000_000, reset_seconds=2.0),
        store=StateStore(str(tmp_path)),
        price_per_1k_tokens=10.0,
        money_budget=0.01,
    )
    with sched.session("t") as s:
        s.add_user("q " * 100)
        assert s.next_action().action == "checkpoint"


def test_spending_is_tracked_and_budget_releases_steps(tmp_path):
    sched = PredictiveScheduler(
        backend=make_backend(tokens=1000),
        telemetry=lambda: 10_000_000,
        store=StateStore(str(tmp_path)),
        price_per_1k_tokens=1.0,             # 1 € per 1k tokens
        money_budget=10.0,
    )
    with sched.session("t") as s:
        s.add_user("q")
        s.call()
        assert sched.money_spent == pytest.approx(1.0)   # 1000 tok × 1€/1k
        assert sched.money_remaining == pytest.approx(9.0)


def test_no_money_budget_means_no_constraint(tmp_path):
    sched = PredictiveScheduler(
        backend=make_backend(),
        telemetry=lambda: 10_000_000,
        store=StateStore(str(tmp_path)),
    )
    with sched.session("t") as s:
        s.add_user("q " * 100)
        assert s.next_action().action == "continue"


# -- B3: observability hooks -------------------------------------------------------

def test_hooks_fire_on_rate_limit_and_retry(tmp_path):
    events = []
    calls = {"n": 0}

    def flaky(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RateLimitHit()
        return "ok", 100

    sched = PredictiveScheduler(
        backend=flaky,
        telemetry=lambda: 100_000,
        store=StateStore(str(tmp_path)),
        retry=RetryPolicy(jitter=0.0, sleep_fn=lambda s: None),
        on_event=lambda name, info: events.append((name, info)),
    )
    with sched.session("t") as s:
        s.add_user("q")
        s.call()
    names = [e[0] for e in events]
    assert "rate_limit_hit" in names
    assert "retry" in names
    assert "step_completed" in names


def test_hooks_fire_on_decisions(tmp_path):
    events = []
    sched = PredictiveScheduler(
        backend=make_backend(),
        telemetry=lambda: 100,               # tiny budget
        store=StateStore(str(tmp_path)),
        on_event=lambda name, info: events.append((name, info)),
    )
    with sched.session("t") as s:
        s.add_user("q " * 100)
        d = s.next_action()
        assert d.action == "checkpoint"
    decisions = [i for n, i in events if n == "decision"]
    assert decisions and decisions[-1]["action"] == "checkpoint"


def test_broken_hook_never_breaks_the_run(tmp_path):
    def bad_hook(name, info):
        raise RuntimeError("observability must never take down the agent")

    sched = PredictiveScheduler(
        backend=make_backend(),
        telemetry=lambda: 100_000,
        store=StateStore(str(tmp_path)),
        on_event=bad_hook,
    )
    with sched.session("t") as s:
        s.add_user("q")
        assert s.call() == "ok"              # hook exploded, run unaffected
