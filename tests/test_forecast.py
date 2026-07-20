"""Task-completion forecast (F11.1): simulate the rest of the run, no network.

forecast_run() is exercised as a pure function against the four window shapes
that matter (fits outright, continuous refill, fixed reset, context wall), and
Session.forecast() end-to-end offline to prove it is a pure read: same answer
twice, no budget consumed, no estimator or session state touched.
"""

import copy

import pytest

from agentpause import Budget, Forecast, PredictiveScheduler, forecast_run


# ---------------------------------------------------------------- pure function

def test_everything_fits_in_one_window():
    b = Budget(remaining_tokens=10000, reset_seconds=60.0, limit_tokens=10000)
    f = forecast_run(b, estimated_per_step=500, sigma=50.0, steps=5, k=2.0,
                     price_per_1k_tokens=0.01)
    per_step = 500 + 2.0 * 50.0                  # 600
    assert f.steps == 5
    assert f.waits == 0 and f.suspensions == 0
    assert f.waiting_s == 0.0
    assert f.total_tokens == 5 * int(per_step)   # 3000
    assert f.money == pytest.approx(3000 / 1000 * 0.01)
    assert f.context_wall is False
    assert f.fits_time_budget is None            # no deadline given


def test_continuous_refill_pause_matches_the_deficit_math():
    # window: 1000-token bucket refilling over 60s, regime unmeasured (None).
    # per_step = 400; two steps fit (1000 -> 200), the third needs a pause.
    b = Budget(remaining_tokens=1000, reset_seconds=60.0, limit_tokens=1000)
    f = forecast_run(b, estimated_per_step=400, sigma=0.0, steps=3, k=2.0,
                     wait_threshold_s=60.0)      # generous: pauses stay waits
    assert f.waiting_s > 0.0
    assert f.waits + f.suspensions >= 1
    # first (and only) pause: deficit / implied rate * 1.15
    deficit = 400 - 200
    rate = (1000 - 200) / 60.0
    expected = deficit / rate * 1.15
    assert f.waiting_s == pytest.approx(expected, rel=0.05)
    assert f.steps == 3
    assert f.wall_clock_s == pytest.approx(f.waiting_s)   # no latency known


def test_fixed_regime_waits_out_full_resets():
    # fixed window: nothing trickles back, every pause is the whole reset
    b = Budget(remaining_tokens=1000, reset_seconds=30.0, limit_tokens=1000,
               refill_regime="fixed")
    f = forecast_run(b, estimated_per_step=400, sigma=0.0, steps=5, k=2.0)
    assert f.steps == 5
    assert f.waiting_s > 0.0
    # waiting is a whole number of full resets
    ratio = f.waiting_s / 30.0
    assert ratio == pytest.approx(round(ratio), abs=1e-6)
    # 30s reset > 15s default threshold: these pauses are suspensions
    assert f.suspensions >= 1 and f.waits == 0


def test_context_wall_stops_the_simulation_early():
    # a single step at 98% of the window: waiting can never fit it (§8.6)
    b = Budget(remaining_tokens=1000, reset_seconds=60.0, limit_tokens=1000)
    f = forecast_run(b, estimated_per_step=980, sigma=0.0, steps=4, k=2.0)
    assert f.context_wall is True
    assert f.steps < 4                           # stopped before the request
    assert f.steps == 1                          # the fresh window held one step
    assert f.total_tokens == 980


def test_time_budget_verdict():
    b = Budget(remaining_tokens=10 ** 6, limit_tokens=10 ** 6)
    fit = forecast_run(b, 500, 0.0, steps=4, latency_per_step=2.0,
                       time_budget_s=10.0)
    miss = forecast_run(b, 500, 0.0, steps=4, latency_per_step=2.0,
                        time_budget_s=7.0)
    assert fit.work_s == pytest.approx(8.0)
    assert fit.fits_time_budget is True
    assert miss.fits_time_budget is False


def test_str_is_a_readable_summary():
    b = Budget(remaining_tokens=10000, limit_tokens=10000)
    f = forecast_run(b, 500, 50.0, steps=5, price_per_1k_tokens=0.01)
    text = str(f)
    assert "5 step(s)" in text
    assert "tokens" in text and "wait" in text and "suspension" in text


# ---------------------------------------------------------------- via Session

def test_session_forecast_is_a_pure_read():
    reads = []

    def telemetry():
        reads.append(1)
        return Budget(remaining_tokens=50000, reset_seconds=60.0,
                      limit_tokens=50000)

    sched = PredictiveScheduler(
        backend=lambda msgs: ("ok", 50),
        telemetry=telemetry,
        price_per_1k_tokens=0.01,
    )
    with sched.session("forecast-1") as s:
        s.add_user("first question")
        s.add_user("second question")
        step_before = s.step
        est_state_before = copy.deepcopy(sched.estimator.to_dict())

        f1 = s.forecast(5)
        f2 = s.forecast(5)

        # coherent numbers: 5 steps fit the wide window with no stalls
        assert isinstance(f1, Forecast)
        assert f1.steps == 5
        assert f1.total_tokens > 0
        assert f1.waits == 0 and f1.suspensions == 0
        assert f1.money == pytest.approx(f1.total_tokens / 1000 * 0.01)
        # deterministic: same state in, same forecast out
        assert f1 == f2
        # no side effects: nothing consumed, nothing learned, nothing advanced
        assert s.step == step_before
        assert s.total_tokens_used == 0
        assert sched.estimator.samples == 0
        assert sched.estimator.to_dict() == est_state_before
    # forecast read telemetry (fresh, like decide) but never called the backend
    assert len(reads) == 2
