"""Behavioral contract for D1: quantile margins for heavy-tailed consumption.

Real token consumption has heavy right tails: most steps are near the mean,
a few are much larger. A mean-plus-k·sigma margin under-covers that tail.
With enough history, the empirical q-quantile of past estimation errors is a
statistically honest upper bound. Until history exists, k·sigma stays in
charge — the quantile never makes early decisions LESS cautious.
"""

import pytest

from agentpause import Estimator, PredictiveScheduler, StateStore


def feed(est, errors, input_tokens=0):
    """Record steps whose realized cost differs from base by the given errors."""
    for e in errors:
        est.record(input_tokens, est.base_estimate(input_tokens) + e)


def test_upper_estimate_needs_history():
    est = Estimator()
    feed(est, [10] * 7)
    assert est.upper_estimate(0) is None          # < 8 samples → fall back
    feed(est, [10])
    assert est.upper_estimate(0) is not None


def test_quantile_covers_the_tail_where_mean_does_not():
    est = Estimator()
    # 9 tame steps and one monster: heavy right tail
    feed(est, [0, 0, 0, 0, 0, 0, 0, 0, 0, 5000])
    p95 = est.upper_estimate(0, q=0.95)
    assert p95 == est.base_estimate(0) + 5000     # the tail IS the bound
    # the median-ish bound stays tame
    assert est.upper_estimate(0, q=0.5) == max(est.min_estimate, est.base_estimate(0))


def test_scheduler_uses_quantile_when_enabled(tmp_path):
    consumptions = iter([600] * 9 + [3000])       # tail event at step 10
    sched = PredictiveScheduler(
        backend=lambda m: ("ok", next(consumptions)),
        telemetry=lambda: 3000,                    # fits the mean, NOT the tail
        store=StateStore(str(tmp_path)),
        quantile=0.95,
    )
    with sched.session("t") as s:
        for i in range(10):
            s.add_user("q")
            s.call()
        # history now includes the 3000-token tail event: the p95 bound
        # must make the next decision more cautious than the mean would
        s.add_user("q")
        assert s.next_action().action == "checkpoint"


def test_sliding_window_forgets_old_phases():
    """A monster step from an old phase must not fatten the margin forever:
    the quantile looks at the last `window` steps only."""
    est = Estimator()
    feed(est, [5000])                     # exploration-era monster
    feed(est, [10] * 30)                  # a full window of tame synthesis
    p95_windowed = est.upper_estimate(0, q=0.95, window=30)
    assert p95_windowed <= est.base_estimate(0) + 10   # monster forgotten
    worst_full = est.upper_estimate(0, q=1.0, window=1000)
    assert worst_full == est.base_estimate(0) + 5000   # kept only if asked


def test_small_window_degrades_to_max_of_recent():
    est = Estimator()
    feed(est, [0, 0, 0, 0, 0, 0, 0, 200])
    # 8 samples: ceil(0.95*8)-1 = 7 → the max. Declared, robust, no fake p95.
    assert est.upper_estimate(0, q=0.95, window=30) == est.base_estimate(0) + 200


def test_quantile_none_changes_nothing(tmp_path):
    sched = PredictiveScheduler(
        backend=lambda m: ("ok", 100),
        telemetry=lambda: 100_000,
        store=StateStore(str(tmp_path)),
    )
    with sched.session("t") as s:
        s.add_user("q")
        assert s.next_action().action == "continue"
