"""Behavioral contract for F10.4 (phase-shift detection) + F9.3 (persistence).

F10.4 — agents change phase on long tasks (exploration ≠ synthesis). When the
recent estimation errors shift decisively away from the older ones, the old
history stops describing reality: drop it, so sigma and the quantile track
the NEW phase instead of a stale mixture.

F9.3 — everything the estimator learned (ε, σ, error history) rides inside
the checkpoint: a resumed session starts CALIBRATED instead of re-learning
from scratch — the telemetry-ping rule's counterpart for statistics (unlike
the budget, learned statistics do NOT go stale during a suspension).
"""

import pytest

from agentpause import Estimator, PredictiveScheduler, StateStore


def feed(est, errors):
    for e in errors:
        est.record(0, est.base_estimate(0) + e)


# -- F10.4: phase-shift detection ----------------------------------------------

def test_stable_errors_keep_full_history():
    est = Estimator()
    feed(est, [100, 110, 90, 105, 95, 100, 108, 92] * 4)   # one steady phase
    assert len(est._all_errors) == 32                       # nothing dropped


def test_phase_shift_drops_the_stale_history():
    est = Estimator()
    feed(est, [0, 5, -5, 3, -3, 4, -4, 2] * 2)      # phase A: tiny errors
    feed(est, [2000, 2050, 1980, 2020, 1990, 2010, 2040, 1970])  # phase B
    # the shift is unmistakable: history must now describe phase B only
    assert len(est._all_errors) <= 8
    sigma = est.sigma(1000)
    assert sigma < 100          # spread of phase B alone, not the A/B mixture


def test_phase_detection_can_be_disabled():
    est = Estimator(phase_window=None)
    feed(est, [0] * 16 + [2000] * 8)
    assert len(est._all_errors) == 24                       # mixture kept


# -- F9.3: the estimator rides inside the checkpoint ------------------------------

def test_to_dict_load_roundtrip():
    a = Estimator()
    feed(a, [50, 60, 40, 55, 45, 52])
    b = Estimator()
    b.load(a.to_dict())
    assert b.epsilon() == a.epsilon()
    assert b.sigma(1000) == a.sigma(1000)
    assert b.samples == a.samples


def test_resumed_session_starts_calibrated(tmp_path):
    store = StateStore(str(tmp_path))
    s1 = PredictiveScheduler(backend=lambda m: ("ok", 700),
                             telemetry=lambda: 1_000_000, store=store)
    with s1.session("job") as a:
        for _ in range(6):
            a.add_user("q")
            a.call()
        a.checkpoint()
    trained_sigma = s1.estimator.sigma(1000)

    # a brand-new process: fresh scheduler, fresh estimator...
    s2 = PredictiveScheduler(backend=lambda m: ("ok", 700),
                             telemetry=lambda: 1_000_000, store=store)
    with s2.session("job") as b:
        assert b.resumed
        # ...but the statistics survived the suspension
        assert s2.estimator.samples == 6
        assert s2.estimator.sigma(1000) == pytest.approx(trained_sigma)
