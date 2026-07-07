"""Behavioral contract for the cost Estimator."""

from agentpause import Estimator


def test_base_estimate_is_sum_of_parts():
    est = Estimator(max_tokens=650, tool_overhead=280)
    assert est.base_estimate(input_tokens=1000) == 1000 + 650 + 280


def test_epsilon_is_zero_without_history():
    est = Estimator()
    assert est.epsilon() == 0


def test_epsilon_tracks_underestimation():
    # If realized consumption consistently exceeds the base estimate,
    # epsilon should become positive and push the estimate up.
    # min_estimate=0 so the floor does not mask the epsilon correction
    est = Estimator(max_tokens=0, tool_overhead=0, min_estimate=0)  # base == input_tokens
    for _ in range(6):
        est.record(input_tokens=100, realized=180)  # error = +80 each time
    assert est.epsilon() == 80
    assert est.estimate(input_tokens=100) == 180  # base(100) + epsilon(80)


def test_epsilon_window_forgets_old_errors():
    est = Estimator(max_tokens=0, tool_overhead=0, error_window=3)
    for _ in range(3):
        est.record(100, 200)      # error +100
    for _ in range(3):
        est.record(100, 110)      # error +10, pushes old ones out
    assert est.epsilon() == 10


def test_estimate_has_a_floor():
    est = Estimator(max_tokens=0, tool_overhead=0, min_estimate=200)
    # tiny input plus negative correction must not drop below the floor
    for _ in range(6):
        est.record(input_tokens=100, realized=10)  # large negative error
    assert est.estimate(input_tokens=5) >= 200


def test_sigma_falls_back_before_four_samples():
    est = Estimator()
    est.record(100, 300)
    est.record(100, 320)
    # < 4 samples -> prudent fallback (fraction of the estimate), not the true std
    assert est.sigma(fallback_estimate=1000) == 1000 * 0.27


def test_sigma_uses_real_std_with_enough_samples():
    est = Estimator()
    for v in (300, 300, 300, 300):
        est.record(100, v)
    assert est.sigma(fallback_estimate=1000) == 0.0  # zero variance in the data


def test_samples_counter():
    est = Estimator()
    assert est.samples == 0
    est.record(100, 300)
    assert est.samples == 1
