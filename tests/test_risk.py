"""Behavioral contract for the decision rule and the risk model."""

from agentpause import RiskModel, should_checkpoint


def test_checkpoint_when_budget_below_estimate_plus_margin():
    # remaining 900 < estimated 800 + margin (k=2 * sigma=100) = 1000  -> suspend
    assert should_checkpoint(remaining=900, estimated=800, sigma=100, k=2.0) is True


def test_continue_when_budget_comfortably_covers_step():
    # remaining 5000 >> 800 + 200 -> continue
    assert should_checkpoint(remaining=5000, estimated=800, sigma=100, k=2.0) is False


def test_higher_k_suspends_earlier():
    # same state, only k changes: cautious k trips, relaxed k does not
    state = dict(remaining=1000, estimated=800, sigma=150)
    assert should_checkpoint(**state, k=2.0) is True    # 800+300=1100 > 1000
    assert should_checkpoint(**state, k=1.0) is False   # 800+150=950  < 1000


def test_zero_sigma_reduces_to_plain_comparison():
    assert should_checkpoint(remaining=799, estimated=800, sigma=0, k=2.0) is True
    assert should_checkpoint(remaining=800, estimated=800, sigma=0, k=2.0) is False


def test_risk_score_rises_as_budget_shrinks():
    m = RiskModel()
    low = m.score(estimated=300, remaining=6000, context_tokens=1000, context_max=128000)
    high = m.score(estimated=300, remaining=400, context_tokens=1000, context_max=128000)
    assert high > low


def test_risk_weights_are_applied():
    m = RiskModel(w1=1.0, w2=0.0, w3=0.0)
    # only rate-limit term counts: est/remaining = 500/1000 = 0.5
    assert abs(m.score(estimated=500, remaining=1000,
                       context_tokens=999, context_max=1000) - 0.5) < 1e-9


def test_risk_handles_zero_remaining_without_crashing():
    m = RiskModel()
    # must not divide by zero
    assert m.score(estimated=500, remaining=0, context_tokens=10, context_max=1000) > 0
