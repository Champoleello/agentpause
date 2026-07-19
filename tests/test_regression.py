"""Behavioral contract for FeatureEstimator (D6): feature-based cost prediction.

The estimator learns tokens (and latency) as a ridge-regularized linear model
over workload features — context length, tool, model, temperature — not just
size. It stays a drop-in for the base Estimator and falls back to it until it
has data. All offline: synthetic samples with a known structure.
"""

import random

import pytest

from agentpause import FeatureEstimator, Estimator


def _train(est, n=60, seed=1):
    random.seed(seed)
    for _ in range(n):
        ctx = random.randint(500, 3000)
        tool = random.choice(["search", "calc"])
        true = 1.2 * ctx + (400 if tool == "search" else 100) + random.gauss(0, 25)
        lat = ctx / 800 + (0.5 if tool == "search" else 0.1) + random.gauss(0, 0.04)
        est.set_context(tool=tool)
        est.record(ctx, int(true), latency=lat)


def test_is_an_estimator_dropin():
    assert issubclass(FeatureEstimator, Estimator)
    est = FeatureEstimator()
    # with no data it behaves exactly like the base estimator
    base = Estimator()
    assert est.estimate(1000) == base.estimate(1000)


def test_learns_a_feature_dependent_model():
    est = FeatureEstimator(min_samples=6)
    _train(est)
    est.set_context(tool="search")
    pred_search = est.estimate(2000)
    est.set_context(tool="calc")
    pred_calc = est.estimate(2000)
    # the tool feature must move the prediction the right way (search costs more)
    assert pred_search > pred_calc
    truth = 1.2 * 2000 + 400
    assert abs(pred_search - truth) / truth < 0.10        # within 10%


def test_predicts_latency():
    est = FeatureEstimator(min_samples=6)
    _train(est)
    est.set_context(tool="search")
    lat = est.estimate_latency(2000)
    assert lat is not None
    assert abs(lat - (2000 / 800 + 0.5)) < 0.4


def test_latency_none_until_enough_samples():
    est = FeatureEstimator(min_samples=6)
    est.set_context(tool="search")
    est.record(1000, 1500, latency=1.0)
    assert est.estimate_latency(1000) is None            # only 1 sample


def test_falls_back_before_min_samples():
    est = FeatureEstimator(min_samples=10)
    base = Estimator()
    for _ in range(3):
        est.record(1000, 1500)
        base.record(1000, 1500)
    # under the threshold → the base-estimator path (which still uses the
    # recorded history), not a half-trained regression fit
    assert est.estimate(1000) == base.estimate(1000)


def test_categorical_features_are_one_hot_encoded():
    est = FeatureEstimator(min_samples=4)
    # model A always cheap, model B always expensive, same context
    for _ in range(8):
        est.set_context(model="A")
        est.record(1000, 1200)
        est.set_context(model="B")
        est.record(1000, 3000)
    est.set_context(model="A")
    a = est.estimate(1000)
    est.set_context(model="B")
    b = est.estimate(1000)
    assert b > a + 500


def test_ambient_context_is_cleared_after_record():
    est = FeatureEstimator(min_samples=4)
    est.set_context(tool="search")
    est.record(1000, 1500)
    # a following record with no set_context must not reuse "search"
    assert est._ambient == {}


def test_persistence_roundtrip_keeps_the_model():
    est = FeatureEstimator(min_samples=6)
    _train(est)
    est.set_context(tool="search")
    before = est.estimate(2000)
    state = est.to_dict()

    fresh = FeatureEstimator(min_samples=6)
    fresh.load(state)
    fresh.set_context(tool="search")
    after = fresh.estimate(2000)
    assert after == before                                # refit from saved samples
