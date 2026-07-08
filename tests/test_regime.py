"""Behavioral contract for D8: online refill-regime detection + chunked waits.

Two refill physics exist in the wild:
* continuous token bucket (Groq, OpenAI): tokens trickle back — refill-aware
  waits are exact;
* fixed window: nothing refills until a boundary instant, then everything —
  refill math would under-wait there.

The detector classifies the regime from observation alone: consecutive
telemetry readings with no real call in between. Rising remaining ≈ implied
rate → continuous; flat → fixed; not enough data → unknown (conservative).
Chunked waiting supplies those samples for free while ALSO letting the agent
resume as soon as the bucket actually holds enough.
"""

import pytest

from agentpause import Budget, decide
from agentpause.adapters.langgraph import AgentPauseGuard
from agentpause.refill import RegimeDetector


# -- the detector ---------------------------------------------------------------

def test_unknown_until_enough_votes():
    d = RegimeDetector()
    assert d.regime == "unknown"
    d.feed(dt=10.0, dr=1000, implied_rate=100.0)   # one vote is not enough
    assert d.regime == "unknown"


def test_rising_remaining_means_continuous():
    d = RegimeDetector()
    d.feed(dt=10.0, dr=950, implied_rate=100.0)    # ~95% of implied rate
    d.feed(dt=10.0, dr=1020, implied_rate=100.0)
    assert d.regime == "continuous"


def test_flat_remaining_means_fixed_window():
    d = RegimeDetector()
    d.feed(dt=10.0, dr=0, implied_rate=100.0)
    d.feed(dt=10.0, dr=-11, implied_rate=100.0)    # pings nibble a few tokens
    assert d.regime == "fixed"


def test_short_gaps_and_zero_rates_are_ignored():
    d = RegimeDetector(min_gap_s=2.0)
    d.feed(dt=0.5, dr=500, implied_rate=100.0)     # gap too short: noise
    d.feed(dt=10.0, dr=500, implied_rate=0.0)      # no implied rate: unusable
    assert d.regime == "unknown"


def test_ambiguous_ratios_do_not_vote():
    d = RegimeDetector()
    d.feed(dt=10.0, dr=300, implied_rate=100.0)    # 30%: neither clearly
    d.feed(dt=10.0, dr=300, implied_rate=100.0)
    assert d.regime == "unknown"


# -- regime drives the wait math ---------------------------------------------------

def test_fixed_regime_waits_the_full_reset():
    """Same numbers that would give a ~10s refill wait — but the provider is
    fixed-window, so the only safe wait is the full reset."""
    b = Budget(remaining_tokens=1000, reset_seconds=12.0, limit_tokens=6000,
               refill_regime="fixed")
    d = decide(b, estimated=1800, sigma=100.0, k=2.0, wait_threshold_s=15.0)
    assert d.action == "wait"
    assert d.wait_seconds == 12.0


def test_continuous_regime_keeps_refill_math():
    b = Budget(remaining_tokens=1000, reset_seconds=50.0, limit_tokens=6000,
               refill_regime="continuous")
    d = decide(b, estimated=1800, sigma=100.0, k=2.0)
    assert d.action == "wait"
    assert d.wait_seconds == pytest.approx(1000 / 100.0 * 1.15)


# -- chunked waiting in the guard ----------------------------------------------------

def test_guard_sleeps_in_chunks_and_resumes_early():
    """A 40s computed wait must not be one blind sleep: cap at chunk_s and
    re-read — if the bucket refills sooner, the agent resumes sooner."""
    budget = Budget(remaining_tokens=0, reset_seconds=14.0, limit_tokens=200_000)
    sleeps = []

    def fake_sleep(s):
        sleeps.append(s)
        if len(sleeps) == 2:                 # refilled after ~2 chunks
            budget.remaining_tokens = 150_000

    guard = AgentPauseGuard(telemetry=lambda: budget,
                            interrupt_fn=lambda p: None,
                            sleep_fn=fake_sleep, chunk_s=5.0)
    guard.check([{"role": "user", "content": "q " * 50}])
    assert len(sleeps) == 2                  # resumed at the SECOND re-read
    assert all(s <= 5.5 for s in sleeps)     # never a blind long sleep
