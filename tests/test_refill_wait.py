"""Behavioral contract for D7: refill-aware waiting.

Providers with token-bucket limits refill CONTINUOUSLY: waiting for the full
window reset is wasteful. From the headers alone we can derive the refill
rate — (limit - remaining) / reset_seconds — and wait only until the bucket
holds enough for the NEXT call (plus a safety buffer).

Live-benchmark motivation (2026-07-08, Groq): full-reset waits cost 162s
wall-clock vs 78s for the reactive baseline; refill-aware waits close the gap.
"""

import pytest

from agentpause import Budget, decide


def test_wait_is_proportional_to_the_deficit():
    # bucket: 6000 cap, 1000 left, full in 50s → refill rate 100 tok/s
    b = Budget(remaining_tokens=1000, reset_seconds=50.0, limit_tokens=6000)
    # need 2000 (estimated+k·sigma≈2000) → deficit 1000 → 10s × 1.15 buffer
    d = decide(b, estimated=1800, sigma=100.0, k=2.0)
    assert d.action == "wait"
    assert d.wait_seconds == pytest.approx((2000 - 1000) / 100.0 * 1.15)


def test_wait_never_exceeds_full_reset():
    b = Budget(remaining_tokens=0, reset_seconds=50.0, limit_tokens=6000)
    d = decide(b, estimated=1_000_000, sigma=0.0,   # absurd need
               wait_threshold_s=60.0)               # generous threshold
    assert d.action == "wait"
    assert d.wait_seconds == 50.0                   # capped at the reset


def test_long_effective_wait_still_checkpoints():
    """Refill math says 48s but the threshold is 15s → suspend, don't idle."""
    b = Budget(remaining_tokens=0, reset_seconds=50.0, limit_tokens=6000)
    d = decide(b, estimated=5000, sigma=0.0, wait_threshold_s=15.0)
    assert d.action == "checkpoint"


def test_short_effective_wait_beats_a_long_reset():
    """The 2026-07-08 benchmark case: full reset 50s, but the next call only
    needs ~10s of refill — with refill-aware math we wait instead of idling."""
    b = Budget(remaining_tokens=1000, reset_seconds=50.0, limit_tokens=6000)
    d = decide(b, estimated=1800, sigma=100.0, k=2.0, wait_threshold_s=15.0)
    assert d.action == "wait"
    assert d.wait_seconds < 15.0


def test_without_limit_header_falls_back_to_full_reset():
    b = Budget(remaining_tokens=100, reset_seconds=8.0)
    d = decide(b, estimated=1000, sigma=100.0)
    assert d.action == "wait"
    assert d.wait_seconds == 8.0


def test_exhausted_requests_wait_the_full_reset():
    """RPM refill rate is unknown: stay conservative on that dimension."""
    b = Budget(remaining_tokens=100_000, remaining_requests=0,
               reset_seconds=8.0, limit_tokens=200_000)
    d = decide(b, estimated=100, sigma=10.0)
    assert d.action == "wait"
    assert d.wait_seconds == 8.0


def test_continue_carries_no_wait():
    b = Budget(remaining_tokens=100_000, limit_tokens=200_000, reset_seconds=30.0)
    d = decide(b, estimated=100, sigma=10.0)
    assert d.action == "continue"
    assert d.wait_seconds is None


def test_checkpoint_carries_no_wait():
    b = Budget(remaining_tokens=10, reset_seconds=300.0, limit_tokens=6000)
    d = decide(b, estimated=1000, sigma=100.0, wait_threshold_s=15.0)
    assert d.action == "checkpoint"
    assert d.wait_seconds is None
