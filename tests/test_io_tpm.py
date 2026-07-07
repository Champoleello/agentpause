"""Behavioral contract for B5 (separate input/output token budgets) + D4
(Anthropic-style headers).

Anthropic rate-limits input tokens and output tokens SEPARATELY: a reasoning
agent can exhaust its output budget while plenty of input budget remains.
When telemetry reports the split dimensions, the decision must honor both.
Unknown dimensions never block (graceful degradation to the combined rule).

Anthropic also uses its own header names and an RFC 3339 timestamp for the
reset — the adapter recognizes them automatically alongside OpenAI-style.
"""

from datetime import datetime, timedelta, timezone

import pytest

from agentpause import Budget, PredictiveScheduler, StateStore, decide
from agentpause.adapters.litellm import LiteLLMAdapter

from tests.test_litellm_adapter import FakeCompletion, make_response


# -- decide() with split dimensions ---------------------------------------------

def test_output_budget_exhausted_blocks_despite_input_room():
    b = Budget(remaining_tokens=1_000_000,
               remaining_input_tokens=1_000_000,
               remaining_output_tokens=100,          # the reasoning ran dry
               reset_seconds=60.0)
    d = decide(b, estimated=2000, sigma=100.0,
               estimated_input=1000, estimated_output=650)
    assert d.action == "checkpoint"


def test_input_budget_exhausted_blocks():
    b = Budget(remaining_tokens=1_000_000,
               remaining_input_tokens=500,
               remaining_output_tokens=1_000_000)
    d = decide(b, estimated=2000, sigma=100.0,
               estimated_input=1000, estimated_output=650)
    assert d.action == "checkpoint"


def test_split_budgets_fit_continues():
    b = Budget(remaining_tokens=1_000_000,
               remaining_input_tokens=10_000,
               remaining_output_tokens=10_000)
    d = decide(b, estimated=2000, sigma=100.0,
               estimated_input=1000, estimated_output=650)
    assert d.action == "continue"


def test_unknown_split_dimensions_do_not_block():
    b = Budget(remaining_tokens=1_000_000)
    d = decide(b, estimated=2000, sigma=100.0,
               estimated_input=1000, estimated_output=650)
    assert d.action == "continue"


def test_split_violation_with_imminent_reset_waits():
    b = Budget(remaining_tokens=1_000_000,
               remaining_output_tokens=10, reset_seconds=3.0)
    d = decide(b, estimated=2000, sigma=100.0,
               estimated_input=1000, estimated_output=650)
    assert d.action == "wait"


def test_session_passes_split_estimates(tmp_path):
    budget = Budget(remaining_tokens=1_000_000, remaining_output_tokens=10)
    sched = PredictiveScheduler(
        backend=lambda m: ("ok", 100),
        telemetry=lambda: budget,
        store=StateStore(str(tmp_path)),
    )
    with sched.session("t") as s:
        s.add_user("q")
        assert s.next_action().action == "checkpoint"   # output dim exhausted


# -- adapter: Anthropic-style headers ----------------------------------------------

def _anthropic_headers(reset_iso=None):
    h = {
        "anthropic-ratelimit-tokens-remaining": "40000",
        "anthropic-ratelimit-input-tokens-remaining": "30000",
        "anthropic-ratelimit-output-tokens-remaining": "8000",
        "anthropic-ratelimit-requests-remaining": "49",
    }
    if reset_iso:
        h["anthropic-ratelimit-tokens-reset"] = reset_iso
    return h


def test_adapter_reads_anthropic_headers():
    fake = FakeCompletion(make_response(headers=_anthropic_headers()))
    adapter = LiteLLMAdapter(model="claude-haiku-4-5", completion_fn=fake)
    b = adapter.budget()
    assert b.remaining_tokens == 40000
    assert b.remaining_input_tokens == 30000
    assert b.remaining_output_tokens == 8000
    assert b.remaining_requests == 49


def test_adapter_parses_rfc3339_reset():
    reset_at = (datetime.now(timezone.utc) + timedelta(seconds=42)).isoformat()
    fake = FakeCompletion(make_response(headers=_anthropic_headers(reset_at)))
    adapter = LiteLLMAdapter(model="claude-haiku-4-5", completion_fn=fake)
    b = adapter.budget()
    assert b.reset_seconds == pytest.approx(42, abs=3)


def test_rfc3339_in_the_past_means_zero():
    reset_at = (datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat()
    fake = FakeCompletion(make_response(headers=_anthropic_headers(reset_at)))
    adapter = LiteLLMAdapter(model="claude-haiku-4-5", completion_fn=fake)
    assert adapter.budget().reset_seconds == 0.0


def test_openai_style_headers_still_work():
    fake = FakeCompletion(make_response(headers={
        "x-ratelimit-remaining-tokens": "5000",
        "x-ratelimit-reset-tokens": "7.66s",
    }))
    adapter = LiteLLMAdapter(model="gpt-4o-mini", completion_fn=fake)
    b = adapter.budget()
    assert b.remaining_tokens == 5000
    assert b.reset_seconds == pytest.approx(7.66)
    assert b.remaining_input_tokens is None
