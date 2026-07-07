"""Behavioral contract for the richer budget model: RPM + reset-aware waiting.

Two additions to the decision inputs:

* **remaining_requests** — providers limit requests/minute (RPM) besides
  tokens/minute (TPM). An agent doing many small steps can exhaust requests
  while plenty of tokens remain; the old token-only rule would sail into a 429.
* **reset_seconds** — how soon the window resets. If the reset is imminent,
  *waiting* beats suspending: a checkpoint + relaunch costs more than a
  3-second pause. The decision becomes three-valued:
  ``continue`` / ``wait`` / ``checkpoint``.

Backward compatibility is part of the contract: telemetry returning a plain
``int`` must keep working everywhere.
"""

import pytest

from agentpause import Budget, PredictiveScheduler, StateStore, decide
from agentpause.adapters.langgraph import AgentPauseGuard
from agentpause.adapters.litellm import LiteLLMAdapter

from tests.test_litellm_adapter import FakeCompletion, make_response


# ---------------------------------------------------------------------------
# decide(): the three-valued rule
# ---------------------------------------------------------------------------

def test_continue_when_everything_fits():
    b = Budget(remaining_tokens=100_000, remaining_requests=50, reset_seconds=30.0)
    d = decide(b, estimated=1000, sigma=100.0)
    assert d.action == "continue"


def test_checkpoint_when_tokens_low_and_reset_far():
    b = Budget(remaining_tokens=500, reset_seconds=45.0)
    d = decide(b, estimated=1000, sigma=100.0, wait_threshold_s=15.0)
    assert d.action == "checkpoint"


def test_wait_when_tokens_low_but_reset_imminent():
    b = Budget(remaining_tokens=500, reset_seconds=5.0)
    d = decide(b, estimated=1000, sigma=100.0, wait_threshold_s=15.0)
    assert d.action == "wait"


def test_requests_exhausted_blocks_even_with_plenty_of_tokens():
    """The RPM dimension: zero requests left must never mean 'continue'."""
    b = Budget(remaining_tokens=100_000, remaining_requests=0, reset_seconds=45.0)
    assert decide(b, estimated=1000, sigma=100.0).action == "checkpoint"
    b_soon = Budget(remaining_tokens=100_000, remaining_requests=0, reset_seconds=3.0)
    assert decide(b_soon, estimated=1000, sigma=100.0).action == "wait"


def test_unknown_requests_and_reset_fall_back_to_token_rule():
    """No RPM/reset info (e.g. legacy telemetry): behave exactly like before."""
    assert decide(Budget(remaining_tokens=100_000), 1000, 100.0).action == "continue"
    assert decide(Budget(remaining_tokens=500), 1000, 100.0).action == "checkpoint"


def test_decision_carries_diagnostics():
    b = Budget(remaining_tokens=500, reset_seconds=5.0)
    d = decide(b, estimated=1000, sigma=100.0)
    assert d.budget is b
    assert d.estimated == 1000
    assert d.sigma == 100.0


# ---------------------------------------------------------------------------
# Session.next_action(): the scheduler speaks the new language
# ---------------------------------------------------------------------------

def make_backend(tokens=300):
    def backend(messages):
        return "ok", tokens
    return backend


def test_next_action_with_rich_telemetry(tmp_path):
    budget = Budget(remaining_tokens=100, reset_seconds=3.0)
    sched = PredictiveScheduler(
        backend=make_backend(),
        telemetry=lambda: budget,
        store=StateStore(str(tmp_path)),
    )
    with sched.session("t") as s:
        s.add_user("a long question " * 20)
        assert s.next_action().action == "wait"          # reset imminent
    budget.reset_seconds = 50.0
    with sched.session("t2") as s:
        s.add_user("a long question " * 20)
        assert s.next_action().action == "checkpoint"    # reset far away


def test_next_action_with_legacy_int_telemetry(tmp_path):
    """Plain-int telemetry still works: it maps to a tokens-only Budget."""
    sched = PredictiveScheduler(
        backend=make_backend(),
        telemetry=lambda: 100_000,
        store=StateStore(str(tmp_path)),
    )
    with sched.session("t") as s:
        s.add_user("hi")
        assert s.next_action().action == "continue"
        assert s.should_suspend() is False               # old API untouched


def test_should_suspend_sees_exhausted_requests(tmp_path):
    sched = PredictiveScheduler(
        backend=make_backend(),
        telemetry=lambda: Budget(remaining_tokens=100_000, remaining_requests=0),
        store=StateStore(str(tmp_path)),
    )
    with sched.session("t") as s:
        s.add_user("hi")
        assert s.should_suspend() is True


# ---------------------------------------------------------------------------
# LiteLLM adapter: parses RPM and reset headers into a Budget
# ---------------------------------------------------------------------------

RICH_HEADERS = {
    "x-ratelimit-remaining-tokens": "5000",
    "x-ratelimit-remaining-requests": "12",
    "x-ratelimit-reset-tokens": "7.66s",
}


def test_litellm_budget_parses_all_headers():
    fake = FakeCompletion(make_response(headers=RICH_HEADERS))
    adapter = LiteLLMAdapter(model="m", completion_fn=fake)
    b = adapter.budget()
    assert b.remaining_tokens == 5000
    assert b.remaining_requests == 12
    assert b.reset_seconds == pytest.approx(7.66)


def test_litellm_budget_with_minutes_format():
    headers = dict(RICH_HEADERS, **{"x-ratelimit-reset-tokens": "2m59.56s"})
    fake = FakeCompletion(make_response(headers=headers))
    adapter = LiteLLMAdapter(model="m", completion_fn=fake)
    assert adapter.budget().reset_seconds == pytest.approx(179.56)


def test_litellm_budget_with_milliseconds_format():
    headers = dict(RICH_HEADERS, **{"x-ratelimit-reset-tokens": "450ms"})
    fake = FakeCompletion(make_response(headers=headers))
    adapter = LiteLLMAdapter(model="m", completion_fn=fake)
    assert adapter.budget().reset_seconds == pytest.approx(0.45)


def test_litellm_budget_missing_optional_headers():
    fake = FakeCompletion(make_response(headers={"x-ratelimit-remaining-tokens": "5000"}))
    adapter = LiteLLMAdapter(model="m", completion_fn=fake)
    b = adapter.budget()
    assert b.remaining_tokens == 5000
    assert b.remaining_requests is None
    assert b.reset_seconds is None


# ---------------------------------------------------------------------------
# LangGraph guard: waits out an imminent reset instead of pausing the graph
# ---------------------------------------------------------------------------

class FakeInterrupt(Exception):
    def __init__(self, payload):
        self.payload = payload


def raising_interrupt(payload):
    raise FakeInterrupt(payload)


MESSAGES = [{"role": "user", "content": "a question " * 30}]


def test_guard_waits_when_reset_is_imminent():
    budget = Budget(remaining_tokens=100, reset_seconds=4.0)
    sleeps = []

    def fake_sleep(s):
        sleeps.append(s)
        budget.remaining_tokens = 100_000       # the window resets while we wait

    guard = AgentPauseGuard(telemetry=lambda: budget,
                            interrupt_fn=raising_interrupt, sleep_fn=fake_sleep)
    guard.check(MESSAGES)                        # waits, does NOT interrupt
    assert len(sleeps) == 1
    assert sleeps[0] >= 4.0                      # at least the reset time


def test_guard_interrupts_when_reset_is_far():
    budget = Budget(remaining_tokens=100, reset_seconds=120.0)
    guard = AgentPauseGuard(telemetry=lambda: budget,
                            interrupt_fn=raising_interrupt,
                            sleep_fn=lambda s: pytest.fail("must not sleep"))
    with pytest.raises(FakeInterrupt) as exc:
        guard.check(MESSAGES)
    assert exc.value.payload["reset_seconds"] == 120.0
