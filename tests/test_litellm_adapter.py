"""Behavioral contract for the LiteLLM adapter.

The adapter turns any LiteLLM-supported provider (Groq, OpenAI, Anthropic,
local servers, ...) into the two callables the scheduler needs:

    adapter = LiteLLMAdapter(model="groq/llama-3.1-8b-instant")
    sched = PredictiveScheduler(backend=adapter.backend, telemetry=adapter.telemetry)

All tests run offline: a fake ``completion_fn`` is injected, so no API key,
no network, and no ``litellm`` install are required.
"""

from types import SimpleNamespace

import pytest

from agentpause import PredictiveScheduler, StateStore
from agentpause.adapters.litellm import LiteLLMAdapter, TelemetryError


# -- test doubles -----------------------------------------------------------

def make_response(reply="ok", total_tokens=120, headers=None):
    """Build an object shaped like litellm's ModelResponse (the parts we use)."""
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=reply))],
        usage=SimpleNamespace(total_tokens=total_tokens),
    )
    resp._hidden_params = {"additional_headers": headers or {}}
    return resp


class FakeCompletion:
    """A fake litellm.completion that records calls and returns canned responses."""

    def __init__(self, response=None):
        self.response = response or make_response()
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeClock:
    """A clock tests can advance by hand."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


HEADERS = {"x-ratelimit-remaining-tokens": "5000"}


# -- backend ----------------------------------------------------------------

def test_backend_returns_reply_and_tokens():
    fake = FakeCompletion(make_response(reply="hello", total_tokens=321, headers=HEADERS))
    adapter = LiteLLMAdapter(model="groq/test", completion_fn=fake)
    reply, used = adapter.backend([{"role": "user", "content": "hi"}])
    assert reply == "hello"
    assert used == 321
    assert fake.calls[0]["model"] == "groq/test"
    assert fake.calls[0]["messages"] == [{"role": "user", "content": "hi"}]


def test_backend_refreshes_telemetry_from_headers():
    fake = FakeCompletion(make_response(headers=HEADERS))
    adapter = LiteLLMAdapter(model="m", completion_fn=fake)
    adapter.backend([{"role": "user", "content": "hi"}])
    # telemetry now answers from the header just seen — no extra LLM call
    assert adapter.telemetry() == 5000
    assert len(fake.calls) == 1


def test_backend_passes_extra_kwargs_to_completion():
    fake = FakeCompletion(make_response(headers=HEADERS))
    adapter = LiteLLMAdapter(model="m", completion_fn=fake, temperature=0.2)
    adapter.backend([{"role": "user", "content": "hi"}])
    assert fake.calls[0]["temperature"] == 0.2


# -- telemetry: the ping ------------------------------------------------------

def test_telemetry_pings_when_no_reading_exists():
    """First telemetry read has nothing cached: it must ping the provider."""
    fake = FakeCompletion(make_response(headers=HEADERS))
    adapter = LiteLLMAdapter(model="m", completion_fn=fake)
    assert adapter.telemetry() == 5000
    assert len(fake.calls) == 1
    # the ping is deliberately tiny
    assert fake.calls[0]["max_tokens"] == 1


def test_telemetry_uses_cache_while_fresh_and_repings_when_stale():
    clock = FakeClock()
    fake = FakeCompletion(make_response(headers=HEADERS))
    adapter = LiteLLMAdapter(model="m", completion_fn=fake, max_age_s=10.0, clock=clock)
    adapter.telemetry()                 # ping #1
    clock.now = 5.0
    adapter.telemetry()                 # fresh: no new call
    assert len(fake.calls) == 1
    clock.now = 20.0
    adapter.telemetry()                 # stale: ping #2
    assert len(fake.calls) == 2


def test_header_key_may_carry_provider_prefix():
    """litellm sometimes prefixes headers, e.g. 'llm_provider-x-ratelimit-...'."""
    headers = {"llm_provider-x-ratelimit-remaining-tokens": "777"}
    fake = FakeCompletion(make_response(headers=headers))
    adapter = LiteLLMAdapter(model="m", completion_fn=fake)
    assert adapter.telemetry() == 777


def test_missing_header_raises_clear_error():
    fake = FakeCompletion(make_response(headers={}))
    adapter = LiteLLMAdapter(model="m", completion_fn=fake)
    with pytest.raises(TelemetryError):
        adapter.telemetry()


def test_missing_header_with_fallback_returns_fallback():
    fake = FakeCompletion(make_response(headers={}))
    adapter = LiteLLMAdapter(model="m", completion_fn=fake, fallback_remaining=9999)
    assert adapter.telemetry() == 9999


# -- integration with the scheduler -------------------------------------------

def test_scheduler_runs_with_adapter(tmp_path):
    fake = FakeCompletion(make_response(reply="fine", total_tokens=100, headers=HEADERS))
    adapter = LiteLLMAdapter(model="m", completion_fn=fake)
    sched = PredictiveScheduler(
        backend=adapter.backend,
        telemetry=adapter.telemetry,
        store=StateStore(str(tmp_path)),
    )
    with sched.session("job") as s:
        s.add_user("question")
        assert s.should_suspend() is False   # 5000 remaining is plenty
        assert s.call() == "fine"
    assert s.total_tokens_used == 100


def test_scheduler_suspends_when_provider_reports_low_budget(tmp_path):
    low = {"x-ratelimit-remaining-tokens": "50"}
    fake = FakeCompletion(make_response(headers=low))
    adapter = LiteLLMAdapter(model="m", completion_fn=fake)
    sched = PredictiveScheduler(
        backend=adapter.backend,
        telemetry=adapter.telemetry,
        store=StateStore(str(tmp_path)),
    )
    with sched.session("job") as s:
        s.add_user("a long question " * 30)
        assert s.should_suspend() is True
