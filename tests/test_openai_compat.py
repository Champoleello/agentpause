"""Behavioral contract for the direct OpenAI-compatible adapter.

Exists because litellm currently drops provider response headers
(BerriAI/litellm#11749): telemetry must not depend on a middle layer.
All tests inject a fake transport — offline, no keys.
"""

import pytest

from agentpause import BackendError, PredictiveScheduler, RateLimitHit, StateStore
from agentpause.adapters.openai_compat import OpenAICompatAdapter

HEADERS = {
    "x-ratelimit-remaining-tokens": "5000",
    "x-ratelimit-remaining-requests": "12",
    "x-ratelimit-reset-tokens": "7.66s",
}
BODY = {"choices": [{"message": {"content": "hello"}}], "usage": {"total_tokens": 42}}


class FakePost:
    def __init__(self, status=200, headers=None, body=None):
        self.status, self.headers = status, HEADERS if headers is None else headers
        self.body = BODY if body is None else body
        self.calls = []

    def __call__(self, url, headers, payload):
        self.calls.append((url, payload))
        return self.status, self.headers, self.body


def make_adapter(post, **kw):
    return OpenAICompatAdapter("m", base_url="https://api.example.com/v1",
                               api_key="k", post_fn=post, **kw)


def test_backend_returns_reply_and_tokens():
    post = FakePost()
    a = make_adapter(post)
    assert a.backend([{"role": "user", "content": "hi"}]) == ("hello", 42)
    url, payload = post.calls[0]
    assert url.endswith("/chat/completions")
    assert payload["model"] == "m"


def test_budget_parsed_from_real_http_headers():
    a = make_adapter(FakePost())
    b = a.budget()                       # pings once
    assert b.remaining_tokens == 5000
    assert b.remaining_requests == 12
    assert b.reset_seconds == pytest.approx(7.66)


def test_backend_refreshes_budget_no_extra_ping():
    post = FakePost()
    a = make_adapter(post)
    a.backend([{"role": "user", "content": "hi"}])
    assert a.telemetry() == 5000
    assert len(post.calls) == 1


def test_429_maps_to_rate_limit_hit():
    post = FakePost(status=429, headers=dict(HEADERS, **{"retry-after": "9"}))
    a = make_adapter(post)
    with pytest.raises(RateLimitHit) as exc:
        a.backend([{"role": "user", "content": "hi"}])
    assert exc.value.retry_after == 9.0


def test_5xx_is_retriable_4xx_is_not():
    a = make_adapter(FakePost(status=503))
    with pytest.raises(BackendError) as exc:
        a.backend([])
    assert exc.value.retriable is True
    a = make_adapter(FakePost(status=400))
    with pytest.raises(BackendError) as exc:
        a.backend([])
    assert exc.value.retriable is False


def test_for_model_maps_known_prefixes(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k1")
    monkeypatch.setenv("OPENAI_API_KEY", "k2")
    g = OpenAICompatAdapter.for_model("groq/llama-3.1-8b-instant")
    assert g.base_url == "https://api.groq.com/openai/v1"
    assert g.model == "llama-3.1-8b-instant"
    o = OpenAICompatAdapter.for_model("gpt-4o-mini")
    assert o.base_url == "https://api.openai.com/v1"


def test_scheduler_integration(tmp_path):
    a = make_adapter(FakePost())
    sched = PredictiveScheduler(backend=a.backend, telemetry=a.budget,
                                store=StateStore(str(tmp_path)))
    with sched.session("t") as s:
        s.add_user("q")
        assert s.next_action().action == "continue"
        assert s.call() == "hello"
