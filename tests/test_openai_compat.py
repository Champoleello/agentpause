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


def test_invalidate_forces_a_fresh_ping():
    """After sleeping out a wait, a pre-wait reading must not be reused."""
    post = FakePost()
    a = make_adapter(post)
    a.budget()                          # ping #1
    a.budget()                          # cached: no new call
    assert len(post.calls) == 1
    a.invalidate()
    a.budget()                          # stale now: ping #2
    assert len(post.calls) == 2


def test_regime_votes_from_call_then_ping_pair():
    """A real call's header reading starts a sampling chain: (post-call,
    post-wait ping) spans no traffic and must vote on the regime."""
    clock = {"now": 0.0}
    readings = iter([1000, 1000, 2000, 2100, 3200])   # remaining rises: bucket

    class RisingPost:
        def __call__(self, url, headers, payload):
            h = {"x-ratelimit-remaining-tokens": str(next(readings)),
                 "x-ratelimit-limit-tokens": "6000",
                 "x-ratelimit-reset-tokens": "50s"}
            return 200, h, BODY

    a = OpenAICompatAdapter("m", base_url="https://x/v1", api_key="k",
                            post_fn=RisingPost(), clock=lambda: clock["now"])
    a.backend([{"role": "user", "content": "hi"}])    # chain starts here
    for _ in range(4):                                # wait chunks: ping each
        clock["now"] += 10.0
        a.invalidate()
        a.budget()
    assert a.detector.regime == "continuous"


def test_429_on_ping_is_telemetry_not_a_crash():
    """Found live by the stress test: when RPM is exhausted the PING itself
    gets 429. That must become a zero-budget reading, never an exception."""
    post = FakePost(status=429, headers={"retry-after": "7"})
    a = make_adapter(post)
    b = a.budget()                       # must NOT raise
    assert b.remaining_tokens == 0
    assert b.remaining_requests == 0
    assert b.reset_seconds == 7.0


def test_requests_reset_header_is_parsed():
    headers = dict(HEADERS, **{"x-ratelimit-reset-requests": "2s"})
    a = make_adapter(FakePost(headers=headers))
    assert a.budget().reset_requests_seconds == pytest.approx(2.0)


def test_exhausted_requests_wait_on_the_requests_clock():
    """The livelock scenario: tokens reset in 0.37s but requests in 20s —
    waiting on the token clock makes pings eat every refilled request slot."""
    from agentpause import Budget, decide
    b = Budget(remaining_tokens=6000, remaining_requests=0,
               reset_seconds=0.37, reset_requests_seconds=20.0,
               limit_tokens=6000)
    d = decide(b, estimated=1000, sigma=100.0, wait_threshold_s=30.0)
    assert d.action == "wait"
    assert d.wait_seconds == 20.0        # the REQUESTS clock, not 0.37s


def test_ping_tokens_are_accounted_for():
    """Telemetry is cheap but not free — the pings' cost must be visible."""
    a = make_adapter(FakePost())        # fake body reports 42 total_tokens
    a.budget()                          # ping #1
    a.invalidate()
    a.budget()                          # ping #2
    assert a.ping_count == 2
    assert a.ping_tokens == 84


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
