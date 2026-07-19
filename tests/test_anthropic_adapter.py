"""Behavioral contract for the direct Anthropic adapter (F9.6 + F9.5).

Anthropic's API is not OpenAI-compatible: /v1/messages endpoint, x-api-key
auth, `system` as a top-level field, usage split into input/output tokens.
The adapter also plants `cache_control` breakpoints on the stable prefix
(F9.5): a resumed/looping agent re-sends a byte-identical prefix, the
provider's prompt cache recognizes it, and the prefill gets ~90% cheaper —
the cloud analog of the KV warm start. The savings are MEASURED
(`cache_read_tokens`), not assumed.

All offline: transport injected, no keys, no network.
"""

import pytest

from agentpause import RateLimitHit
from agentpause.adapters.anthropic import AnthropicAdapter

HEADERS = {
    "anthropic-ratelimit-tokens-remaining": "38000",
    "anthropic-ratelimit-input-tokens-remaining": "30000",
    "anthropic-ratelimit-output-tokens-remaining": "8000",
    "anthropic-ratelimit-requests-remaining": "49",
}
BODY = {
    "content": [{"type": "text", "text": "hello from claude"}],
    "usage": {"input_tokens": 100, "output_tokens": 40,
              "cache_read_input_tokens": 80, "cache_creation_input_tokens": 20},
}


class FakePost:
    def __init__(self, status=200, headers=None, body=None):
        self.status = status
        self.headers = HEADERS if headers is None else headers
        self.body = BODY if body is None else body
        self.calls = []

    def __call__(self, url, headers, payload):
        self.calls.append((url, headers, payload))
        return self.status, self.headers, self.body


MESSAGES = [
    {"role": "system", "content": "you are a careful analyst"},
    {"role": "user", "content": "first question"},
    {"role": "assistant", "content": "first answer"},
    {"role": "user", "content": "second question"},
]


def make_adapter(post, **kw):
    return AnthropicAdapter("claude-haiku-4-5", api_key="k", post_fn=post, **kw)


def test_backend_speaks_the_anthropic_protocol():
    post = FakePost()
    a = make_adapter(post)
    reply, used = a.backend(MESSAGES)
    assert reply == "hello from claude"
    assert used == 140                                    # input + output
    url, headers, payload = post.calls[0]
    assert url.endswith("/messages")
    assert headers["x-api-key"] == "k"
    assert "anthropic-version" in headers
    assert payload["model"] == "claude-haiku-4-5"
    assert "max_tokens" in payload                        # required by the API
    # system travels as a top-level field, not as a message
    assert all(m["role"] != "system" for m in payload["messages"])
    assert payload["system"][0]["text"] == "you are a careful analyst"


def test_cache_breakpoints_on_the_stable_prefix():
    post = FakePost()
    a = make_adapter(post)                                # cache_prompt=True default
    a.backend(MESSAGES)
    _, _, payload = post.calls[0]
    assert payload["system"][0]["cache_control"] == {"type": "ephemeral"}
    # the last message BEFORE the fresh question closes the stable prefix
    prefix_closer = payload["messages"][-2]
    assert prefix_closer["content"][0]["cache_control"] == {"type": "ephemeral"}
    # the fresh question itself is NOT cached
    assert isinstance(payload["messages"][-1]["content"], str)


def test_cache_prompt_false_sends_plain_payload():
    post = FakePost()
    a = make_adapter(post, cache_prompt=False)
    a.backend(MESSAGES)
    _, _, payload = post.calls[0]
    assert "cache_control" not in payload["system"][0]
    assert all(isinstance(m["content"], str) for m in payload["messages"])


def test_cache_savings_are_measured_not_assumed():
    a = make_adapter(FakePost())
    a.backend(MESSAGES)
    a.backend(MESSAGES)
    assert a.cache_read_tokens == 160                     # 80 per call, measured
    assert a.cache_write_tokens == 40


def test_budget_reads_anthropic_headers_through_the_adapter():
    a = make_adapter(FakePost())
    a.backend(MESSAGES)                                   # headers absorbed
    b = a.budget()
    assert b.remaining_tokens == 38000
    assert b.remaining_input_tokens == 30000
    assert b.remaining_output_tokens == 8000
    assert b.remaining_requests == 49


def test_429_maps_to_rate_limit_hit():
    post = FakePost(status=429, headers=dict(HEADERS, **{"retry-after": "11"}))
    a = make_adapter(post)
    with pytest.raises(RateLimitHit) as exc:
        a.backend(MESSAGES)
    assert exc.value.retry_after == 11.0
