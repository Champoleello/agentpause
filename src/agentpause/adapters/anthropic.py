"""Direct adapter for Anthropic's Messages API (F9.6), cache-aware (F9.5).

Anthropic is not OpenAI-compatible — different endpoint (``/v1/messages``),
different auth (``x-api-key``), ``system`` as a top-level field, usage split
into input/output tokens — so it gets its own thin adapter, built on the
same telemetry machinery as :class:`OpenAICompatAdapter` (whose header
parser already understands ``anthropic-ratelimit-*`` names and RFC 3339
resets).

Prompt caching — the cloud analog of the KV warm start
------------------------------------------------------
True KV-cache export is impossible on cloud APIs, but Anthropic caches
byte-identical prompt prefixes server-side (cache reads cost ~10% of the
input price). agentpause's resume already re-sends the prefix verbatim by
design, so with ``cache_prompt=True`` (default) the adapter plants
``cache_control`` breakpoints on the stable prefix — the system message and
the last message before the fresh question — and MEASURES what came back
from cache (``cache_read_tokens``). Two honest caveats: the cache TTL is
minutes (helps waits and short suspensions, not tomorrow's resume), and
cache reads discount price/latency, not the rate-limit token count.

    from agentpause.adapters.anthropic import AnthropicAdapter

    adapter = AnthropicAdapter("claude-haiku-4-5")
    sched = PredictiveScheduler(backend=adapter.backend, telemetry=adapter.budget)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .openai_compat import OpenAICompatAdapter

__all__ = ["AnthropicAdapter"]

ANTHROPIC_VERSION = "2023-06-01"


class AnthropicAdapter(OpenAICompatAdapter):
    """Backend + telemetry for Anthropic models, with prompt-cache breakpoints.

    Args:
        model: e.g. ``"claude-haiku-4-5"``.
        api_key: literal key, or None to read ``ANTHROPIC_API_KEY``.
        cache_prompt: plant ``cache_control`` breakpoints on the stable
            prefix (system + last message before the fresh question).
        default_max_tokens: the Messages API requires ``max_tokens``.
        Everything else as in :class:`OpenAICompatAdapter`.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        cache_prompt: bool = True,
        default_max_tokens: int = 1024,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model,
            base_url="https://api.anthropic.com/v1",
            api_key=api_key,
            api_key_env="ANTHROPIC_API_KEY",
            **kwargs,
        )
        self._auth = {
            "x-api-key": self._api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        self.cache_prompt = cache_prompt
        self.default_max_tokens = default_max_tokens
        # measured cache effect (F9.5): never assumed, always counted
        self.cache_read_tokens = 0
        self.cache_write_tokens = 0

    # -- protocol translation ---------------------------------------------------

    def _call(self, messages: List[Dict[str, str]], **extra: Any) -> Dict[str, Any]:
        system: Optional[str] = None
        msgs = messages
        if msgs and msgs[0].get("role") == "system":
            system = msgs[0].get("content") or ""
            msgs = msgs[1:]
        payload: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": extra.pop("max_tokens", self.default_max_tokens),
            "messages": [dict(m) for m in msgs],
            **extra,
        }
        if system is not None:
            block: Dict[str, Any] = {"type": "text", "text": system}
            if self.cache_prompt:
                block["cache_control"] = {"type": "ephemeral"}
            payload["system"] = [block]
        if self.cache_prompt and len(payload["messages"]) >= 2:
            # the stable prefix ends just before the fresh question: closing
            # the breakpoint there lets a resumed (byte-identical) history
            # hit the server-side cache
            closer = payload["messages"][-2]
            content = closer.get("content")
            if isinstance(content, str):
                closer["content"] = [{"type": "text", "text": content,
                                      "cache_control": {"type": "ephemeral"}}]
        body = self._post_and_check(f"{self.base_url}/messages", payload)
        usage = body.get("usage") or {}
        self.cache_read_tokens += int(usage.get("cache_read_input_tokens") or 0)
        self.cache_write_tokens += int(usage.get("cache_creation_input_tokens") or 0)
        return body

    def backend(self, messages: List[Dict[str, str]]) -> Tuple[str, int]:
        body = self._call(messages, **self.request_kwargs)
        self._prev_ping = self._sample_point()   # real call starts a sampling chain
        text = "".join(b.get("text", "") for b in (body.get("content") or [])
                       if b.get("type") == "text")
        usage = body.get("usage") or {}
        used = int((usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0))
        return text, used
