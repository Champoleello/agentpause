"""Direct adapter for OpenAI-compatible providers (Groq, OpenAI, ...).

Reads rate-limit headers straight from the HTTP response — no middle layer.
Born in the field: litellm currently drops provider response headers
(BerriAI/litellm#11749), which blinds telemetry. This adapter talks to the
provider's ``/chat/completions`` endpoint directly, so the headers the
scheduler needs are always there.

    from agentpause.adapters.openai_compat import OpenAICompatAdapter

    adapter = OpenAICompatAdapter.for_model("groq/llama-3.1-8b-instant")
    sched = PredictiveScheduler(backend=adapter.backend,
                                telemetry=adapter.budget,
                                count_tokens=adapter.count_tokens)

Works with any provider exposing the OpenAI chat API: pass ``base_url`` and
the env var holding the key, or use :meth:`for_model` for known prefixes.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..errors import BackendError, RateLimitHit, TelemetryError
from ..refill import RegimeDetector
from ..risk import Budget
from .litellm import _parse_reset

__all__ = ["OpenAICompatAdapter"]

KNOWN_PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
}


def _default_post(url: str, headers: Dict[str, str], payload: Dict[str, Any]):
    """Perform the HTTP POST; returns (status, response_headers, json_body)."""
    import httpx
    r = httpx.post(url, headers=headers, json=payload, timeout=120.0)
    try:
        body = r.json()
    except Exception:
        body = {}
    return r.status_code, dict(r.headers), body


class OpenAICompatAdapter:
    """Backend + telemetry via the provider's raw HTTP API.

    Args:
        model: model name as the PROVIDER knows it (e.g. ``llama-3.1-8b-instant``).
        base_url: API root, e.g. ``https://api.groq.com/openai/v1``.
        api_key: literal key, or None to read it from ``api_key_env``.
        api_key_env: environment variable holding the key.
        post_fn: transport, injectable for tests:
            ``(url, headers, payload) -> (status, headers, json)``.
        fallback_remaining / max_age_s / clock: as in the LiteLLM adapter.
        **request_kwargs: extra payload fields sent on every call.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        post_fn: Optional[Callable[..., Tuple[int, Dict[str, str], Dict[str, Any]]]] = None,
        fallback_remaining: Optional[int] = None,
        max_age_s: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
        **request_kwargs: Any,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        key = api_key if api_key is not None else os.environ.get(api_key_env)
        if not key:
            raise TelemetryError(f"No API key: set {api_key_env} or pass api_key=...")
        self._api_key = key
        self._auth = {"Authorization": f"Bearer {key}",
                      "Content-Type": "application/json"}
        self._post = post_fn if post_fn is not None else _default_post
        self.fallback_remaining = fallback_remaining
        self.max_age_s = max_age_s
        self._clock = clock
        self.request_kwargs = request_kwargs
        self._budget: Optional[Budget] = None
        self._read_at: Optional[float] = None
        # D8: refill-regime detection from ping-to-ping samples
        self.detector = RegimeDetector()
        self._prev_ping: Optional[Tuple[float, int, float]] = None
        # honest accounting: telemetry is cheap, but it is not free
        self.ping_tokens = 0
        self.ping_count = 0

    @classmethod
    def for_model(cls, model: str, **kwargs: Any) -> "OpenAICompatAdapter":
        """Build from a prefixed model string: ``groq/...``, ``openai/...``,
        or a bare OpenAI model name like ``gpt-4o-mini``."""
        prefix, _, rest = model.partition("/")
        if rest and prefix in KNOWN_PROVIDERS:
            base_url, env = KNOWN_PROVIDERS[prefix]
            return cls(rest, base_url=base_url, api_key_env=env, **kwargs)
        base_url, env = KNOWN_PROVIDERS["openai"]
        return cls(model, base_url=base_url, api_key_env=env, **kwargs)

    # -- the callables the scheduler wants -------------------------------------

    def backend(self, messages: List[Dict[str, str]]) -> Tuple[str, int]:
        body = self._call(messages, **self.request_kwargs)
        # a real call ends any sampling pair — but its own header reading is
        # a valid chain START: (post-call reading → post-wait ping) spans no
        # real traffic, so it can vote on the refill regime
        self._prev_ping = self._sample_point()
        reply = (body.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        used = int(body.get("usage", {}).get("total_tokens") or 0)
        return reply, used

    def _sample_point(self) -> Optional[Tuple[float, int, float]]:
        b, t = self._budget, self._read_at
        if b is None or t is None:
            return None
        implied = 0.0
        if b.limit_tokens is not None and (b.reset_seconds or 0) > 0:
            implied = (b.limit_tokens - b.remaining_tokens) / b.reset_seconds
        return (t, b.remaining_tokens, implied)

    def telemetry(self) -> int:
        return self.budget().remaining_tokens

    def budget(self) -> Budget:
        if not self._is_fresh():
            self.ping()
        if self._budget is not None:
            regime = self.detector.regime
            if regime != "unknown":
                self._budget.refill_regime = regime
            return self._age_adjusted(self._budget)
        if self.fallback_remaining is not None:
            return Budget(remaining_tokens=self.fallback_remaining)
        raise TelemetryError(
            "Provider sent no rate-limit headers; set fallback_remaining=... "
            "to proceed with a fixed assumption."
        )

    def _age_adjusted(self, b: Budget) -> Budget:
        """A copy of the cached reading with reset clocks advanced by its age.

        The cache may serve a reading up to ``max_age_s`` old; the reset
        countdowns in it were true at capture time, not now. Returning them
        raw makes the caller wait up to ``max_age_s`` longer than needed —
        so subtract the elapsed time (floor 0). The cache itself is never
        mutated: the next reader gets its own adjustment.
        """
        if self._read_at is None:
            return b
        age = max(0.0, self._clock() - self._read_at)
        if age == 0.0:
            return b
        from dataclasses import replace
        out = replace(b)
        if out.reset_seconds is not None:
            out.reset_seconds = max(0.0, out.reset_seconds - age)
        if out.reset_requests_seconds is not None:
            out.reset_requests_seconds = max(0.0, out.reset_requests_seconds - age)
        return out

    def invalidate(self) -> None:
        """Mark the cached telemetry stale (e.g. after sleeping out a wait):
        the next read will ping for fresh headers instead of reusing a
        reading taken before the wait."""
        self._read_at = None

    def ping(self) -> None:
        """1-token call whose only purpose is fresh rate-limit headers.

        Consecutive pings with no real call in between are exactly the
        no-traffic samples the regime detector needs: it observes whether
        ``remaining`` rises between them (continuous bucket) or stays flat
        (fixed window).
        """
        prev = self._prev_ping
        try:
            body = self._call([{"role": "user", "content": "ping"}], max_tokens=1)
        except RateLimitHit as hit:
            # a 429 on a telemetry ping is not an error — it IS telemetry:
            # the budget is exhausted, retry after what the provider says.
            # Raising here would crash the decision path (found by stress test).
            self.ping_count += 1
            self._budget = Budget(
                remaining_tokens=0,
                remaining_requests=0,
                reset_seconds=hit.retry_after or (self._budget.reset_seconds
                                                  if self._budget else 5.0),
                reset_requests_seconds=hit.retry_after,
            )
            self._read_at = self._clock()
            self._prev_ping = None          # not a valid regime sample
            return
        self.ping_count += 1
        self.ping_tokens += int(body.get("usage", {}).get("total_tokens") or 0)
        point = self._sample_point()
        if point is None:
            return
        if prev is not None:
            self.detector.feed(dt=point[0] - prev[0],
                               dr=point[1] - prev[1],
                               implied_rate=prev[2])
        self._prev_ping = point

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    # -- internals ---------------------------------------------------------------

    def _is_fresh(self) -> bool:
        return (self._read_at is not None
                and (self._clock() - self._read_at) <= self.max_age_s)

    def _call(self, messages: List[Dict[str, str]], **extra: Any) -> Dict[str, Any]:
        return self._post_and_check(
            f"{self.base_url}/chat/completions",
            {"model": self.model, "messages": messages, **extra},
        )

    def _post_and_check(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST, absorb telemetry headers, map HTTP status to typed errors."""
        status, headers, body = self._post(url, self._auth, payload)
        self._absorb(headers)
        if status == 429:
            ra = headers.get("retry-after")
            raise RateLimitHit(retry_after=float(ra) if ra else None)
        if 500 <= status <= 599:
            raise BackendError(f"provider returned {status}", retriable=True)
        if status >= 400:
            raise BackendError(f"provider returned {status}: {str(body)[:200]}")
        return body

    def _absorb(self, headers: Dict[str, str]) -> None:
        h = {str(k).lower(): v for k, v in headers.items()}

        def grab(*names: str) -> Optional[str]:
            for n in names:
                if n in h:
                    return h[n]
            return None

        tokens = grab("x-ratelimit-remaining-tokens",
                      "anthropic-ratelimit-tokens-remaining")
        if tokens is None:
            return                      # no telemetry in this response
        limit = grab("x-ratelimit-limit-tokens", "anthropic-ratelimit-tokens-limit")
        reset = grab("x-ratelimit-reset-tokens", "anthropic-ratelimit-tokens-reset")
        reset_req = grab("x-ratelimit-reset-requests",
                         "anthropic-ratelimit-requests-reset")
        requests_ = grab("x-ratelimit-remaining-requests",
                         "anthropic-ratelimit-requests-remaining")
        in_ = grab("anthropic-ratelimit-input-tokens-remaining")
        out = grab("anthropic-ratelimit-output-tokens-remaining")
        self._budget = Budget(
            remaining_tokens=int(float(tokens)),
            remaining_requests=int(float(requests_)) if requests_ is not None else None,
            reset_seconds=_parse_reset(reset) if reset is not None else None,
            reset_requests_seconds=_parse_reset(reset_req) if reset_req is not None else None,
            remaining_input_tokens=int(float(in_)) if in_ is not None else None,
            remaining_output_tokens=int(float(out)) if out is not None else None,
            limit_tokens=int(float(limit)) if limit is not None else None,
        )
        self._read_at = self._clock()
