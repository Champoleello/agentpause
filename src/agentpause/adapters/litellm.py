"""LiteLLM adapter: one adapter, every provider.

`LiteLLM <https://github.com/BerriAI/litellm>`_ exposes 100+ LLM providers
(OpenAI, Groq, Anthropic, local servers, ...) behind a single
``completion()`` call, and surfaces each provider's rate-limit response
headers. This adapter turns that into the two callables the scheduler needs:

    from agentpause import PredictiveScheduler
    from agentpause.adapters.litellm import LiteLLMAdapter

    adapter = LiteLLMAdapter(model="groq/llama-3.1-8b-instant")
    sched = PredictiveScheduler(backend=adapter.backend, telemetry=adapter.telemetry)

Design notes
------------
* **Telemetry is read from response headers** (``x-ratelimit-remaining-tokens``),
  which providers attach to every reply — so each ``backend()`` call refreshes
  the budget reading for free.
* **The telemetry ping.** A budget value serialized in a checkpoint goes stale
  (the window resets while the agent sleeps). So when ``telemetry()`` is asked
  for a value and has no *fresh* reading, it performs a deliberately tiny
  1-token call just to read the headers again. Cost: ~a dozen tokens and one
  round-trip — negligible next to a wrong suspend/resume decision.
* ``completion_fn`` is injectable so the adapter is fully testable offline;
  by default it lazily imports ``litellm.completion``.
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..errors import BackendError, RateLimitHit, TelemetryError
from ..risk import Budget

__all__ = ["LiteLLMAdapter", "TelemetryError", "RateLimitHit"]


def _parse_reset(value: str) -> Optional[float]:
    """Convert provider reset values to seconds-from-now.

    Handles duration strings ('7.66s', '2m59.56s', '450ms' — OpenAI/Groq
    style) and RFC 3339 timestamps ('2026-07-07T21:04:05Z' — Anthropic style).
    """
    if not value:
        return None
    text = str(value).strip()
    # RFC 3339 timestamp → seconds until that instant (never negative)
    if "T" in text and "-" in text:
        try:
            from datetime import datetime, timezone
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())
        except ValueError:
            return None
    m = re.fullmatch(
        r"(?:(\d+)h)?(?:(\d+)m(?!s))?(?:([\d.]+)\s*(ms|s)?)?",
        text,
    )
    if not m or not any(m.groups()):
        return None
    hours = int(m.group(1) or 0)
    minutes = int(m.group(2) or 0)
    number = float(m.group(3) or 0)
    seconds = number / 1000 if m.group(4) == "ms" else number
    return hours * 3600 + minutes * 60 + seconds


_INSTALL_HINT = ("The LiteLLM adapter needs the 'litellm' package. "
                 "Install it with:  pip install agentpause[litellm]")


def _enable_headers() -> None:
    """Ask litellm to surface provider response headers.

    Without this flag litellm does NOT populate
    ``_hidden_params["additional_headers"]`` — telemetry would be blind.
    (Found in the field on the very first real-provider run.)
    """
    import litellm
    litellm.return_response_headers = True


def _default_completion() -> Callable[..., Any]:
    try:
        from litellm import completion
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(_INSTALL_HINT) from exc
    _enable_headers()
    return completion


def _default_acompletion() -> Callable[..., Any]:
    try:
        from litellm import acompletion
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(_INSTALL_HINT) from exc
    _enable_headers()
    return acompletion


def _default_token_counter() -> Callable[..., int]:
    try:
        from litellm import token_counter
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(_INSTALL_HINT) from exc
    return token_counter


class LiteLLMAdapter:
    """Backend + telemetry for any LiteLLM-supported provider.

    Args:
        model: LiteLLM model string, e.g. ``"groq/llama-3.1-8b-instant"``
            or ``"gpt-4o-mini"``.
        completion_fn: the function performing the call. Defaults to
            ``litellm.completion`` (imported lazily); tests inject a fake.
        remaining_header: response header holding the remaining token budget.
        fallback_remaining: value ``telemetry()`` returns if the provider
            sends no budget header. ``None`` (default) raises
            :class:`TelemetryError` instead, so the problem is never silent.
        max_age_s: how long a header reading stays "fresh". Older readings
            trigger a telemetry ping.
        clock: time source (injectable for tests). Defaults to
            ``time.monotonic``.
        **completion_kwargs: extra arguments forwarded to every completion
            call (e.g. ``temperature=0.2``).
    """

    def __init__(
        self,
        model: str,
        completion_fn: Optional[Callable[..., Any]] = None,
        acompletion_fn: Optional[Callable[..., Any]] = None,
        token_counter_fn: Optional[Callable[..., int]] = None,
        remaining_header: str = "x-ratelimit-remaining-tokens",
        requests_header: str = "x-ratelimit-remaining-requests",
        reset_header: str = "x-ratelimit-reset-tokens",
        fallback_remaining: Optional[int] = None,
        max_age_s: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
        **completion_kwargs: Any,
    ) -> None:
        self.model = model
        self._completion = completion_fn
        self._acompletion = acompletion_fn
        self._token_counter = token_counter_fn
        self.remaining_header = remaining_header.lower()
        self.requests_header = requests_header.lower()
        self.reset_header = reset_header.lower()
        self.fallback_remaining = fallback_remaining
        self.max_age_s = max_age_s
        self._clock = clock
        self.completion_kwargs = completion_kwargs
        self._remaining: Optional[int] = None
        self._remaining_input: Optional[int] = None
        self._remaining_output: Optional[int] = None
        self._remaining_requests: Optional[int] = None
        self._reset_seconds: Optional[float] = None
        self._read_at: Optional[float] = None

    # -- the two callables the scheduler wants --------------------------------

    def backend(self, messages: List[Dict[str, str]]) -> Tuple[str, int]:
        """Perform the LLM call; returns ``(reply_text, total_tokens_used)``.

        Every real call doubles as a telemetry reading: the provider's
        rate-limit headers ride along with the response.
        """
        response = self._call(messages=messages, **self.completion_kwargs)
        self._absorb_headers(response)
        reply = response.choices[0].message.content or ""
        used = int(response.usage.total_tokens)
        return reply, used

    def telemetry(self) -> int:
        """Remaining tokens in the provider's rate-limit window.

        Answers from the last response's headers if that reading is still
        fresh (younger than ``max_age_s``); otherwise performs a tiny
        1-token ping to read the headers again. Never trusts old numbers —
        a stale budget is how agents crash mid-task.
        """
        if not self._is_fresh():
            self.ping()
        if self._remaining is not None:
            return self._remaining
        if self.fallback_remaining is not None:
            return self.fallback_remaining
        raise TelemetryError(
            f"No '{self.remaining_header}' header in the provider response. "
            "Either the provider does not expose token budgets, or the header "
            "name differs (set remaining_header=...), or set "
            "fallback_remaining=... to proceed with a fixed assumption."
        )

    async def abackend(self, messages: List[Dict[str, str]]) -> Tuple[str, int]:
        """Async twin of :meth:`backend` (uses ``litellm.acompletion``)."""
        response = await self._acall(messages=messages, **self.completion_kwargs)
        self._absorb_headers(response)
        reply = response.choices[0].message.content or ""
        used = int(response.usage.total_tokens)
        return reply, used

    async def atelemetry(self) -> int:
        """Async twin of :meth:`telemetry` (the ping, if needed, is async)."""
        if not self._is_fresh():
            await self.aping()
        if self._remaining is not None:
            return self._remaining
        if self.fallback_remaining is not None:
            return self.fallback_remaining
        raise TelemetryError(
            f"No '{self.remaining_header}' header in the provider response."
        )

    async def abudget(self) -> Budget:
        """Async twin of :meth:`budget`."""
        return Budget(
            remaining_tokens=await self.atelemetry(),
            remaining_requests=self._remaining_requests,
            reset_seconds=self._reset_seconds,
            remaining_input_tokens=self._remaining_input,
            remaining_output_tokens=self._remaining_output,
        )

    def count_tokens(self, text: str) -> int:
        """Per-model token count via litellm, with a safe heuristic fallback.

        Wire it into the scheduler for a much more accurate input estimate::

            PredictiveScheduler(..., count_tokens=adapter.count_tokens)
        """
        try:
            if self._token_counter is None:
                self._token_counter = _default_token_counter()
            return int(self._token_counter(model=self.model, text=text))
        except Exception:
            return max(1, len(text) // 4)   # ~4 chars/token heuristic

    def budget(self) -> Budget:
        """The full telemetry reading: tokens (TPM), requests (RPM), reset time.

        Same freshness policy as :meth:`telemetry` (stale reading → tiny ping).
        Pass this to the scheduler to unlock the three-valued decision
        (``continue`` / ``wait`` / ``checkpoint``)::

            sched = PredictiveScheduler(backend=adapter.backend,
                                        telemetry=adapter.budget)
        """
        return Budget(
            remaining_tokens=self.telemetry(),
            remaining_requests=self._remaining_requests,
            reset_seconds=self._reset_seconds,
            remaining_input_tokens=self._remaining_input,
            remaining_output_tokens=self._remaining_output,
        )

    # -- internals -------------------------------------------------------------

    def ping(self) -> None:
        """Issue a deliberately tiny call whose only purpose is fresh headers."""
        response = self._call(
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        self._absorb_headers(response)

    async def aping(self) -> None:
        """Async twin of :meth:`ping`."""
        response = await self._acall(
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        self._absorb_headers(response)

    async def _acall(self, **kwargs: Any) -> Any:
        if self._acompletion is None:
            self._acompletion = _default_acompletion()
        try:
            return await self._acompletion(model=self.model, **kwargs)
        except Exception as exc:
            self._map_provider_error(exc)
            raise

    def _map_provider_error(self, exc: Exception) -> None:
        """Translate provider exceptions into agentpause's typed errors.

        429 → RateLimitHit (with retry-after when reported).
        5xx / timeouts → BackendError(retriable=True): transient infrastructure.
        Anything else propagates unchanged (4xx = request problem, not retriable).
        """
        status = getattr(exc, "status_code", None)
        if status == 429 or exc.__class__.__name__ == "RateLimitError":
            headers = getattr(exc, "headers", None) or {}
            ra = headers.get("retry-after") if hasattr(headers, "get") else None
            raise RateLimitHit(retry_after=float(ra) if ra is not None else None) from exc
        if (status is not None and 500 <= int(status) <= 599) or exc.__class__.__name__ in (
            "Timeout", "APITimeoutError", "APIConnectionError", "ServiceUnavailableError",
            "InternalServerError",
        ):
            raise BackendError(f"transient provider failure: {exc}", retriable=True) from exc

    def _call(self, **kwargs: Any) -> Any:
        if self._completion is None:
            self._completion = _default_completion()
        try:
            return self._completion(model=self.model, **kwargs)
        except Exception as exc:
            self._map_provider_error(exc)
            raise

    def _is_fresh(self) -> bool:
        return (
            self._read_at is not None
            and (self._clock() - self._read_at) <= self.max_age_s
        )

    def _absorb_headers(self, response: Any) -> None:
        """Extract the budget reading from headers.

        Recognizes OpenAI/Groq-style names (``x-ratelimit-remaining-tokens``)
        and Anthropic-style names (``anthropic-ratelimit-tokens-remaining``,
        with separate input/output dimensions) automatically.
        """
        headers = getattr(response, "_hidden_params", {}).get("additional_headers", {})
        tokens: Optional[str] = None
        input_tokens: Optional[str] = None
        output_tokens: Optional[str] = None
        requests_: Optional[str] = None
        reset: Optional[str] = None
        for key, raw in headers.items():
            k = str(key).lower()
            # litellm may prefix provider headers, e.g. "llm_provider-x-ratelimit-..."
            if k.endswith("input-tokens-remaining"):
                input_tokens = raw
            elif k.endswith("output-tokens-remaining"):
                output_tokens = raw
            elif k.endswith(self.remaining_header) or k.endswith("ratelimit-tokens-remaining"):
                tokens = raw
            elif k.endswith(self.requests_header) or k.endswith("requests-remaining"):
                requests_ = raw
            elif k.endswith(self.reset_header) or k.endswith("ratelimit-tokens-reset"):
                reset = raw
        self._read_at = self._clock()
        self._remaining = int(float(tokens)) if tokens is not None else None
        self._remaining_input = int(float(input_tokens)) if input_tokens is not None else None
        self._remaining_output = int(float(output_tokens)) if output_tokens is not None else None
        self._remaining_requests = int(float(requests_)) if requests_ is not None else None
        self._reset_seconds = _parse_reset(reset) if reset is not None else None
