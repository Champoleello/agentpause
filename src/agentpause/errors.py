"""Typed error hierarchy.

Everything agentpause raises derives from :class:`AgentPauseError`, so callers
can catch one base class — or handle each failure mode precisely:

* :class:`RateLimitHit` — the provider returned 429 despite the prediction
  (estimates are statistical, not oracular). Carries ``retry_after`` when the
  provider says how long to wait. The scheduler's retry policy handles these
  automatically; you only see one if all retries are exhausted.
* :class:`TelemetryError` — the budget could not be read (missing headers).
* :class:`CheckpointError` — the checkpoint could not be written or read
  (disk full, permissions, corrupted file).
* :class:`BackendError` — the LLM call failed for a non-rate-limit reason.
* :class:`KVError` — a llama.cpp KV-cache slot operation (save/restore/props)
  failed. Raised by the optional :mod:`agentpause.llamacpp_kv` plugin;
  callers of that plugin decide whether to degrade to a logical warm start
  or propagate.
* :class:`GPUError` — a GPU hardware query failed (driver not found, the
  `pynvml` bindings not installed, a nonexistent device index, or an NVML
  call itself failing). Raised by the optional
  :mod:`agentpause.adapters.local_resources` GPU reader; callers there catch
  it and re-raise as :class:`TelemetryError` so it reads as a budget-signal
  failure, not a rate limit.
"""

from __future__ import annotations

from typing import Optional

__all__ = [
    "AgentPauseError",
    "RateLimitHit",
    "TelemetryError",
    "CheckpointError",
    "BackendError",
    "KVError",
    "GPUError",
]


class AgentPauseError(Exception):
    """Base class for every error agentpause raises."""


class RateLimitHit(AgentPauseError):
    """The provider rate-limited a call (HTTP 429).

    Args:
        retry_after: seconds the provider asks to wait, when reported.
    """

    def __init__(self, message: str = "provider returned 429 (rate limit)",
                 retry_after: Optional[float] = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class TelemetryError(AgentPauseError):
    """The rate-limit budget could not be read from the provider response."""


class CheckpointError(AgentPauseError):
    """A checkpoint could not be persisted or restored."""


class BackendError(AgentPauseError):
    """The LLM call failed for a reason other than rate limiting.

    Args:
        retriable: True for transient infrastructure failures (HTTP 5xx,
            timeouts) that deserve a retry; False for request problems
            (4xx, validation) where retrying only wastes budget.
    """

    def __init__(self, message: str = "LLM call failed",
                 retriable: bool = False) -> None:
        super().__init__(message)
        self.retriable = retriable


class KVError(AgentPauseError):
    """A llama.cpp KV-cache slot operation (save/restore/props) failed.

    Raised by :class:`agentpause.llamacpp_kv.LlamaCppSlots` on HTTP or
    connection failure. The KV-cache is an accelerator, never the source of
    truth, so callers (see :class:`agentpause.llamacpp_kv.KVStateStore`)
    catch this and degrade to a logical warm start rather than crash.
    """


class GPUError(AgentPauseError):
    """A GPU hardware query failed.

    Covers: the NVIDIA driver/GPU not being present, the `pynvml` bindings
    not being installed, a ``device_index`` that does not exist, or an NVML
    call itself failing. Raised by the default reader in
    :mod:`agentpause.adapters.local_resources`
    (:class:`~agentpause.adapters.local_resources.GPUMemoryBudget`), which
    catches it (and any other exception from an injected ``reader_fn``) and
    re-raises it as :class:`TelemetryError` -- a VRAM-read failure, not a
    rate limit.
    """
