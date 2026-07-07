"""Retry policy for rate-limit hits the prediction did not prevent.

Estimates are statistical: a 429 can still slip through. When it does, the
scheduler waits and retries instead of crashing — honoring the provider's
``retry-after`` when given, otherwise backing off exponentially.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

__all__ = ["RetryPolicy"]


@dataclass
class RetryPolicy:
    """How to wait-and-retry after an unexpected 429.

    Args:
        max_retries: attempts after the first failure before giving up
            (the original :class:`~agentpause.errors.RateLimitHit` is re-raised).
        base_delay_s: first backoff delay.
        factor: multiplier per attempt (exponential backoff).
        max_delay_s: ceiling for any single delay.
        sleep_fn: how to wait (defaults to ``time.sleep``; tests inject a fake).
        async_sleep_fn: how to wait on the async path (defaults to
            ``asyncio.sleep``; tests inject a fake).
        jitter: randomize each delay by ±this fraction (default ±25%), so many
            agents hitting the same window don't all retry in the same instant
            (the "thundering herd" problem). Set 0.0 for deterministic delays.
    """

    max_retries: int = 3
    base_delay_s: float = 1.0
    factor: float = 2.0
    max_delay_s: float = 60.0
    sleep_fn: Callable[[float], None] = field(default=time.sleep)
    async_sleep_fn: Optional[Callable[[float], Awaitable[None]]] = None
    jitter: float = 0.25

    def delay(self, attempt: int) -> float:
        """Backoff delay for the given attempt number (0-based), with jitter."""
        base = min(self.base_delay_s * self.factor ** attempt, self.max_delay_s)
        if self.jitter <= 0:
            return base
        return base * (1 + self.jitter * (2 * random.random() - 1))

    async def asleep(self, seconds: float) -> None:
        """Async wait, honoring an injected ``async_sleep_fn``."""
        if self.async_sleep_fn is not None:
            await self.async_sleep_fn(seconds)
        else:
            await asyncio.sleep(seconds)
