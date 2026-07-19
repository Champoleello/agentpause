"""Tools as budgeted resources (F9.4).

LLM calls are not the only rate-limited thing an agent consumes: search
APIs, scrapers, geocoders all have quotas — and most send no telemetry
headers at all. :class:`ToolQuota` keeps a client-side sliding window of
recent calls so the agent can ask *before* calling (predictively, like
everything in agentpause) and learn exactly how long until a slot frees.

    quota = ToolQuota(calls_per_window=30, window_s=60)   # 30 calls/minute

    if quota.ready():
        quota.record(); result = search(q)
    # or let it pace the tool for you:
    search = quota.guard(search)
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Callable, Deque

__all__ = ["ToolQuota"]


class ToolQuota:
    """Client-side sliding-window quota for a tool.

    Args:
        calls_per_window: how many calls fit in one window.
        window_s: window length in seconds.
        clock: time source (injectable for tests).
    """

    def __init__(self, calls_per_window: int, window_s: float = 60.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.calls_per_window = calls_per_window
        self.window_s = window_s
        self._clock = clock
        self._calls: Deque[float] = deque()

    def _prune(self) -> None:
        cutoff = self._clock() - self.window_s
        while self._calls and self._calls[0] <= cutoff:
            self._calls.popleft()

    def ready(self) -> bool:
        """Whether a call can go out right now without hitting the quota."""
        self._prune()
        return len(self._calls) < self.calls_per_window

    def wait_seconds(self) -> float:
        """Seconds until the next slot frees (0 when ready)."""
        self._prune()
        if len(self._calls) < self.calls_per_window:
            return 0.0
        return self._calls[0] + self.window_s - self._clock()

    def record(self) -> None:
        """Register one call against the quota."""
        self._prune()
        self._calls.append(self._clock())

    def acquire(self, sleep_fn: Callable[[float], None] = time.sleep) -> None:
        """Block until a slot is available, then take it."""
        wait = self.wait_seconds()
        if wait > 0:
            sleep_fn(wait)
        self.record()

    def guard(self, fn: Callable[..., Any],
              sleep_fn: Callable[[float], None] = time.sleep) -> Callable[..., Any]:
        """Wrap a tool so every call paces itself against the quota."""
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            self.acquire(sleep_fn=sleep_fn)
            return fn(*args, **kwargs)
        return wrapped
