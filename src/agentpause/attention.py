"""The human in the loop as a budgeted resource (F11.3).

Overflow policy "ask" (see the §8.6 discussion in risk.py's neighbors) lets an
agent park a task for a human decision when it cannot safely proceed alone.
But a human is not an infinite, always-on channel — the same way a provider
enforces TPM/RPM, a person can only be interrupted so often before the
interruptions themselves become the cost. :class:`HumanAttentionBudget` models
that: a client-side sliding window of "asks" (mirroring :class:`~agentpause.
tools.ToolQuota`'s pattern for un-metered tool APIs), plus a manual override
for when the human explicitly steps away.

Two refill modes, and they compose:

* WINDOW (default) — at most ``max_asks`` questions per rolling
  ``window_s`` seconds. Exactly the sliding-window bookkeeping ``ToolQuota``
  uses for calls; here the "calls" are interruptions of a person.
* MANUAL OVERRIDE — the human declares presence directly. ``away()`` (with
  no argument) means "I'm gone, don't count on me, and I don't know when
  I'll be back" -- the budget reads as zero with no clock to wait on.
  ``away(until_s=5 * 60)`` means "back off my back in 5 minutes" -- the
  budget reads as zero but with a known, auto-expiring reset time.
  ``available()`` clears the override immediately.

The point of :meth:`HumanAttentionBudget.as_budget` is that none of this needs
new plumbing: it hands back a real :class:`agentpause.risk.Budget`, so
:func:`agentpause.risk.decide` (and anything built on it, like
``BudgetRouter`` or ``MultiAgentCoordinator``) treats "may I interrupt the
human?" with the exact same three-valued rule it already applies to TPM/RPM:
``continue`` (ask now), ``wait`` (a short pause beats a full suspend, e.g. the
window frees up in 40s), or ``checkpoint`` (suspend and resume later -- e.g.
the human is away with no return time, so there is nothing to wait FOR).

Wire-in example, overflow policy "ask" (§8.6):

    if plan.overflow_policy == "ask":
        if not attention.ready():
            wait = attention.wait_seconds()
            if wait is None:
                checkpoint()                 # human absent, no ETA: suspend
            else:
                schedule_retry(after=wait)   # short pause, try again
            return
        attention.record_ask()
        answer = ask_the_human(plan.question)
"""

from __future__ import annotations

import time
from collections import deque
from typing import Callable, Deque, Optional

from .risk import Budget

__all__ = ["HumanAttentionBudget"]

# A stand-in for "tokens never bind" when the budget being modeled is really
# about requests (asks), not tokens: large enough that no realistic
# `should_checkpoint` margin will ever trip on it.
_TOKEN_SENTINEL = 10 ** 9


class HumanAttentionBudget:
    """Client-side sliding-window budget for interrupting a human, plus a
    manual presence override.

    Args:
        max_asks: how many questions fit in one rolling window.
        window_s: window length in seconds.
        clock: time source (injectable for tests), matching ToolQuota's style.
    """

    def __init__(self, max_asks: int = 3, window_s: float = 3600.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.max_asks = max_asks
        self.window_s = window_s
        self._clock = clock
        self._asks: Deque[float] = deque()
        self._away = False
        self._away_until: Optional[float] = None   # None while away == "unknowable ETA"

    # -- sliding window plumbing (mirrors ToolQuota) -------------------------

    def _prune(self) -> None:
        cutoff = self._clock() - self.window_s
        while self._asks and self._asks[0] <= cutoff:
            self._asks.popleft()

    def _sync_away_expiry(self) -> None:
        """Clear an expired `away(until_s=...)` against the injected clock."""
        if self._away and self._away_until is not None and self._clock() >= self._away_until:
            self._away = False
            self._away_until = None

    # -- manual override ------------------------------------------------------

    def available(self) -> None:
        """The human is back. Clears any override immediately."""
        self._away = False
        self._away_until = None

    def away(self, until_s: Optional[float] = None) -> None:
        """The human steps away.

        Args:
            until_s: seconds from now until they're expected back. Omit (or
                pass None) when there's no known return time -- the budget
                will then report `wait_seconds() is None` (unknowable) and
                `as_budget()` will yield a request budget with no reset
                clock, which is the correct signal for a checkpoint rather
                than an indefinite wait.
        """
        self._away = True
        self._away_until = None if until_s is None else self._clock() + until_s

    @property
    def is_away(self) -> bool:
        """Whether the manual override is currently in effect."""
        self._sync_away_expiry()
        return self._away

    # -- budget reads -----------------------------------------------------

    def ready(self) -> bool:
        """Whether the human can be asked something right now."""
        if self.is_away:
            return False
        self._prune()
        return len(self._asks) < self.max_asks

    def wait_seconds(self) -> Optional[float]:
        """Seconds until the next ask is possible.

        Returns 0.0 when ready right now. Returns None when the human is away
        with no declared return time -- there's no clock to wait on, and an
        agent blocked only on this should checkpoint, not spin.
        """
        if self.is_away:
            if self._away_until is None:
                return None
            return max(0.0, self._away_until - self._clock())
        self._prune()
        if len(self._asks) < self.max_asks:
            return 0.0
        return self._asks[0] + self.window_s - self._clock()

    def record_ask(self) -> None:
        """Register one question asked of the human against the window."""
        self._prune()
        self._asks.append(self._clock())

    def remaining(self) -> int:
        """How many more questions fit in the current window right now."""
        self._prune()
        return max(0, self.max_asks - len(self._asks))

    # -- composition with the rest of the library --------------------------

    def as_budget(self) -> Budget:
        """Expose this state as a :class:`agentpause.risk.Budget`.

        Mapping:
            remaining_requests       -> asks left (0 while away)
            reset_requests_seconds   -> seconds to the next available slot,
                                         or to the declared return time; None
                                         when unknowable (away, no ETA)
            remaining_tokens         -> a large sentinel, so the token
                                         dimension of `decide()` never binds --
                                         this budget is purely about requests

        This is deliberately a real `Budget`, not a lookalike: it composes
        with `decide()`, `BudgetRouter`, and `MultiAgentCoordinator` with no
        new plumbing. In particular, an away human with no return time yields
        `remaining_requests=0` and `reset_requests_seconds=None` -- exactly
        the shape `decide()` already reads as "cannot compute a bounded wait",
        which routes to `checkpoint`. An agent whose only blocker is an
        absent human should suspend and resume later, never busy-wait.
        """
        if self.is_away:
            reset = None if self._away_until is None else max(0.0, self._away_until - self._clock())
            return Budget(
                remaining_tokens=_TOKEN_SENTINEL,
                remaining_requests=0,
                reset_requests_seconds=reset,
            )
        self._prune()
        remaining = self.remaining()
        if remaining > 0:
            reset = 0.0
        else:
            reset = self._asks[0] + self.window_s - self._clock()
        return Budget(
            remaining_tokens=_TOKEN_SENTINEL,
            remaining_requests=remaining,
            reset_requests_seconds=reset,
        )
