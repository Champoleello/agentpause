"""Online detection of the provider's refill regime (D8).

Two refill physics exist behind rate limits:

* **continuous token bucket** — tokens trickle back at a steady rate
  (Groq, OpenAI): refill-aware waits are exact;
* **fixed window** — nothing returns until a boundary instant, then the
  whole budget at once: the only safe wait is the full reset.

Nobody documents which one a provider uses, but it can be *measured*:
take two telemetry readings with no real call in between and look at the
slope of ``remaining``. Rising at roughly the header-implied rate →
continuous. Flat (or nibbled slightly by the pings themselves) → fixed.

The detector votes on each valid sample pair and reports the majority of
the recent votes; with insufficient evidence it says ``unknown``, and the
caller stays conservative (full-reset waits).
"""

from __future__ import annotations

from collections import deque
from typing import Deque

__all__ = ["RegimeDetector"]


class RegimeDetector:
    """Classify the refill regime from (Δt, Δremaining) observations.

    Args:
        min_gap_s: sample pairs closer than this are noise and are ignored.
        window: how many recent votes participate in the majority.
        min_votes: evidence required before leaving ``unknown``.
    """

    CONTINUOUS_RATIO = 0.5   # observed/implied rate above this → continuous
    FIXED_RATIO = 0.1        # below this → fixed window

    def __init__(self, min_gap_s: float = 2.0, window: int = 3,
                 min_votes: int = 2) -> None:
        self.min_gap_s = min_gap_s
        self.min_votes = min_votes
        self._votes: Deque[str] = deque(maxlen=window)

    def feed(self, dt: float, dr: float, implied_rate: float) -> None:
        """Record one observation.

        Args:
            dt: seconds between the two readings (no real call in between).
            dr: change in remaining tokens over that gap.
            implied_rate: ``(limit - remaining) / reset_seconds`` at the
                first reading — what a continuous bucket would refill at.
        """
        if dt < self.min_gap_s or implied_rate <= 0:
            return
        ratio = (dr / dt) / implied_rate
        if ratio >= self.CONTINUOUS_RATIO:
            self._votes.append("continuous")
        elif ratio <= self.FIXED_RATIO:
            self._votes.append("fixed")
        # ambiguous ratios cast no vote

    @property
    def regime(self) -> str:
        """``"continuous"``, ``"fixed"``, or ``"unknown"`` (be conservative)."""
        if len(self._votes) < self.min_votes:
            return "unknown"
        cont = sum(1 for v in self._votes if v == "continuous")
        fixed = len(self._votes) - cont
        if cont > fixed:
            return "continuous"
        if fixed > cont:
            return "fixed"
        return "unknown"
