"""Cost estimation for the next agent step.

The estimator predicts how many tokens the next LLM call will consume, so the
scheduler can decide whether it fits within the remaining rate-limit budget.

The estimate has two parts:

* a *base* estimate from observable quantities (current input size, the
  generation cap, and a fixed tool overhead), and
* a data-driven correction ``epsilon`` — the moving average of recent
  estimation errors — that adapts the base estimate to the actual behavior of
  the model and workload.

It also tracks ``sigma``, the standard deviation of realized consumption, which
the decision rule uses as a safety margin.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional


class Estimator:
    """Predicts next-step token cost and tracks consumption statistics.

    Args:
        max_tokens: the generation cap passed to the LLM (upper bound on output).
        tool_overhead: fixed token overhead attributed to tool calls.
        error_window: how many recent estimation errors feed ``epsilon``.
        min_estimate: a floor so the estimate is never absurdly small.
    """

    def __init__(
        self,
        max_tokens: int = 650,
        tool_overhead: int = 280,
        error_window: int = 6,
        min_estimate: int = 200,
        phase_window: "int | None" = 8,
        phase_threshold: float = 2.0,
    ) -> None:
        self.max_tokens = max_tokens
        self.tool_overhead = tool_overhead
        self.min_estimate = min_estimate
        self.phase_window = phase_window
        self.phase_threshold = phase_threshold
        self._errors: Deque[int] = deque(maxlen=error_window)
        self._all_errors: list[int] = []   # full history, for quantile margins
        self._realized: list[int] = []

    # -- estimation ---------------------------------------------------------

    def base_estimate(self, input_tokens: int) -> int:
        """Estimate from observable quantities only, without the correction."""
        return input_tokens + self.max_tokens + self.tool_overhead

    def epsilon(self) -> int:
        """Moving average of recent estimation errors (0 until data exists)."""
        if not self._errors:
            return 0
        return int(sum(self._errors) / len(self._errors))

    def estimate(self, input_tokens: int) -> int:
        """Full next-step estimate: base plus the learned correction."""
        return max(self.min_estimate, self.base_estimate(input_tokens) + self.epsilon())

    def sigma(self, fallback_estimate: int) -> float:
        """Std. dev. of the estimation RESIDUALS (realized − base estimate).

        Raw consumption trends upward as the context grows; measuring its
        spread would mistake systematic growth for uncertainty and inflate
        the safety margin (and the waits) enormously. The residuals absorb
        the trend, so their spread is the honest uncertainty of the estimate.

        Falls back to a fraction of the current estimate until enough samples
        exist (< 4), so early decisions stay prudent.
        """
        h = self._all_errors
        if len(h) >= 4:
            mean = sum(h) / len(h)
            return (sum((x - mean) ** 2 for x in h) / len(h)) ** 0.5
        return fallback_estimate * 0.27

    # -- learning -----------------------------------------------------------

    def record(self, input_tokens: int, realized: int) -> None:
        """Update history with the actual consumption of a completed step.

        Phase-shift detection (on by default): long tasks change behavior —
        exploration steps and synthesis steps consume differently. When the
        recent errors shift decisively away from the older ones, the old
        history stops describing reality and is dropped, so ``sigma`` and the
        quantile track the NEW phase instead of a stale mixture.
        """
        error = realized - self.base_estimate(input_tokens)
        self._errors.append(error)
        self._all_errors.append(error)
        self._realized.append(realized)
        w = self.phase_window
        if w is not None and len(self._all_errors) >= 2 * w:
            prev = self._all_errors[-2 * w:-w]
            recent = self._all_errors[-w:]
            mean_p = sum(prev) / w
            mean_r = sum(recent) / w
            std_p = (sum((x - mean_p) ** 2 for x in prev) / w) ** 0.5
            std_r = (sum((x - mean_r) ** 2 for x in recent) / w) ** 0.5
            scale = max(std_p, abs(mean_p) * 0.1, 10.0)   # floor: avoid zero-std traps
            shift = abs(mean_r - mean_p)
            # reset only on a COHERENT shift: the jump is large AND the recent
            # window is internally tight relative to ITS OWN level — a window
            # that straddles two phases has a huge relative spread and must
            # not become the fresh history
            tight = std_r <= max(abs(mean_r) * 0.1, 10.0) * self.phase_threshold
            if shift > self.phase_threshold * scale and tight:
                self._all_errors = list(recent)           # new phase: fresh eyes

    def upper_estimate(self, input_tokens: int, q: float = 0.95,
                       min_samples: int = 8,
                       window: int = 30) -> Optional[int]:
        """Quantile-based upper bound for the next step's cost.

        Consumption distributions are heavy-tailed: the mean-plus-``k·sigma``
        margin under-covers the tail. When enough history exists, the
        empirical ``q``-quantile of RECENT estimation errors gives a
        statistically honest bound: base estimate + q-th worst recent error.

        The **sliding window** (last ``window`` steps) matters on long tasks:
        agent behavior changes phase (exploration ≠ synthesis), and a monster
        step from an old phase should not fatten the margin forever. With few
        samples the formula naturally degrades to "max of the recent errors"
        — a declared, robust small-sample margin.

        Returns ``None`` until ``min_samples`` steps are recorded (callers
        fall back to the ``k·sigma`` rule).
        """
        if len(self._all_errors) < min_samples:
            return None
        recent = self._all_errors[-window:]
        n = len(recent)
        ordered = sorted(recent)
        idx = min(n - 1, max(0, int(q * n + 0.999999) - 1))  # ceil(q·n)-1, clamped
        return max(self.min_estimate, self.base_estimate(input_tokens) + ordered[idx])

    @property
    def samples(self) -> int:
        """Number of completed steps recorded so far."""
        return len(self._realized)

    # -- persistence (F9.3): learned statistics ride inside the checkpoint ----

    def to_dict(self) -> dict:
        """Serializable snapshot of everything learned (capped history)."""
        return {
            "errors": self._all_errors[-200:],
            "recent_errors": list(self._errors),
            "realized": self._realized[-200:],
        }

    def load(self, state: dict) -> None:
        """Restore a snapshot produced by :meth:`to_dict`.

        Unlike a serialized budget (which goes stale during a suspension),
        learned statistics stay valid: a resumed session starts calibrated.
        """
        self._all_errors = list(state.get("errors", []))
        self._errors.clear()
        self._errors.extend(state.get("recent_errors", []))
        self._realized = list(state.get("realized", []))
