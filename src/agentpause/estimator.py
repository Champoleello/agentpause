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
    ) -> None:
        self.max_tokens = max_tokens
        self.tool_overhead = tool_overhead
        self.min_estimate = min_estimate
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
        """Std. dev. of realized consumption.

        Falls back to a fraction of the current estimate until enough samples
        exist (< 4), so early decisions stay prudent.
        """
        h = self._realized
        if len(h) >= 4:
            mean = sum(h) / len(h)
            return (sum((x - mean) ** 2 for x in h) / len(h)) ** 0.5
        return fallback_estimate * 0.27

    # -- learning -----------------------------------------------------------

    def record(self, input_tokens: int, realized: int) -> None:
        """Update history with the actual consumption of a completed step."""
        error = realized - self.base_estimate(input_tokens)
        self._errors.append(error)
        self._all_errors.append(error)
        self._realized.append(realized)

    def upper_estimate(self, input_tokens: int, q: float = 0.95,
                       min_samples: int = 8) -> Optional[int]:
        """Quantile-based upper bound for the next step's cost.

        Consumption distributions are heavy-tailed: the mean-plus-``k·sigma``
        margin under-covers the tail. When enough history exists, the
        empirical ``q``-quantile of past estimation errors gives a
        statistically honest bound: base estimate + q-th worst error seen.
        Returns ``None`` until ``min_samples`` steps are recorded (callers
        fall back to the ``k·sigma`` rule).
        """
        n = len(self._all_errors)
        if n < min_samples:
            return None
        ordered = sorted(self._all_errors)
        idx = min(n - 1, max(0, int(q * n + 0.999999) - 1))  # ceil(q·n)-1, clamped
        return max(self.min_estimate, self.base_estimate(input_tokens) + ordered[idx])

    @property
    def samples(self) -> int:
        """Number of completed steps recorded so far."""
        return len(self._realized)
