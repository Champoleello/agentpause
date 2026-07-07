"""Risk model and the suspension decision rules.

The pieces:

* :func:`should_checkpoint` — the hard token rule. Suspend when the remaining
  token budget cannot cover the estimated next step plus a safety margin
  ``k * sigma``.

* :class:`Budget` — what telemetry can report beyond a bare token count:
  remaining requests (RPM limits exist alongside TPM limits) and the time
  until the window resets.

* :func:`decide` — the three-valued rule built on top: ``continue`` when the
  next step fits (both tokens AND requests), ``wait`` when it does not fit but
  the window resets imminently (a short pause beats a full suspend/resume
  cycle), ``checkpoint`` otherwise.

* :class:`RiskModel` — a soft, multi-dimensional risk score used for monitoring
  and for richer policies. It aggregates rate-limit pressure, context-window
  pressure, and monetary-budget pressure into a single number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def should_checkpoint(remaining: int, estimated: int, sigma: float, k: float = 2.0) -> bool:
    """Return True when the next step does not safely fit in the remaining budget.

    Rule:  ``remaining < estimated + k * sigma``.

    Args:
        remaining: tokens left in the current rate-limit window.
        estimated: estimated cost of the next step.
        sigma: observed std. dev. of consumption (the margin's scale).
        k: safety factor; higher ``k`` suspends earlier (more cautious).
    """
    return remaining < estimated + k * sigma


@dataclass
class Budget:
    """A telemetry reading richer than a bare token count.

    Args:
        remaining_tokens: tokens left in the current rate-limit window (TPM).
        remaining_requests: requests left in the window (RPM), if known.
        reset_seconds: seconds until the window resets, if known.
        remaining_input_tokens: input-token budget, if the provider limits
            input and output separately (Anthropic does).
        remaining_output_tokens: output-token budget, if reported separately —
            a reasoning-heavy agent can drain this while input budget abounds.
    """

    remaining_tokens: int
    remaining_requests: Optional[int] = None
    reset_seconds: Optional[float] = None
    remaining_input_tokens: Optional[int] = None
    remaining_output_tokens: Optional[int] = None


@dataclass
class Decision:
    """The outcome of :func:`decide`, with its inputs kept for diagnostics."""

    action: str            # "continue" | "wait" | "checkpoint"
    budget: Budget
    estimated: int
    sigma: float


def decide(
    budget: Budget,
    estimated: int,
    sigma: float,
    k: float = 2.0,
    wait_threshold_s: float = 15.0,
    estimated_input: Optional[int] = None,
    estimated_output: Optional[int] = None,
) -> Decision:
    """The three-valued decision rule.

    ``continue``    the next step fits: tokens cover ``estimated + k*sigma``
                    AND at least one request remains (when RPM is known).
    ``wait``        it does not fit, but the window resets within
                    ``wait_threshold_s`` — pausing in place is cheaper than a
                    full checkpoint + relaunch.
    ``checkpoint``  it does not fit and the reset is far away (or unknown).

    Unknown dimensions never block: if the provider reports no RPM, reset,
    or split input/output information, the rule degrades gracefully to the
    combined token-only behavior.
    """
    tokens_fit = not should_checkpoint(budget.remaining_tokens, estimated, sigma, k)
    requests_fit = budget.remaining_requests is None or budget.remaining_requests >= 1
    input_fit = (budget.remaining_input_tokens is None or estimated_input is None
                 or estimated_input <= budget.remaining_input_tokens)
    output_fit = (budget.remaining_output_tokens is None or estimated_output is None
                  or estimated_output <= budget.remaining_output_tokens)
    if tokens_fit and requests_fit and input_fit and output_fit:
        action = "continue"
    elif budget.reset_seconds is not None and budget.reset_seconds <= wait_threshold_s:
        action = "wait"
    else:
        action = "checkpoint"
    return Decision(action=action, budget=budget, estimated=estimated, sigma=sigma)


@dataclass
class RiskModel:
    """Weighted, multi-dimensional risk score (diagnostic / for rich policies).

    ``risk = w1 * est/remaining + w2 * context/context_max + w3 * cost/budget``

    The weights default to values validated in the accompanying research and can
    be overridden or, in future, learned online.
    """

    w1: float = 0.55  # rate-limit pressure
    w2: float = 0.20  # context-window pressure
    w3: float = 0.25  # monetary-budget pressure

    def score(
        self,
        estimated: int,
        remaining: int,
        context_tokens: int,
        context_max: int,
        cost_estimate: float = 0.0,
        budget_remaining: float = 1.0,
    ) -> float:
        """Compute the aggregate risk score (>= 0; higher means riskier)."""
        rate = estimated / max(remaining, 1)
        ctx = context_tokens / max(context_max, 1)
        econ = cost_estimate / max(budget_remaining, 1e-9)
        return self.w1 * rate + self.w2 * ctx + self.w3 * econ
