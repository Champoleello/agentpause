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
        limit_tokens: the bucket capacity (TPM limit). Together with
            ``reset_seconds`` it yields the refill rate, enabling
            refill-aware waits.
        refill_regime: ``"continuous"`` / ``"fixed"`` / None, as measured by
            :class:`~agentpause.refill.RegimeDetector`. Fixed-window providers
            get full-reset waits regardless of the refill math.
        remaining_seconds: wall-clock time left before the task's own deadline,
            if it has one. Unlike token/request budgets this does NOT refill by
            waiting — time only runs down — so a step that can't finish in the
            time left means checkpoint, never wait.
    """

    remaining_tokens: int
    remaining_requests: Optional[int] = None
    reset_seconds: Optional[float] = None
    reset_requests_seconds: Optional[float] = None   # RPM refills on its own clock
    remaining_input_tokens: Optional[int] = None
    remaining_output_tokens: Optional[int] = None
    limit_tokens: Optional[int] = None
    refill_regime: Optional[str] = None
    remaining_seconds: Optional[float] = None         # time budget (a deadline)


@dataclass
class Decision:
    """The outcome of :func:`decide`, with its inputs kept for diagnostics."""

    action: str            # "continue" | "wait" | "checkpoint"
    budget: Budget
    estimated: int
    sigma: float
    wait_seconds: Optional[float] = None   # suggested wait, when action == "wait"


_WAIT_BUFFER = 1.15     # wait 15% longer than the bare refill math suggests


def _refill_wait(budget: Budget, needed: float) -> Optional[float]:
    """Seconds until the bucket holds ``needed`` tokens, from headers alone.

    Token buckets refill continuously; the refill rate falls out of the
    headers with no extra assumptions:

        rate = (limit - remaining) / reset_seconds

    so the wait for a deficit is ``deficit / rate`` (buffered, and capped at
    the full reset — never wait longer than a guaranteed-full bucket).
    Returns None when the headers don't allow the math.
    """
    if budget.reset_seconds is None:
        return None
    if budget.refill_regime == "fixed":
        return budget.reset_seconds      # nothing trickles back: wait it out
    if budget.limit_tokens is None or budget.reset_seconds <= 0:
        return budget.reset_seconds
    missing = budget.limit_tokens - budget.remaining_tokens
    if missing <= 0:
        return budget.reset_seconds
    rate = missing / budget.reset_seconds
    deficit = needed - budget.remaining_tokens
    if deficit <= 0:
        return 0.5
    return min(budget.reset_seconds, deficit / rate * _WAIT_BUFFER)


def decide(
    budget: Budget,
    estimated: int,
    sigma: float,
    k: float = 2.0,
    wait_threshold_s: float = 15.0,
    estimated_input: Optional[int] = None,
    estimated_output: Optional[int] = None,
    estimated_latency: Optional[float] = None,
) -> Decision:
    """The three-valued decision rule.

    ``continue``    the next step fits: tokens cover ``estimated + k*sigma``
                    AND at least one request remains (when RPM is known) AND it
                    can finish before any deadline.
    ``wait``        it does not fit, but the window resets within
                    ``wait_threshold_s`` — pausing in place is cheaper than a
                    full checkpoint + relaunch.
    ``checkpoint``  it does not fit and the reset is far away (or unknown), or
                    the deadline can't be met (waiting never helps time).

    Unknown dimensions never block: if the provider reports no RPM, reset,
    split input/output, or deadline, the rule degrades gracefully to the
    combined token-only behavior.
    """
    tokens_fit = not should_checkpoint(budget.remaining_tokens, estimated, sigma, k)
    requests_fit = budget.remaining_requests is None or budget.remaining_requests >= 1
    input_fit = (budget.remaining_input_tokens is None or estimated_input is None
                 or estimated_input <= budget.remaining_input_tokens)
    output_fit = (budget.remaining_output_tokens is None or estimated_output is None
                  or estimated_output <= budget.remaining_output_tokens)
    time_fit = (budget.remaining_seconds is None or estimated_latency is None
                or estimated_latency <= budget.remaining_seconds)
    if not time_fit:
        # the deadline is the binding constraint: the step can't complete in
        # the time left, and waiting only burns more of it. Suspend and save
        # state rather than start a call that will overrun.
        return Decision("checkpoint", budget, estimated, sigma)
    if tokens_fit and requests_fit and input_fit and output_fit:
        return Decision("continue", budget, estimated, sigma)
    # compute the ACTUAL wait needed, then compare THAT to the threshold:
    # a bucket full in 50s may hold enough for the next call after 10s
    wait_s: Optional[float] = None
    if not requests_fit:
        # the REQUEST budget is the binding constraint: wait on ITS clock,
        # not the token clock (they refill independently — mixing them up
        # causes a livelock where telemetry pings eat the refilled slots)
        wait_s = budget.reset_requests_seconds or budget.reset_seconds
    elif budget.reset_seconds is not None:
        if not tokens_fit and input_fit and output_fit:
            needed = estimated + k * sigma
            if (budget.limit_tokens is not None
                    and needed >= budget.limit_tokens * 0.98):
                # anti-livelock (§8.6 of the research, met live in testing):
                # the call needs ~the WHOLE window. Waiting can never get
                # there — telemetry pings nibble whatever refills, and the
                # bucket hovers a hair below the bar forever. Suspend: a
                # resume starts against a truly full, untouched window.
                wait_s = None
            else:
                # pure token deficit: wait until the bucket refills enough
                wait_s = _refill_wait(budget, needed=needed)
        else:
            # split dimensions exhausted: refill rate unknown,
            # stay conservative and wait out the full reset
            wait_s = budget.reset_seconds
    if wait_s is not None and wait_s <= wait_threshold_s:
        return Decision("wait", budget, estimated, sigma, wait_seconds=wait_s)
    return Decision("checkpoint", budget, estimated, sigma)


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
