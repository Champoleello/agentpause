"""Task-completion forecast (F11.1): what will the REST of the run cost?

The scheduler decides one step at a time; this module answers the question the
operator actually asks before launching (or resuming) an agent: *"if I let it
run N more steps, how many tokens, how much money, how long — and how many
times will it stall on the rate limit?"*

The answer comes from a pure simulation, no network calls: walk the remaining
steps against a snapshot of the live token window, charging each step the same
conservative cost the decision rule uses (``estimate + k·sigma``). When a step
does not fit, the simulation pauses exactly the way :func:`~agentpause.risk.decide`
would — refill-aware math via :func:`~agentpause.risk._refill_wait` — refills
the simulated window accordingly, and classifies the pause: a short one is a
``wait`` (the agent sleeps in place), a long one is a ``suspension`` (the agent
checkpoints and resumes against a fresh, full window, having waited out the
full reset).

The forecast is honest about the two ways a plan dies rather than stalls:

* ``context_wall`` — a single step needs (nearly) the WHOLE window, so no
  amount of waiting ever fits it (the anti-livelock rule, §8.6). The
  simulation stops there and reports the partial numbers.
* ``fits_time_budget`` — the projected wall-clock time checked against the
  run's deadline, when one exists.

Entry points: :func:`forecast_run` (pure, testable in isolation) and
``Session.forecast(steps_remaining)`` which feeds it the live budget and the
estimator's current beliefs without consuming anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .risk import Budget, _refill_wait


@dataclass
class Forecast:
    """Projected cost of the remaining run, with its stall structure.

    Args:
        steps: steps actually simulated (< requested when a wall stops the run).
        total_tokens: projected token consumption (conservative: per-step
            estimate plus the ``k·sigma`` margin, like the decision rule).
        money: projected spend, when a price per 1k tokens is known.
        wall_clock_s: projected elapsed time = work + waiting.
        work_s: time spent inside LLM calls (0.0 when latency is unknown).
        waiting_s: time spent paused on the rate-limit window.
        suspensions: pauses long enough to warrant checkpoint + resume.
        waits: short pauses served by sleeping in place.
        fits_time_budget: whether wall_clock_s meets the deadline (None if none).
        context_wall: True when a single step can never fit a whole window —
            the simulation stopped early and the numbers are partial.
    """

    steps: int
    total_tokens: int
    money: Optional[float]
    wall_clock_s: float
    work_s: float
    waiting_s: float
    suspensions: int
    waits: int
    fits_time_budget: Optional[bool]
    context_wall: bool = False

    def __str__(self) -> str:
        parts = ["forecast: {} step(s), ~{} tokens".format(self.steps, self.total_tokens)]
        if self.money is not None:
            parts.append("~${:.4f}".format(self.money))
        parts.append("{:.1f}s wall-clock ({:.1f}s work + {:.1f}s waiting)".format(
            self.wall_clock_s, self.work_s, self.waiting_s))
        parts.append("{} wait(s), {} suspension(s)".format(self.waits, self.suspensions))
        if self.fits_time_budget is not None:
            parts.append("fits the time budget" if self.fits_time_budget
                         else "MISSES the time budget")
        if self.context_wall:
            parts.append("stopped at the context wall")
        return ", ".join(parts)


def forecast_run(
    budget: Budget,
    estimated_per_step: int,
    sigma: float,
    steps: int,
    k: float = 2.0,
    latency_per_step: Optional[float] = None,
    price_per_1k_tokens: Optional[float] = None,
    wait_threshold_s: float = 15.0,
    time_budget_s: Optional[float] = None,
) -> Forecast:
    """Simulate ``steps`` agent steps against the token window. No network.

    Each step charges ``estimated_per_step + k*sigma`` — the same conservative
    figure the decision rule checks — against a simulated remaining that starts
    at ``budget.remaining_tokens``. When the next step does not fit, the pause
    comes from the same refill math as the live decision (continuous regime:
    only the deficit at the implied refill rate, buffered and capped at the
    full reset; fixed regime or unknown rate: the full reset), the simulated
    window refills accordingly, and the pause is classified against
    ``wait_threshold_s``: at or below it, a ``wait``; above it, a
    ``suspension`` — modeled as a resume against a fresh full window, having
    waited out the full reset.

    Anti-livelock guard (mirrors ``decide()``, §8.6): when a single step needs
    >= 98% of the window capacity, waiting can never fit it — the simulation
    stops there, sets ``context_wall``, and reports the partial numbers.
    """
    per_step = estimated_per_step + k * sigma
    remaining = float(budget.remaining_tokens)
    start = float(budget.remaining_tokens)   # refill target when the limit is unknown
    limit = budget.limit_tokens
    reset = budget.reset_seconds

    steps_done = 0
    total = 0.0
    waiting_s = 0.0
    waits = 0
    suspensions = 0
    context_wall = False

    for _ in range(steps):
        if per_step > remaining:
            if limit is not None and per_step >= 0.98 * limit:
                # the step needs ~the whole window: no pause ever fits it
                context_wall = True
                break
            # same math as the live decision, fed the SIMULATED remaining
            sim = Budget(remaining_tokens=int(remaining), reset_seconds=reset,
                         limit_tokens=limit, refill_regime=budget.refill_regime)
            pause = _refill_wait(sim, needed=per_step)
            if pause is None:
                # no reset info: the window never refills in this model — only
                # a checkpoint/resume against a fresh window can move the run
                suspensions += 1
                remaining = float(limit) if limit is not None else start
            elif pause <= wait_threshold_s:
                waits += 1
                waiting_s += pause
                if (budget.refill_regime != "fixed" and limit is not None
                        and reset is not None and reset > 0 and limit > remaining):
                    # continuous bucket: the pause buys back rate * pause tokens
                    rate = (limit - remaining) / reset
                    remaining = min(float(limit), remaining + rate * pause)
                else:
                    # fixed window (or rate unknown): the pause was the full
                    # reset, so the bucket comes back full
                    remaining = float(limit) if limit is not None else start
            else:
                # long pause: cheaper to checkpoint. The resume sees a fresh
                # full window, and the waiting cost is the full reset.
                suspensions += 1
                waiting_s += reset if reset is not None else 0.0
                remaining = float(limit) if limit is not None else start
            if per_step > remaining:
                # even a fresh window cannot hold the step: same wall
                context_wall = True
                break
        remaining -= per_step
        total += per_step
        steps_done += 1

    work_s = steps_done * latency_per_step if latency_per_step is not None else 0.0
    wall_clock_s = work_s + waiting_s
    total_tokens = int(round(total))
    money = (total_tokens / 1000.0 * price_per_1k_tokens
             if price_per_1k_tokens is not None else None)
    fits = (wall_clock_s <= time_budget_s) if time_budget_s is not None else None
    return Forecast(
        steps=steps_done,
        total_tokens=total_tokens,
        money=money,
        wall_clock_s=wall_clock_s,
        work_s=work_s,
        waiting_s=waiting_s,
        suspensions=suspensions,
        waits=waits,
        fits_time_budget=fits,
        context_wall=context_wall,
    )
