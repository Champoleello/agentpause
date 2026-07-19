#!/usr/bin/env python3
"""Stress test: a long, irregular, multi-window task on a real provider.

Harder than the benchmark on every axis:
  * 20 steps with a ~2k-token context → most calls need a large slice of the
    free-tier TPM window (6k): constant pressure, several window crossings;
  * WILDLY varying output demands (one word … ~300 words) → heavy-tailed
    consumption, stressing the residual sigma and the p95 quantile margin;
  * the full decision ladder in play: continue / refill-aware chunked wait /
    graceful checkpoint — when a wait would exceed the threshold the script
    EXITS with a checkpoint; run it again and it resumes from the exact step.

Usage:
    export GROQ_API_KEY=...
    python3 scripts/stress_test.py           # exits on checkpoint; rerun to resume
    python3 scripts/stress_test.py --auto    # acts as its own orchestrator:
                                             # sleeps out the window and resumes
                                             # with a FRESH session from disk

Why does the default exit instead of resuming by itself? Because that is the
point of a checkpoint: the process can DIE (freeing RAM, slots, money) and the
state survives on disk. In production a cron/orchestrator/LangGraph thread
does the relaunching; --auto just plays that role for you.

A final report prints everything worth knowing, including what the regime
detector concluded and what the telemetry cost.
"""

import sys
import time

from agentpause import PredictiveScheduler, RateLimitHit, StateStore
from agentpause.adapters.openai_compat import OpenAICompatAdapter

AUTO = "--auto" in sys.argv
ARGS = [a for a in sys.argv[1:] if a != "--auto"]
MODEL = ARGS[0] if ARGS else "groq/llama-3.1-8b-instant"

PAD = ("You are analyzing a distributed system. Inspection log: " +
       " ".join(f"module service_{i:03d}: {2+i%7} deps, p95 {60+(i*7)%240}ms, "
                f"err {round(0.1+(i%9)*0.4,1)}%, coverage {30+(i*13)%60}%."
                for i in range(1, 61)))          # ~2k tokens of context

ASKS = [
    "Name the single most critical module. One word only.",
    "Explain the three biggest risks in this system in ~300 words.",
    "Give one number: the average error rate. Just the number.",
    "Write a detailed ~250-word refactoring plan for the worst module.",
    "Yes or no: is the auth gateway a single point of failure?",
    "Describe in ~200 words how you would add caching to the order flow.",
    "List 3 module names, comma-separated, nothing else.",
    "Write a ~300-word incident postmortem for a hypothetical outage.",
    "One word: which metric matters most here?",
    "Summarize the whole analysis so far in ~150 words.",
] * 2                                            # 20 steps, alternating extremes

adapter = OpenAICompatAdapter.for_model(MODEL)
events = {"waits": 0, "retries": 0, "suspensions": 0}
sched = PredictiveScheduler(
    backend=adapter.backend,
    telemetry=adapter.budget,
    count_tokens=adapter.count_tokens,
    store=StateStore(".stress"),
    quantile=0.95,                               # tail-honest margins
    on_event=lambda n, i: events.__setitem__("retries", events["retries"] + 1)
                          if n == "retry" else None,
)


def run_once() -> bool:
    """One session pass. Returns True when the task is COMPLETE."""
    with sched.session("stress") as s:
        if s.resumed:
            print(f"↻ RESUMED from step {s.step} — no work redone\n")
        else:
            s.add_system(PAD)
        for q in ASKS[s.step:]:
            s.add_user(q)
            while True:
                d = s.next_action()
                if d.action == "continue":
                    break
                if d.action == "wait":
                    events["waits"] += 1
                    w = min(d.wait_seconds or 10, 15) + 0.5
                    print(f"  [wait {events['waits']}] {w:.0f}s "
                          f"(reset {d.budget.reset_seconds}s, "
                          f"regime {d.budget.refill_regime or 'unknown'})")
                    time.sleep(w)
                    adapter.invalidate()
                else:
                    events["suspensions"] += 1
                    path = s.checkpoint()
                    print(f"\n🛑 GRACEFUL CHECKPOINT at step {s.step} → {path}")
                    return False
            reply = s.call()
            print(f"[step {s.step:2d}] {len(reply):4d} chars | "
                  f"tokens left {d.budget.remaining_tokens}")
        s.complete()
        events["final"] = (s.step, s.total_tokens_used)
        return True


t0 = time.time()
last_suspend_step = -1
compacted_at = -1
while not run_once():
    step_now = sched.store.load("stress").step
    if step_now == last_suspend_step:
        if compacted_at != step_now:
            # §8.6 overflow: the context itself no longer fits a full window.
            # Apply the mandatory-summarize policy OFFLINE, on the suspended
            # checkpoint — useful work during the suspension, no LLM needed.
            cp = sched.store.load("stress")
            saved = cp.compact(keep_last=4, max_chars=200)
            sched.store.save(cp)
            compacted_at = step_now
            print(f"  [overflow §8.6] compacted suspended checkpoint: "
                  f"~{saved} chars of old history summarized away")
        else:
            print("\n⚠ Still stuck after compaction: raise the TPM tier "
                  "or shrink the task context.")
            sys.exit(1)
    last_suspend_step = step_now
    if not AUTO:
        print("   The next call would need too long a wait.")
        print("   Run the script again in ~1 minute (or use --auto).")
        sys.exit(0)
    pause = (adapter.budget().reset_seconds or 60) + 2
    print(f"   [--auto] playing orchestrator: sleeping {pause:.0f}s, "
          f"then resuming with a FRESH session from disk...")
    time.sleep(pause)
    adapter.invalidate()

steps, total_tokens = events["final"]
print("\n" + "=" * 64)
print("STRESS TEST COMPLETE")
print(f"  steps: {steps} | total tokens: {total_tokens} "
      f"| wall-clock: {time.time()-t0:.0f}s")
print(f"  waits: {events['waits']} | suspensions+resumes: {events['suspensions']} "
      f"| 429 absorbed: {sched.rate_limit_hits} | retries: {events['retries']}")
print(f"  telemetry: {adapter.ping_count} pings, {adapter.ping_tokens} tokens "
      f"| regime detected: {adapter.detector.regime}")
print(f"  estimator: {sched.estimator.samples} samples learned")
print("=" * 64)
