#!/usr/bin/env python3
"""Reproducible A/B benchmark — see the numbers with YOUR OWN free key.

Same 12-step task, same model, same real rate limit (Groq free tier, 6K
tokens/minute). Two conditions:

  A) REACTIVE baseline: calls blindly. When the provider answers 429, waits
     for the window and re-sends the WHOLE context (cold restart) and redoes
     the failed step — exactly what naive agent loops do.
  B) PREDICTIVE (agentpause): reads telemetry before each call, waits or
     suspends before the crash, resumes without redoing work.

Reported per condition: 429 errors suffered, steps redone, tokens re-sent
during recoveries, total tokens, wall-clock time.

Run it:
    export GROQ_API_KEY=...        # free at console.groq.com
    pip install agentpause[litellm]
    python scripts/benchmark_groq.py
"""

import os
import sys
import time

from agentpause import PredictiveScheduler, RateLimitHit
from agentpause.adapters.litellm import LiteLLMAdapter

MODEL = "groq/llama-3.1-8b-instant"
N_STEPS = 12
PROMPT_PAD = ("Background context for the analysis task. " * 120)  # ~1k tokens

if not os.environ.get("GROQ_API_KEY"):
    sys.exit("Set GROQ_API_KEY first (free key at console.groq.com).")

QUESTIONS = [f"Step {i}: answer in ONE short sentence — why do long-running "
             f"agents need resource-aware scheduling? (variation {i})"
             for i in range(1, N_STEPS + 1)]


def build_messages(history):
    return [{"role": "system", "content": PROMPT_PAD}] + history


# ---------------------------------------------------------------- condition A

def run_reactive():
    adapter = LiteLLMAdapter(model=MODEL)
    history, redone, hits, resent = [], 0, 0, 0
    t0 = time.time()
    i = 0
    while i < len(QUESTIONS):
        history.append({"role": "user", "content": QUESTIONS[i]})
        try:
            reply, used = adapter.backend(build_messages(history))
            history.append({"role": "assistant", "content": reply})
            i += 1
        except RateLimitHit as hit:
            hits += 1
            redone += 1
            resent += sum(adapter.count_tokens(m["content"]) for m in build_messages(history))
            wait = hit.retry_after or 20
            print(f"  [A] 429 at step {i + 1} — waiting {wait:.0f}s, will re-send everything")
            time.sleep(wait + 1)
            history.pop()          # the question is re-added and the step redone
    return {"429": hits, "redone": redone, "resent_tokens": resent,
            "time_s": time.time() - t0}


# ---------------------------------------------------------------- condition B

def run_predictive():
    adapter = LiteLLMAdapter(model=MODEL)
    sched = PredictiveScheduler(backend=adapter.backend, telemetry=adapter.budget,
                                count_tokens=adapter.count_tokens)
    waits = 0
    t0 = time.time()
    with sched.session("benchmark") as s:
        s.add_system(PROMPT_PAD)
        for q in QUESTIONS[s.step:]:
            s.add_user(q)
            while True:
                d = s.next_action()
                if d.action == "continue":
                    break
                waits += 1
                wait = (d.budget.reset_seconds or 20) + 0.5
                print(f"  [B] predicted overflow — waiting {wait:.0f}s BEFORE the call")
                time.sleep(wait)
            s.call()
        s.complete()
    return {"429": sched.rate_limit_hits, "redone": 0, "resent_tokens": 0,
            "waits": waits, "time_s": time.time() - t0}


if __name__ == "__main__":
    print(f"Benchmark on {MODEL} — {N_STEPS} steps, ~1k-token context, free-tier TPM")
    print("\n[A] REACTIVE baseline (crash & cold-restart):")
    a = run_reactive()
    print("\n[B] PREDICTIVE (agentpause):")
    b = run_predictive()

    print("\n" + "=" * 62)
    print(f"{'':28s} {'A reactive':>14s} {'B agentpause':>14s}")
    print("-" * 62)
    print(f"{'429 errors suffered':28s} {a['429']:14d} {b['429']:14d}")
    print(f"{'steps redone':28s} {a['redone']:14d} {b['redone']:14d}")
    print(f"{'tokens re-sent (waste)':28s} {a['resent_tokens']:14d} {b['resent_tokens']:14d}")
    print(f"{'wall-clock time':28s} {a['time_s']:13.0f}s {b['time_s']:13.0f}s")
    print("=" * 62)
    print("\nWaste avoided is money on paid tiers: re-sending a 50k-token context")
    print("on a frontier model costs real cents-per-crash. agentpause avoids it.")
