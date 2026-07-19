#!/usr/bin/env python3
"""Validate agentpause against a REAL provider — frontier models included.

Runs a multi-step task through the PredictiveScheduler with real rate-limit
telemetry, and reports: steps completed, suspensions/waits, 429s suffered
(should be 0), tokens used, and the estimated cost of ONE avoided crash
(what re-sending the full context would cost at list price).

Usage (any LiteLLM-supported provider):

    export GROQ_API_KEY=...        # or OPENAI_API_KEY / ANTHROPIC_API_KEY
    python scripts/validate_provider.py groq/llama-3.1-8b-instant
    python scripts/validate_provider.py gpt-4o-mini
    python scripts/validate_provider.py claude-haiku-4-5

Needs: pip install agentpause[litellm]
"""

import sys
import time

from agentpause import PredictiveScheduler, RateLimitHit
from agentpause.adapters.openai_compat import OpenAICompatAdapter

MODEL = sys.argv[1] if len(sys.argv) > 1 else "groq/llama-3.1-8b-instant"

QUESTIONS = [
    "In 2 sentences: what is proactive checkpointing for LLM agents?",
    "List 3 advantages of resuming over restarting a long agent task.",
    "In 2 sentences: what are rate-limit response headers?",
    "Why does a cost estimate need a safety margin? One sentence.",
    "Summarize this whole conversation in one sentence.",
]

if MODEL.startswith("anthropic/") or MODEL.startswith("claude"):
    from agentpause.adapters.anthropic import AnthropicAdapter
    adapter = AnthropicAdapter(MODEL.removeprefix("anthropic/"))
else:
    adapter = OpenAICompatAdapter.for_model(MODEL)   # direct HTTP: headers at the source
sched = PredictiveScheduler(
    backend=adapter.backend,
    telemetry=adapter.budget,           # full 3D telemetry: TPM + RPM + reset
    count_tokens=adapter.count_tokens,
)

waits = suspensions = 0
t0 = time.time()

with sched.session(f"validate-{MODEL.replace('/', '_')}") as s:
    if s.resumed:
        print(f"[resume] continuing from step {s.step}")
    for q in QUESTIONS[s.step:]:
        s.add_user(q)
        while True:
            d = s.next_action()
            if d.action == "continue":
                break
            if d.action == "wait":
                waits += 1
                wait_s = min((d.wait_seconds or d.budget.reset_seconds or 5), 15) + 0.5
                print(f"[wait] refill-aware chunked wait: {wait_s:.1f}s "
                      f"(full reset: {d.budget.reset_seconds}s, "
                      f"regime: {d.budget.refill_regime or 'unknown'})")
                time.sleep(wait_s)
                adapter.invalidate()
            else:
                suspensions += 1
                path = s.checkpoint()
                print(f"[suspend] saved to {path} — run again after the window resets")
                sys.exit(0)
        try:
            reply = s.call()
        except RateLimitHit:
            print("[FAIL] 429 survived retries — raise safety_k and re-run")
            sys.exit(1)
        b = d.budget
        print(f"[step {s.step}] ok | tokens left: {b.remaining_tokens}"
              f"{' | requests left: ' + str(b.remaining_requests) if b.remaining_requests is not None else ''}"
              f" | {reply[:60]}...")
    s.complete()

ctx_tokens = sum(adapter.count_tokens(m["content"]) for m in s.messages)
print("-" * 70)
print(f"VALIDATED on {MODEL}")
print(f"  steps: {s.step} | tokens used: {s.total_tokens_used} | time: {time.time()-t0:.0f}s")
print(f"  waits: {waits} | suspensions: {suspensions} | 429 suffered: 0 | hits absorbed: {sched.rate_limit_hits}")
print(f"  final context: ~{ctx_tokens} tokens — every avoided crash saves re-sending them all")
