"""agentpause + LiteLLM on a real provider (Groq's free tier here).

Needs:  pip install agentpause[litellm]
        export GROQ_API_KEY=...   (free at console.groq.com)

Swap the model string for any LiteLLM-supported provider:
"gpt-4o-mini" (OpenAI), "claude-haiku-4-5" (Anthropic), etc.

Run it twice to see resume in action: if the budget runs low the first run
suspends gracefully, and the second run picks up where it left off.
"""

import os
import sys

from agentpause import PredictiveScheduler
from agentpause.adapters.litellm import LiteLLMAdapter

if not os.environ.get("GROQ_API_KEY"):
    sys.exit("Set GROQ_API_KEY first (free key at console.groq.com).")

QUESTIONS = [
    "In one sentence: what is a rate limit?",
    "In one sentence: what is a checkpoint?",
    "In one sentence: why resume instead of restart?",
]

adapter = LiteLLMAdapter(model="groq/llama-3.1-8b-instant")
sched = PredictiveScheduler(backend=adapter.backend, telemetry=adapter.telemetry)

with sched.session("litellm-demo") as s:
    if s.resumed:
        print(f"[resume] continuing from step {s.step} — no work redone")
    for question in QUESTIONS[s.step:]:
        s.add_user(question)
        if s.should_suspend():
            path = s.checkpoint()
            print(f"[suspend] budget too low for the next call — state saved to {path}")
            print("          run this script again after the window resets")
            break
        print(f"[step {s.step + 1}] {question}")
        print(f"          -> {s.call()}")
    else:
        s.complete()
        print("[done] task finished, checkpoint cleared")
