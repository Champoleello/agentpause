"""Quality A/B/C: what does context slimming cost in answer quality?

The stress test proved that compaction lets a stuck task COMPLETE (§8.6).
This script measures the other side of the trade: how much the ANSWER
degrades when old history is slimmed before the final call.

Protocol (facts-recall probe):
  1. Build one fixed conversation. Six specific facts (codename, budget,
     deadline, contact, port, city) are planted in the EARLY messages;
     the middle is verbose filler — exactly what compaction cuts first.
  2. Ask the same final quiz question under three conditions:
       A  FULL       — entire history sent untouched (upper bound)
       B  COMPACT    — Checkpoint.compact(): blind truncation, no LLM
       C  SUMMARIZE  — Checkpoint.summarize_with(): one-LLM-call summary
  3. Score = how many of the 6 planted facts the reply states correctly
     (normalized substring match), plus the prompt size each condition paid.

Usage:
    python3 scripts/quality_ab.py                       # Groq (needs GROQ_API_KEY)
    python3 scripts/quality_ab.py groq/llama-3.1-8b-instant
    python3 scripts/quality_ab.py --big                 # ~3x prompt, near the TPM ceiling
    python3 scripts/quality_ab.py --offline             # no key: mechanics check

--big scales the filler until the FULL prompt sits just under the provider's
whole rate window (~5k tokens on Groq free): the largest single call the tier
can physically serve. It cannot go beyond that — a single request bigger than
the window is the §8.6 wall itself and would 429 forever — so between calls
the script waits (chunked, telemetry re-read) for the window to refill.
Expect a few minutes of wall-clock; the waits are printed.

The offline backend answers by grepping its own prompt, so it scores 6/6
when the facts survive and less when they don't — it validates the harness,
not the model.
"""

from __future__ import annotations

import copy
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agentpause.state import Checkpoint

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
OFFLINE = "--offline" in sys.argv
BIG = "--big" in sys.argv
MODEL = ARGS[0] if ARGS else "groq/llama-3.1-8b-instant"

# small mode: 6 filler pairs x ~1.2k chars (~2k tok total prompt)
# big mode: 10 pairs x ~3.5k chars (~20k chars, ~5k tok — near the 6k TPM
#           ceiling of the Groq free tier; beyond it a single call can NEVER
#           succeed on that tier: that is the §8.6 wall, not a waiting problem)
FILLER_PAIRS = 7 if BIG else 6
FILLER_SCALE = 2 if BIG else 1

# ---------------------------------------------------------------- the facts
FACTS = {
    "codename":  "HERON-9",
    "budget":    "47500",
    "deadline":  "March 3",
    "contact":   "Marta Righi",
    "port":      "8443",
    "city":      "Trieste",
}

FILLER = (
    "Here is my detailed analysis of this phase. " * (12 * FILLER_SCALE)
    + "Several architectural considerations apply, including latency, "
      "observability, rollback strategy, and dependency isolation. " * (6 * FILLER_SCALE)
)

QUIZ = (
    "Final report time. WITHOUT asking me anything, state exactly: "
    "(1) the project codename, (2) the approved budget in EUR, "
    "(3) the launch deadline, (4) the name of the security contact, "
    "(5) the gateway port, (6) the city of the pilot deployment. "
    "Answer as a numbered list, values only."
)


def build_conversation() -> list:
    """Early messages carry the facts; the middle is expendable bulk."""
    msgs = [{"role": "system",
             "content": "You are a project assistant. Remember details "
                        "the user gives you; they will be needed later."}]
    # Facts are BURIED past the 200-char head that compact() preserves —
    # like real conversations, where the detail sits mid-paragraph.
    preamble = ("Before I give you the operational details, some background. "
                "This initiative went through three review rounds and the "
                "steering committee finally aligned on scope last week. The "
                "constraints below are binding and must not be renegotiated "
                "at implementation time, so record them carefully. ")
    msgs += [
        {"role": "user", "content":
            preamble +
            f"The project codename is {FACTS['codename']} and the approved "
            f"budget is {FACTS['budget']} EUR."},
        {"role": "assistant", "content":
            "All recorded in the project register."},
        {"role": "user", "content":
            preamble +
            f"Also: launch deadline {FACTS['deadline']}, security contact "
            f"{FACTS['contact']}, gateway on port {FACTS['port']}, pilot "
            f"deployment in {FACTS['city']}."},
        {"role": "assistant", "content":
            "Noted as binding constraints."},
    ]
    for i in range(3, 3 + FILLER_PAIRS):     # verbose middle steps, facts-free
        msgs.append({"role": "user",
                     "content": f"Work on phase {i}: review the plan."})
        msgs.append({"role": "assistant", "content": FILLER})
    return msgs


def score(reply: str) -> tuple:
    """(hits, missed_keys) — normalized substring match per fact."""
    norm = reply.lower().replace(",", "").replace(".", " ").replace("€", " ")
    hits, missed = 0, []
    for key, value in FACTS.items():
        v = value.lower().replace(",", "")
        if v in norm or v.replace(" ", "") in norm.replace(" ", ""):
            hits += 1
        else:
            missed.append(key)
    return hits, missed


# ---------------------------------------------------------------- backends
if OFFLINE:
    def call(messages):
        """Answers with whatever facts are still visible in its prompt."""
        prompt = " ".join(m.get("content") or "" for m in messages)
        found = [v for v in FACTS.values() if v in prompt]
        return "; ".join(found), sum(len(m.get("content") or "") for m in messages) // 4

    def summarizer(text):
        # a lossy but honest offline summarizer: keeps facts it can see
        found = [v for v in FACTS.values() if v in text]
        return "Earlier the user set up a project: " + ", ".join(found)
else:
    from agentpause.adapters.openai_compat import OpenAICompatAdapter
    from agentpause.adapters.anthropic import AnthropicAdapter

    if MODEL.startswith("anthropic/") or MODEL.startswith("claude"):
        adapter = AnthropicAdapter(MODEL.removeprefix("anthropic/"))
    else:
        adapter = OpenAICompatAdapter.for_model(MODEL)

    def _respect_budget(needed_tokens: int):
        """Wait (chunked, telemetry re-read) until the window fits the call."""
        deadline = time.time() + 300          # give up after 5 minutes
        while True:
            b = adapter.budget()
            if b.limit_tokens and needed_tokens >= b.limit_tokens:
                sys.exit(
                    f"\nSTOP: this call needs ~{needed_tokens} tokens but the "
                    f"provider's WHOLE window is {b.limit_tokens}. No amount of "
                    "waiting can serve it — this is the §8.6 wall. Shrink the "
                    "prompt or use a bigger tier/provider.")
            if b.remaining_tokens >= needed_tokens:
                return
            wait = min(b.reset_seconds or 15, 20)
            print(f"   (window: {b.remaining_tokens}/{b.limit_tokens or '?'} tok, "
                  f"need ~{needed_tokens} — waiting {wait:.0f}s for refill)")
            time.sleep(wait + 1)
            adapter.invalidate()
            if time.time() > deadline:
                sys.exit("STOP: window never refilled enough in 5 minutes.")

    def call(messages):
        est = sum(len(m.get("content") or "") for m in messages) // 4 + 400
        _respect_budget(est)
        return adapter.backend(messages)

    def summarizer(text):
        _respect_budget(len(text) // 4 + 500)
        reply, _ = adapter.backend([
            {"role": "user",
             "content": "Summarize the following conversation log in under 120 "
                        "words. PRESERVE every specific value verbatim (names, "
                        "numbers, dates, ports, places):\n\n" + text}])
        return reply


# ---------------------------------------------------------------- conditions
def run_condition(label: str, messages: list) -> dict:
    msgs = messages + [{"role": "user", "content": QUIZ}]
    prompt_chars = sum(len(m.get("content") or "") for m in msgs)
    reply, tokens = call(msgs)
    hits, missed = score(reply)
    print(f"\n[{label}] prompt {prompt_chars} chars | ~{tokens} tok | "
          f"facts {hits}/{len(FACTS)}"
          + (f" | LOST: {', '.join(missed)}" if missed else ""))
    print(f"   reply: {reply[:180]}{'…' if len(reply) > 180 else ''}")
    return {"label": label, "chars": prompt_chars, "tokens": tokens,
            "hits": hits, "missed": missed}


def main() -> None:
    base = build_conversation()
    results = []

    # A — full history
    results.append(run_condition("A FULL     ", copy.deepcopy(base)))

    # B — blind truncation (the model-free §8.6 fallback)
    cp = Checkpoint(session_id="quality-b", messages=copy.deepcopy(base))
    saved = cp.compact(keep_last=4)
    print(f"\n   compact() removed {saved} chars")
    results.append(run_condition("B COMPACT  ", cp.messages))

    # C — semantic summary (one extra LLM call)
    cp = Checkpoint(session_id="quality-c", messages=copy.deepcopy(base))
    saved = cp.summarize_with(summarizer, keep_last=4)
    print(f"\n   summarize_with() removed {saved} chars")
    results.append(run_condition("C SUMMARIZE", cp.messages))

    print("\n" + "=" * 64)
    print(f"QUALITY vs SLIMMING — model: {'offline-fake' if OFFLINE else MODEL}")
    print(f"{'condition':<14} {'prompt chars':>12} {'facts recalled':>15}")
    for r in results:
        print(f"{r['label']:<14} {r['chars']:>12} {r['hits']:>11}/{len(FACTS)}")
    print("=" * 64)
    print("Reading: A is the ceiling. B pays the §8.6 escape with lost early")
    print("facts. C spends one summary call to keep them — if C ≈ A at a")
    print("fraction of the prompt, semantic summarize earns its LLM call.")


if __name__ == "__main__":
    main()
