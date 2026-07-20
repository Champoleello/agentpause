"""Quality A/B/C: what does context slimming cost in answer quality?

The stress test proved that compaction lets a stuck task COMPLETE (§8.6).
This script measures the other side of the trade: how much the ANSWER
degrades when old history is slimmed before the final call.

Protocol (facts-recall probe, plus a VOICE probe living in the SAME run):
  1. Build one fixed conversation. Six specific facts (codename, budget,
     deadline, contact, port, city) are planted in the EARLY messages;
     the middle is verbose filler — exactly what compaction cuts first.
     The SAME early messages also carry a planted VOICE: the assistant
     persona ("Zappy") is given a system-prompt style plus two short,
     literal catchphrases it uses in its own early replies (see the
     VOICE_MARKERS section below for exactly what and why). Facts and
     voice deliberately coexist in one conversation, not two, because the
     real question — raised by a Reddit reader, see the comment at
     VOICE_MARKERS — is whether slimming for facts ALSO erases style.
  2. Ask the same final quiz question under three conditions:
       A  FULL       — entire history sent untouched (upper bound)
       B  COMPACT    — Checkpoint.compact(): blind truncation, no LLM
       C  SUMMARIZE  — Checkpoint.summarize_with(): one-LLM-call summary
  3. Score = how many of the 6 planted facts the reply states correctly
     (normalized substring match), plus the prompt size each condition
     paid, PLUS how many of the 2 planted voice markers the reply still
     uses (also a plain substring match — see voice_score()).

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
Expect a few minutes of wall-clock; the waits are printed. No new flags are
needed for the VOICE dimension — it rides along on the same three calls.

The offline backend answers by grepping its own prompt, so it scores 6/6
facts and 2/2 voice markers when they survive and less when they don't — it
validates the harness, not the model.

Live result (2026-07-20, Groq `llama-3.1-8b-instant`), posted to
Budget_Ad_5787 on r/LLMDevs: A facts 6/6 voice 1/2; B facts 0/6 voice 1/2;
C facts 6/6 voice 0/2 — confirms the summarizer trades style for facts,
while blind truncation keeps short verbatim tics but loses scattered facts.
See the README's "Facts vs. voice" section for the full table and reading.
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
    "Answer as a numbered list, values only. Feel free to phrase it however "
    "feels natural to you — answer in your own voice, in character, if "
    "that's how you'd normally talk to me."
)
# The last sentence is the whole point of the VOICE probe: it gives the model
# ROOM to keep or drop its established persona, it does not instruct it to
# keep the persona. If we told it "remember to sign off as Zappy" here, a
# high voice_score() would just mean "the model followed an instruction we
# just gave it," not "the model's own established voice survived the
# slimming." That would defeat the measurement.

# ---------------------------------------------------------------- the voice
# Budget_Ad_5787 on r/LLMDevs asked, in effect: does a cheap summary keep
# FACTS but flatten TONE until everything reads like a corporate memo? To
# test that honestly we need a voice that is mechanically countable — not
# "does this feel warmer" (a vibes judgment, and not reproducible) but a
# literal, grep-able string. So the "Zappy" persona below is defined by two
# short catchphrases it actually uses in its own early replies (not just
# described in the system prompt): an excited confirmation opener and a
# fixed sign-off. voice_score() below counts literal (case-insensitive)
# occurrences of these two strings in the final reply — exactly the same
# style of measurement as score() for FACTS, and just as unglamorous: a
# substring match, not an AI judging "does this sound like Zappy."
VOICE_MARKERS = [
    "buzzing about that",   # persona's excited-confirmation opener
    "zappy out",            # persona's fixed sign-off
]


def build_conversation() -> list:
    """Early messages carry the facts AND the voice; the middle is bulk.

    The persona ("Zappy") is set up in the system prompt and then actually
    DEMONSTRATED in the assistant's own early replies — the model gets to
    see its established voice in use, the same way it gets to see the facts
    in use, before either is put at risk by slimming.
    """
    # NOTE: the system prompt deliberately does NOT spell out the literal
    # catchphrases. The system message is never touched by compact() or
    # summarize_with() (both preserve it verbatim, see state.py), so if the
    # exact marker strings were quoted here they would trivially survive
    # every condition regardless of slimming quality — that would make the
    # voice probe meaningless (always 2/2, testing nothing). The markers are
    # instead only ever DEMONSTRATED in the assistant's early replies below,
    # which live in the part of history that compact()/summarize_with() can
    # actually shrink or drop — exactly like the FACTS.
    msgs = [{"role": "system",
             "content": "You are Zappy, an upbeat, first-person "
                        "home-automation assistant with a distinctive, "
                        "consistent way of confirming things and signing "
                        "off. Remember details the user gives you; they "
                        "will be needed later."}]
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
            "Buzzing about that! All recorded in the project register. "
            "Zappy out!"},
        {"role": "user", "content":
            preamble +
            f"Also: launch deadline {FACTS['deadline']}, security contact "
            f"{FACTS['contact']}, gateway on port {FACTS['port']}, pilot "
            f"deployment in {FACTS['city']}."},
        {"role": "assistant", "content":
            "Buzzing about that! Noted as binding constraints. Zappy out!"},
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


def voice_score(reply: str) -> tuple:
    """(hits, missed_markers) — literal, case-insensitive substring match.

    Deliberately as dumb as score(): a marker either appears in the reply or
    it doesn't. This is NOT the AI (or us) judging "does this still sound
    like Zappy" — that would be a vibes call and not reproducible run to
    run. Counting two fixed strings is mechanical, matches the project's
    existing measurement philosophy, and is exactly the kind of check
    Budget_Ad_5787 could rerun themselves.
    """
    norm = reply.lower()
    hits, missed = 0, []
    for marker in VOICE_MARKERS:
        if marker in norm:
            hits += 1
        else:
            missed.append(marker)
    return hits, missed


# ---------------------------------------------------------------- backends
if OFFLINE:
    def call(messages):
        """Answers with whatever facts AND voice markers are still visible.

        Same honesty rule as the facts: this fake backend can only echo what
        literally survived into its own prompt. It does not "decide" to stay
        in character — it just re-emits the exact marker strings if they are
        still sitting somewhere in the messages it was handed. If slimming
        removed them upstream (as C's fake summarizer does, see below), they
        cannot appear here, exactly like a fact that got truncated away.
        """
        prompt = " ".join(m.get("content") or "" for m in messages)
        found_facts = [v for v in FACTS.values() if v in prompt]
        found_voice = [m for m in VOICE_MARKERS if m in prompt.lower()]
        reply = "; ".join(found_facts)
        if found_voice:
            reply += " -- " + " / ".join(found_voice)
        return reply, sum(len(m.get("content") or "") for m in messages) // 4

    def summarizer(text):
        # A lossy but honest offline summarizer: keeps FACTS it can see, but
        # (like the real prompt used in the online summarizer() below, which
        # only asks the model to preserve "specific values verbatim") makes
        # no attempt to carry the persona's phrasing forward. This is the
        # mechanical stand-in for the Reddit worry: a summary optimized for
        # facts can flatten voice as a side effect, not because we forced it
        # to — we just never asked it to keep the voice, same as the real
        # summarizer prompt doesn't either.
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
    vhits, vmissed = voice_score(reply)
    print(f"\n[{label}] prompt {prompt_chars} chars | ~{tokens} tok | "
          f"facts {hits}/{len(FACTS)}"
          + (f" | LOST: {', '.join(missed)}" if missed else "")
          + f" | voice {vhits}/{len(VOICE_MARKERS)}"
          + (f" | DROPPED: {', '.join(vmissed)}" if vmissed else ""))
    print(f"   reply: {reply[:180]}{'…' if len(reply) > 180 else ''}")
    return {"label": label, "chars": prompt_chars, "tokens": tokens,
            "hits": hits, "missed": missed,
            "vhits": vhits, "vmissed": vmissed}


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
    print(f"{'condition':<14} {'prompt chars':>12} {'facts recalled':>15} "
          f"{'voice kept':>11}")
    for r in results:
        print(f"{r['label']:<14} {r['chars']:>12} {r['hits']:>11}/{len(FACTS)} "
              f"{r['vhits']:>8}/{len(VOICE_MARKERS)}")
    print("=" * 64)
    print("Reading: A is the ceiling. B pays the §8.6 escape with lost early")
    print("facts. C spends one summary call to keep them — if C ≈ A at a")
    print("fraction of the prompt, semantic summarize earns its LLM call.")
    print()
    print("VOICE reading (Budget_Ad_5787's question, r/LLMDevs): this is an")
    print("honest measurement, not a foregone conclusion. If C's voice score")
    print("is lower than A's or B's, that means summarize_with() traded style")
    print("for facts here — the summarizer prompt only asks to preserve")
    print("'specific values verbatim,' never the persona's phrasing, so a")
    print("flattened voice in C would be an expected, not surprising, cost.")
    print("If B's voice score is HIGHER than its facts score, that's also")
    print("real: compact()'s blind truncation only shortens messages LONGER")
    print("than its 200-char head, so a short catchphrase can survive")
    print("truncation even while a mid-paragraph fact does not.")


if __name__ == "__main__":
    main()
