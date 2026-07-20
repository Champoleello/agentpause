"""Fork & migration demo (F11.2) — runnable WITHOUT any API key.

A suspended checkpoint is inert data: a frozen process image. This demo
treats it the way an OS treats a process:

1. "Machine A" runs the first 3 steps of a plan, then suspends (checkpoint).
2. MIGRATION — the checkpoint is exported as a plain json bundle, shipped
   "over the wire" (json.dumps/loads), and imported into "machine B" (a
   different StateStore directory). B resumes from the exact step: no work
   redone. Machine-local KV-cache blobs, when present in self-hosted setups,
   deliberately stay behind — resume is a logical warm start.
3. FORK — from that one suspended past, B clones two independent futures
   ("plan-cautious" and "plan-bold") and steers each with a different prompt.
   The branches share the same first 3 steps and diverge from there.

Run:  python examples/migrate_fork.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agentpause import PredictiveScheduler, StateStore


# --- a deterministic simulated LLM backend (no keys, no network) ------------
def fake_backend(messages):
    n = sum(1 for m in messages if m["role"] == "assistant") + 1
    reply = f"[reply {n} to: {messages[-1]['content']}]"
    tokens = 120 + 10 * len(messages)   # cost grows with the conversation
    return reply, tokens


def make_scheduler(store: StateStore) -> PredictiveScheduler:
    # a huge window: nothing forces a suspend, machine A suspends voluntarily
    return PredictiveScheduler(backend=fake_backend,
                               telemetry=lambda: 1_000_000,
                               store=store)


PLAN = [
    "Survey the terrain.",
    "List the constraints.",
    "Draft a strategy outline.",
]


def main() -> None:
    dir_a = tempfile.mkdtemp(prefix="agentpause-machine-a-")
    dir_b = tempfile.mkdtemp(prefix="agentpause-machine-b-")

    # --- (a) machine A: run 3 steps, then suspend ---------------------------
    print("== Machine A ==")
    store_a = StateStore(dir_a)
    with make_scheduler(store_a).session("mission") as s:
        for question in PLAN[s.step:]:
            s.add_user(question)
            print(f"  step {s.step + 1}: {s.call()}")
        s.checkpoint()
        print(f"  suspended at step {s.step} "
              f"({s.total_tokens_used} tokens spent). Machine A shuts down.\n")

    # --- (b) MIGRATION: export from A, import into B, resume there ----------
    print("== Migration A -> B ==")
    bundle = store_a.export_bundle("mission")
    wire = json.dumps(bundle)               # portable: any transport works
    print(f"  bundle format {bundle['format']}, {len(wire)} bytes on the wire")

    store_b = StateStore(dir_b)
    store_b.import_bundle(json.loads(wire))
    sched_b = make_scheduler(store_b)
    with sched_b.session("mission") as s:
        print(f"  machine B resumed={s.resumed} at step {s.step} "
              f"({s.total_tokens_used} tokens already accounted) "
              f"-> no work redone\n")

    # --- (c) FORK: one suspended past, two independent futures --------------
    print("== Fork on machine B ==")
    branches = [
        ("plan-cautious", "Refine the strategy to minimize risk."),
        ("plan-bold", "Refine the strategy to maximize speed."),
    ]
    for fork_id, prompt in branches:
        store_b.fork("mission", fork_id)
        with sched_b.session(fork_id) as s:
            assert s.resumed and s.step == 3   # inherits the shared past
            s.add_user(prompt)
            s.call()
            print(f"  [{fork_id}] resumed at step 3, now step {s.step}:")
            for m in s.messages:
                print(f"    {m['role']:9s}| {m['content']}")
            s.complete()
        print()

    print("Both branches shared the same first 3 steps and diverged from "
          "step 4 — independent state, independent idempotency namespaces.")


if __name__ == "__main__":
    main()
