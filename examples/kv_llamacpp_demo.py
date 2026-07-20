"""TRUE (KV-cache) warm start demo, composed with FORK + MIGRATION (F11.2).

Runnable WITHOUT a real llama-server — ``FakeSlots`` is a tiny, deterministic,
in-memory double for :class:`agentpause.llamacpp_kv.LlamaCppSlots` (same
``fingerprint``/``save``/``restore`` interface), so the demo is instant and
needs no network.

The story, in order:

  a. Save a checkpoint WITH kv via ``save_with_kv`` -- n_saved + blob filename.
  b. Load it back via ``load_with_kv`` on a FRESH ``KVStateStore`` pointed at
     the same directories -- true restore, no re-prefill needed.
  c. Simulate a MODEL CHANGE (fake fingerprint differs) -- load again --
     graceful degradation to a LOGICAL warm start, session still resumes at
     the right step.
  d. ``fork_with_kv`` into "plan-cautious" and "plan-bold" -- load BOTH
     independently -- both restore their OWN kv blob, proving independence.
  e. MIGRATION: export/import the original session into a KVStateStore on a
     "different machine" (a different tmp directory) -- the blob never
     travels, so the load degrades gracefully (`kv_file_missing`), yet the
     logical state (step/messages) survives intact.

Run:  python examples/kv_llamacpp_demo.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agentpause import Checkpoint, StateStore
from agentpause.llamacpp_kv import KVStateStore


class FakeSlots:
    """In-memory double for LlamaCppSlots: no network, fully deterministic.

    Mirrors the REAL llama-server's behavior: it resolves the ``filename`` it
    receives against its OWN ``--slot-save-path`` directory, so this fake is
    constructed with that same directory (``save_dir``) and the caller must
    always send a BARE filename -- exactly what a real deployment requires
    ``kv_dir`` (KVStateStore's own bookkeeping directory) to equal.
    """

    def __init__(self, save_dir: str, model: str = "models/llama-3.1-8b-instruct.gguf") -> None:
        self.save_dir = save_dir
        self.model = model

    def fingerprint(self, base_url: str) -> str:
        return self.model

    def save(self, base_url: str, id_slot: int, filename: str) -> int:
        os.makedirs(self.save_dir, exist_ok=True)
        with open(os.path.join(self.save_dir, filename), "wb") as f:
            f.write(b"fake-kv-cells")
        return 4096

    def restore(self, base_url: str, id_slot: int, filename: str) -> int:
        return 4096


def main() -> None:
    base_dir = tempfile.mkdtemp(prefix="agentpause-kv-demo-")
    store_dir = os.path.join(base_dir, "store")
    kv_dir = os.path.join(base_dir, "kv_cache")
    slots = FakeSlots(save_dir=kv_dir)

    kv_store = KVStateStore(StateStore(store_dir), slots=slots,
                            base_url="http://127.0.0.1:8080", id_slot=0,
                            kv_dir=kv_dir)

    print("=" * 78)
    print("  TRUE (KV-cache) warm start for llama.cpp -- fork + migration demo")
    print("=" * 78)

    # -- (a) save a checkpoint WITH kv -----------------------------------------
    print("\n-- (a) save_with_kv --")
    cp = Checkpoint(
        session_id="mission",
        step=3,
        messages=[
            {"role": "system", "content": "You are a terse planning assistant."},
            {"role": "user", "content": "Survey the terrain."},
            {"role": "assistant", "content": "Terrain surveyed: 3 constraints found."},
        ],
        total_tokens_used=900,
    )
    kv_store.save_with_kv(cp)
    kv = cp.extra["kv"]
    print(f"  n_saved={kv['n_saved']} cells, blob file={kv['file']!r}")

    # -- (b) load it back on a FRESH KVStateStore ------------------------------
    print("\n-- (b) load_with_kv (same machine, fresh store instance) --")
    fresh_store = KVStateStore(StateStore(store_dir), slots=slots,
                               base_url="http://127.0.0.1:8080", id_slot=0,
                               kv_dir=kv_dir)
    loaded, info = fresh_store.load_with_kv("mission")
    print(f"  kv_restored={info['kv_restored']}, n_restored={info['n_restored']}")
    print(f"  step={loaded.step} -- resumed with NO re-prefill needed "
          f"(contrast with a logical-only warm start, which would re-run "
          f"the full {len(loaded.messages)}-message prefill)")
    fresh_store.gc_consumed("mission")  # first post-resume step succeeded

    # -- (c) simulate a MODEL CHANGE -------------------------------------------
    print("\n-- (c) simulate a model change on the server --")
    kv_store.save_with_kv(cp)  # re-save so there's a fresh blob to test against
    slots.model = "models/a-completely-different-model.gguf"
    loaded, info = kv_store.load_with_kv("mission")
    print(f"  kv_restored={info['kv_restored']}, reason={info.get('reason')}")
    print(f"  step={loaded.step} -- session still resumes correctly "
          f"(graceful degradation to a LOGICAL warm start)")
    slots.model = "models/llama-3.1-8b-instruct.gguf"  # restore for (d)/(e)

    # -- (d) fork_with_kv: independent blobs per branch ------------------------
    print("\n-- (d) fork_with_kv: plan-cautious / plan-bold --")
    kv_store.save_with_kv(cp)  # fresh blob to fork from
    cautious = kv_store.fork_with_kv("mission", "plan-cautious")
    bold = kv_store.fork_with_kv("mission", "plan-bold")
    print(f"  plan-cautious blob={cautious.extra['kv']['file']!r}")
    print(f"  plan-bold     blob={bold.extra['kv']['file']!r}")
    assert cautious.extra["kv"]["file"] != bold.extra["kv"]["file"]

    loaded_c, info_c = kv_store.load_with_kv("plan-cautious")
    loaded_b, info_b = kv_store.load_with_kv("plan-bold")
    print(f"  plan-cautious: kv_restored={info_c['kv_restored']}, "
          f"n_restored={info_c['n_restored']}, step={loaded_c.step}")
    print(f"  plan-bold:     kv_restored={info_b['kv_restored']}, "
          f"n_restored={info_b['n_restored']}, step={loaded_b.step}")
    print("  both branches restored their OWN kv blob independently.")

    # -- (e) MIGRATION: export the original session to a "different machine" --
    print("\n-- (e) migration: export/import to a different machine --")
    bundle = kv_store.store.export_bundle("mission")
    wire = json.dumps(bundle)   # portable: any transport works
    print(f"  bundle format {bundle['format']}, {len(wire)} bytes on the wire "
          f"(carries extra['kv'] along as plain JSON data)")

    other_machine_dir = os.path.join(base_dir, "machine-b")
    other_store_dir = os.path.join(other_machine_dir, "store")
    other_kv_dir = os.path.join(other_machine_dir, "kv_cache")  # note: EMPTY --
    #                                                              the blob file
    #                                                              never travels
    other_kv_store = KVStateStore(StateStore(other_store_dir), slots=slots,
                                  base_url="http://127.0.0.1:8080", id_slot=0,
                                  kv_dir=other_kv_dir)
    other_kv_store.store.import_bundle(json.loads(wire))

    loaded_migrated, info_migrated = other_kv_store.load_with_kv("mission")
    print(f"  kv_restored={info_migrated['kv_restored']}, "
          f"reason={info_migrated.get('reason')}")
    print(f"  step={loaded_migrated.step}, messages="
          f"{len(loaded_migrated.messages)} -- logical state survives even "
          f"when the accelerator doesn't")

    print("\n" + "=" * 78)
    print("Done: KV-cache true warm start degrades gracefully in every case")
    print("(model mismatch, migration) and never crashes the caller.")
    print("=" * 78)


if __name__ == "__main__":
    main()
