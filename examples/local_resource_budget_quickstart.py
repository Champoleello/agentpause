"""Plug-and-play quickstart: LocalResourceBudget, the RECOMMENDED way to wire
up local-resource budgets for a self-hosted llama.cpp deployment.

This is the one-line-setup counterpart to ``examples/local_resources_quickstart.py``,
which shows the same five signals wired by hand, one piece at a time, for
anyone who wants fine control over a single axis (a custom ``used_field=``, a
hand-picked ``bytes_per_token``, ...). Read THIS file first if you just want
things to work; read the other one if you need to override a default.

Runnable WITHOUT a real llama-server or GPU: ``FakeLlamaCppSlots`` below is a
complete, in-memory double for :class:`agentpause.llamacpp_kv.LlamaCppSlots`
(all five methods it needs: ``get_props``, ``get_slot``, ``fingerprint``,
``save``, ``restore``), and a fake GPU reader stands in for ``pynvml``. Both
follow the exact same fake-transport discipline as
``examples/kv_llamacpp_demo.py`` and the test suite -- nothing here talks to
a real socket.

The story, in order:

  1. Construct ``LocalResourceBudget(kv_dir=...)`` -- ONE call. It reads the
     server's real ``n_ctx`` immediately (fails loudly right here if the
     server is unreachable, not three steps into a run) and best-effort
     detects whether a GPU is available. Nothing about VRAM, KV-save timing,
     or a disk-space threshold is calibrated yet -- there is no real save to
     calibrate them from.
  2. Get a ``KVStateStore`` from it via ``.kv_store(store)`` and use it
     completely normally. The FIRST real ``save_with_kv(...)`` call
     automatically calibrates ``bytes_per_token``, the measured save
     duration, and (unless you asked for a specific value yourself) a
     disk-space guard threshold -- all from real numbers this save just
     produced, never a synthetic probe.
  3. Call the ``LocalResourceBudget`` instance itself as ``telemetry=`` for a
     real ``PredictiveScheduler``. Before step 2 happens, it quietly reports
     context alone (still a complete, honest budget). After step 2, VRAM
     joins automatically -- no code change, no re-wiring.
  4. Run a short session, save-and-suspend partway through, and print what
     changed after calibration -- notice the log line at the moment
     ``kv_store().save_with_kv()`` is first called with real data.

Run:  python examples/local_resource_budget_quickstart.py
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agentpause import Checkpoint, PredictiveScheduler, StateStore
from agentpause.adapters.local_resources import LocalResourceBudget
from agentpause.errors import KVError


class FakeLlamaCppSlots:
    """Complete in-memory double for LlamaCppSlots -- all five methods
    LocalResourceBudget touches (get_props, get_slot, fingerprint, save,
    restore). No network. ``grow(tokens)`` simulates the slot's KV cache
    accumulating tokens turn by turn, same as a real conversation would.
    """

    def __init__(self, save_dir: str, n_ctx: int = 8192,
                 model: str = "models/qwen3-8b-q4_k_m.gguf") -> None:
        self.save_dir = save_dir
        self.n_ctx = n_ctx
        self.model = model
        self.used_tokens = 0
        self.save_calls = 0
        self.restore_calls = 0

    def grow(self, tokens: int) -> None:
        self.used_tokens += tokens

    def get_props(self, base_url: str):
        return {"default_generation_settings": {"n_ctx": self.n_ctx},
                "model_path": self.model}

    def get_slot(self, base_url: str, id_slot: int):
        # n_prompt_tokens -- CONFIRMED live 2026-07-21 against a real
        # llama-server as the correct field (see local_resources.py's module
        # docstring for the measured evidence).
        return {"id": id_slot, "n_prompt_tokens": self.used_tokens,
                "is_processing": False}

    def fingerprint(self, base_url: str) -> str:
        return self.model

    def save(self, base_url: str, id_slot: int, filename: str) -> int:
        # blob size scales with tokens used, so the bytes_per_token this demo
        # calibrates is meaningful and hand-verifiable, not a fixed constant.
        os.makedirs(self.save_dir, exist_ok=True)
        n_saved = max(1, self.used_tokens)
        with open(os.path.join(self.save_dir, filename), "wb") as f:
            f.write(b"x" * (n_saved * 147_475))  # ~ the real-world byte/token
                                                   # ratio measured for
                                                   # Qwen3-8B-GGUF Q4_K_M
        self.save_calls += 1
        return n_saved

    def restore(self, base_url: str, id_slot: int, filename: str) -> int:
        self.restore_calls += 1
        return max(1, self.used_tokens)


def fake_gpu_reader(device_index: int):
    """Stands in for pynvml.nvmlDeviceGetMemoryInfo: (free_bytes, total_bytes)
    on a fake 24 GB card with 18 GB free -- plenty of headroom for this demo,
    so the story below is driven by CONTEXT running out, not VRAM."""
    return (18_000_000_000, 24_000_000_000)


def main() -> None:
    print("=" * 78)
    print("  LocalResourceBudget: the plug-and-play way (one call, not six)")
    print("=" * 78)

    work_dir = tempfile.mkdtemp(prefix="agentpause-local-budget-demo-")
    kv_dir = os.path.join(work_dir, "kv_cache")
    fake_slots = FakeLlamaCppSlots(save_dir=kv_dir, n_ctx=2048)

    # -- step 1: ONE call. Reads n_ctx immediately, detects GPU availability,
    # nothing else calibrated yet (no real save has happened).
    print("\n-- (1) LocalResourceBudget(kv_dir=...) -- the entire setup --")
    local = LocalResourceBudget(
        kv_dir=kv_dir,
        slots=fake_slots,                 # real use: omit this, a real
                                            # LlamaCppSlots() is built for you
        gpu_reader_fn=fake_gpu_reader,     # real use: omit this too, pynvml
                                            # is tried for real
        time_budget_s=120.0,
        hourly_cost=0.0084,                # ~30W laptop @ EUR0.28/kWh
        safety_margin_tokens=100,
    )
    print(f"  gpu_available={local.gpu_available}  (before any real save: "
          f"VRAM axis is NOT yet part of the composite telemetry)")

    # -- step 2: normal use. Get a KVStateStore, use it exactly as documented
    # elsewhere in the README -- the calibration happens on its own.
    print("\n-- (2) local.kv_store(...) -- use it completely normally --")
    kv_store = local.kv_store(StateStore(os.path.join(work_dir, ".agentpause")))

    def fake_backend(messages):
        return "(fake local model reply)", 250

    sched = PredictiveScheduler(backend=fake_backend, telemetry=local)

    with sched.session("local-budget-demo") as s:
        for i in range(1, 9):
            s.add_user(f"Question {i} about the ongoing plan.")

            reading = local()  # exactly what the scheduler itself reads
            print(f"\n  step {i}: remaining_tokens={reading.remaining_tokens} "
                  f"(gpu_available={local.gpu_available}, "
                  f"bytes_per_token={'n/a' if local._bytes_per_token is None else f'{local._bytes_per_token:.1f}'})")

            if s.should_suspend():
                print(f"    -> checkpoint: local context ran out at step {i}")
                s.checkpoint()
                break

            reply = s.call()
            fake_slots.grow(250)  # this step's tokens joined the KV cache

            if i == 3:
                # First real checkpoint save -- this is the ONE moment
                # calibration happens. Nothing special about step 3; any
                # real save would trigger it the same way.
                print(f"\n  -- (3) first real save_with_kv() call: watch it "
                      f"calibrate on its own --")
                cp = Checkpoint(session_id="local-budget-demo-manual-save")
                cp.messages = list(s.messages)
                try:
                    kv_store.save_with_kv(cp)
                except KVError as exc:
                    print(f"     (unexpected in this demo: {exc})")
        else:
            s.complete()
            print("\n  (ran to completion without ever hitting a local ceiling)")

    print("\n" + "=" * 78)
    print(f"After calibration: gpu_available={local.gpu_available}, "
          f"bytes_per_token={local._bytes_per_token}, "
          f"measured_save_s={local._measured_save_s}")
    print("From this point on, local() automatically includes the VRAM axis")
    print("too -- no code change, because __call__() checks calibration state")
    print("fresh on every read.")
    print("=" * 78)


if __name__ == "__main__":
    main()
