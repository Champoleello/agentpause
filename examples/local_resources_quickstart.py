"""Local-resource budgets quickstart: context + VRAM + KV-aware time + a
derived local price, composed into ONE real telemetry callable.

Runnable WITHOUT a real llama-server or GPU -- every transport this demo
touches is a small, deterministic, in-memory fake, same discipline as
``examples/kv_llamacpp_demo.py``'s ``FakeSlots``: ``FakeLlamaCppSlots`` stands
in for :class:`agentpause.llamacpp_kv.LlamaCppSlots` (same
``get_props``/``get_slot`` interface) and a plain mutable ``FakeGPU`` stands in
for the hardware ``pynvml`` would otherwise read. The estimator
(:class:`agentpause.regression.FeatureEstimator`) is the REAL class -- it is
pure computation, no I/O, so nothing needs faking there; it is only fed a
handful of made-up ("fittizio") prior steps so it has enough history to
predict a throughput from the start.

The story, in order:

  1. ``LlamaCppContextBudget`` over a ``FakeLlamaCppSlots`` whose reported
     KV-cache usage grows a little every step, simulating a real conversation
     slowly eating into a fixed ``--ctx-size`` window.
  2. Composed, via ``CompositeLocalBudget``, with a ``GPUMemoryBudget`` over a
     ``FakeGPU`` whose free memory shrinks every step too (another process
     resident on the same card is also growing its own allocation).
  3. That composite wrapped in a ``KVAwareTimeBudget``, which takes DIRECT
     ownership of the wall-clock dimension and reserves time up front
     (``estimated_kv_save_s``) to save the KV-cache blob before a deadline
     actually hits.
  4. A ``price_per_1k_tokens`` derived from the estimator's own learned
     throughput (``price_per_1k_tokens_from_estimator``) and an assumed
     electricity cost (``estimate_hourly_cost_from_power``) -- wired into
     ``PredictiveScheduler``'s EXISTING ``price_per_1k_tokens``/
     ``money_budget`` parameters, exactly as they already work for a cloud
     provider (no new scheduler mechanism needed for this part).
  5. A REAL ``PredictiveScheduler`` (nothing faked at this layer) driven by a
     ``CompositeLocalBudget`` as its ``telemetry=``, run step by step with the
     library's usual ``session()`` / ``should_suspend()`` / ``call()`` /
     ``checkpoint()`` cycle, until one of the LOCAL signals runs out (never a
     cloud rate limit -- there is none here) and the decision flips to
     ``checkpoint``.

The numbers below (bytes-per-token, context size, VRAM) are illustrative and
round for readability, NOT a real calibration -- see ``GPUMemoryBudget``'s own
docstring for how to derive a real ``bytes_per_token`` from an actual KV save.

Run:  python examples/local_resources_quickstart.py
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agentpause import PredictiveScheduler, StateStore
from agentpause.adapters.local_resources import (
    CompositeLocalBudget,
    GPUMemoryBudget,
    KVAwareTimeBudget,
    LlamaCppContextBudget,
    estimate_hourly_cost_from_power,
    price_per_1k_tokens_from_estimator,
)
from agentpause.regression import FeatureEstimator


class FakeLlamaCppSlots:
    """In-memory double for LlamaCppSlots -- only the two read methods
    LlamaCppContextBudget calls (``get_props``/``get_slot``); no network.

    ``grow(tokens)`` simulates the slot's KV cache accumulating more tokens as
    the conversation continues, exactly what a real llama-server's
    ``GET /slots`` response would show turn by turn (see the "guessed field"
    discussion in ``adapters/local_resources.py``'s module docstring --
    ``cache_tokens`` is used here as the (unconfirmed) default).
    """

    def __init__(self, n_ctx: int) -> None:
        self.n_ctx = n_ctx
        self.used_tokens = 0

    def grow(self, tokens: int) -> None:
        self.used_tokens += tokens

    def get_props(self, base_url: str):
        return {"default_generation_settings": {"n_ctx": self.n_ctx}}

    def get_slot(self, base_url: str, id_slot: int):
        return {"id": id_slot, "cache_tokens": list(range(self.used_tokens))}


class FakeGPU:
    """Tiny mutable stand-in for a GPU's VRAM state -- no pynvml, no real
    hardware. ``consume(n)`` shrinks free memory, standing in for another
    process on the same card slowly growing its own allocation over time."""

    def __init__(self, total_bytes: int, free_bytes: int) -> None:
        self.total_bytes = total_bytes
        self.free_bytes = free_bytes

    def consume(self, amount: int) -> None:
        self.free_bytes = max(0, self.free_bytes - amount)

    def read(self, device_index: int):
        return (self.free_bytes, self.total_bytes)


def main() -> None:
    print("=" * 78)
    print("  Local resource budgets: context + VRAM + KV-aware time + price")
    print("=" * 78)

    # -- (1) LlamaCppContextBudget over a fake, growing KV-cache slot ----------
    print("\n-- (1) LlamaCppContextBudget: a fake llama.cpp slot near its --ctx-size --")
    n_ctx = 8192
    fake_slots = FakeLlamaCppSlots(n_ctx=n_ctx)
    context_budget = LlamaCppContextBudget(
        fake_slots, base_url="http://127.0.0.1:8080", id_slot=0,
        safety_margin_tokens=200,
    )
    print(f"  n_ctx={n_ctx}, safety_margin_tokens=200, starts empty")

    # -- (2) composed with a fake GPUMemoryBudget: VRAM shrinking separately --
    print("\n-- (2) + GPUMemoryBudget: a fake GPU whose free VRAM shrinks too --")
    fake_gpu = FakeGPU(total_bytes=24_000_000_000, free_bytes=1_200_000_000)
    gpu_budget = GPUMemoryBudget(
        device_index=0, bytes_per_token=1_000_000.0, reader_fn=fake_gpu.read,
    )
    print(f"  total=24 GB, free starts at 1.2 GB, bytes_per_token=1e6 (illustrative)")

    local_signals = CompositeLocalBudget(context_budget, gpu_budget)
    print("  local_signals = CompositeLocalBudget(context_budget, gpu_budget)")
    print("  -- reports whichever of the two runs out FIRST, not an average")

    # -- (3) the whole thing wrapped in a KVAwareTimeBudget --------------------
    print("\n-- (3) wrapped in KVAwareTimeBudget: reserve time for the KV blob save --")
    kv_aware = KVAwareTimeBudget(
        inner_telemetry=local_signals,
        time_budget_s=600.0,
        estimated_kv_save_s=45.0,
    )
    print("  time_budget_s=600s, estimated_kv_save_s=45s reserved up front")

    # Wrapping the wrapped signal in ANOTHER CompositeLocalBudget keeps the
    # scheduler's telemetry= uniformly a CompositeLocalBudget even though
    # there is only one source right now -- this is exactly the "single
    # callable" case CompositeLocalBudget is explicitly designed (and
    # tested) to support, and it means a second local signal (a second GPU,
    # a disk-space budget, ...) could be added later with a one-line change
    # here, not a restructure.
    telemetry = CompositeLocalBudget(kv_aware)

    # -- (4) price_per_1k_tokens derived from the estimator's own throughput --
    print("\n-- (4) price_per_1k_tokens derived from a FeatureEstimator's throughput --")
    estimator = FeatureEstimator()
    for _ in range(6):  # a handful of made-up ("fittizio") prior steps, just
        # enough history (min_samples=6) for estimate_latency() to stop
        # returning None -- a real deployment would have this from actual
        # completed steps instead of a seeded loop.
        estimator.record(input_tokens=1500, realized=300, latency=2.0)

    hourly_cost = estimate_hourly_cost_from_power(watts=350.0, price_per_kwh=0.28)
    price = price_per_1k_tokens_from_estimator(
        estimator, input_tokens=1500, hourly_cost=hourly_cost,
    )
    print(f"  hourly_cost (350W @ $0.28/kWh) = ${hourly_cost:.4f}/h")
    print(f"  derived price_per_1k_tokens = ${price:.6f}  "
          f"(fed into PredictiveScheduler's EXISTING price_per_1k_tokens/"
          f"money_budget -- no new mechanism)")

    # -- (5) a REAL PredictiveScheduler driven by CompositeLocalBudget ---------
    print("\n-- (5) a real PredictiveScheduler, telemetry=CompositeLocalBudget(...) --")

    def fake_backend(messages):
        # stands in for the real llama.cpp inference call: fixed reply,
        # fixed token cost, so the demo's numbers stay hand-verifiable.
        return "(fake local model reply)", 300

    store_dir = tempfile.mkdtemp(prefix="agentpause-local-resources-demo-")
    sched = PredictiveScheduler(
        backend=fake_backend,
        telemetry=telemetry,
        estimator=estimator,
        price_per_1k_tokens=price,
        money_budget=1.0,
        store=StateStore(store_dir),
    )

    with sched.session("local-resources-demo") as s:
        for i in range(1, 13):
            s.add_user(f"Question {i} about the ongoing plan.")

            # read the two underlying signals directly too, purely to narrate
            # which one is closest to binding this step -- the scheduler
            # itself only ever sees the merged CompositeLocalBudget reading.
            ctx_reading = context_budget()
            gpu_reading = gpu_budget()
            tighter = "context" if ctx_reading.remaining_tokens <= gpu_reading.remaining_tokens else "GPU VRAM"
            print(f"\n  step {i}: context remaining={ctx_reading.remaining_tokens} "
                  f"tok, GPU remaining={gpu_reading.remaining_tokens} tok "
                  f"(tighter: {tighter})")

            decision = s.next_action()   # next_action().action != "continue" IS should_suspend()
            print(f"    decision={decision.action!r} estimated={decision.estimated} tok, "
                  f"composite remaining_tokens={decision.budget.remaining_tokens}, "
                  f"remaining_seconds={decision.budget.remaining_seconds:.1f}")

            if s.should_suspend():
                print(f"    -> checkpoint: a local signal ran out ({tighter}), "
                      f"not a cloud rate limit -- there is none here")
                s.checkpoint()
                break

            reply = s.call()
            fake_slots.grow(300)          # this step's tokens joined the KV cache
            fake_gpu.consume(200_000_000)  # ...and so did some more VRAM
        else:
            s.complete()
            print("\n  (ran to completion without ever hitting a local ceiling)")

    print("\n" + "=" * 78)
    print("Done: the composite caught the binding local constraint (context or")
    print("VRAM, whichever ran out first) and the scheduler checkpointed instead")
    print("of running past it -- exactly as it would for a cloud rate limit, but")
    print("driven entirely by real local-resource signals instead of headers.")
    print("=" * 78)


if __name__ == "__main__":
    main()
