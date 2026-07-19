"""Fleet quickstart: BudgetRouter + MultiAgentCoordinator + FeatureEstimator.

Runnable WITHOUT any API key — providers are tiny in-memory fakes that mimic
the adapter shape (``backend(messages) -> (reply, tokens)`` and
``budget() -> Budget``). It shows the three v0.3 pieces composing:

1. **BudgetRouter (F9.1)** — reads every provider's telemetry BEFORE calling
   and routes to the one with the most headroom (predictive, not reactive).
2. **MultiAgentCoordinator (D5)** — two agents share ONE rate-limit window;
   each granted call reserves its predicted cost from the shared pool, so the
   agents can't blow the window together.
3. **FeatureEstimator (D6)** — learns token cost from features (which tool,
   context size), not just a moving average; feeds the coordinator's requests.

Run:  python examples/fleet_quickstart.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agentpause import BudgetRouter, FeatureEstimator, MultiAgentCoordinator
from agentpause.risk import Budget


class FakeProvider:
    """Mimics an adapter: a token bucket that drains as you call it."""

    def __init__(self, name: str, window: int) -> None:
        self.name = name
        self.remaining = window
        self.window = window

    def backend(self, messages):
        # cost grows with the conversation, like a real prefill
        cost = 200 + 50 * len(messages)
        self.remaining = max(0, self.remaining - cost)
        return f"[{self.name}] reply #{len(messages)}", cost

    def budget(self) -> Budget:
        return Budget(
            remaining_tokens=self.remaining,
            remaining_requests=100,
            reset_seconds=60.0,
            limit_tokens=self.window,
        )


def main() -> None:
    # --- 1. Router over two providers with different headroom ---------------
    groq_like = FakeProvider("groq-ish", window=3_000)
    openai_like = FakeProvider("openai-ish", window=4_500)
    router = BudgetRouter(("groq-ish", groq_like), ("openai-ish", openai_like))

    # --- 2. One shared window, two agents ------------------------------------
    coord = MultiAgentCoordinator(telemetry=router.budget, k=2.0)
    coord.register("researcher", priority=1)   # higher priority wins contention
    coord.register("summarizer", priority=0)

    # --- 3. A feature-aware estimator per agent ------------------------------
    est = {a: FeatureEstimator() for a in ("researcher", "summarizer")}

    messages = [{"role": "user", "content": "map the field of LLM scheduling"}]
    for step in range(1, 9):
        for agent in ("researcher", "summarizer"):
            tool = "web_search" if agent == "researcher" else "summarize"
            est[agent].set_context(tool=tool)

            predicted = est[agent].estimate(input_tokens=200 + 50 * len(messages))
            sig = est[agent].sigma(fallback_estimate=predicted)

            d = coord.request(agent, estimated=predicted, sigma=sig)
            if d.action != "continue":
                print(f"step {step} | {agent:<11} -> {d.action.upper()} "
                      f"(pool has {d.budget.remaining_tokens} tok, "
                      f"needs ~{predicted}+{sig:.0f})")
                continue

            reply, used = router.backend(messages)      # router picks provider
            coord.complete(agent, actual_tokens=used)
            est[agent].record(input_tokens=200 + 50 * len(messages), realized=used)
            messages.append({"role": "assistant", "content": reply})
            print(f"step {step} | {agent:<11} -> ok, {used} tok via {reply.split(']')[0][1:]}")

    print("\nDone. The router drained the big window first and switched to the")
    print("small one; when the shared pool got too thin for the predicted cost,")
    print("the coordinator said WAIT — before any provider could return a 429.")


if __name__ == "__main__":
    main()
