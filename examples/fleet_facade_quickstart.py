"""Plug-and-play quickstart: AgentFleet, the RECOMMENDED way to share
providers and a rate-limit window across several agents.

This is the one-call-setup counterpart to ``examples/fleet_quickstart.py``,
which wires the same three pieces (BudgetRouter, MultiAgentCoordinator, an
estimator per agent) by hand, one at a time, for anyone who wants fine
control over a single axis (a custom ``estimator_factory=``, direct
``coordinator.arbitrate()`` batch calls, ...). Read THIS file first if you
just want a fleet of agents to share providers and a window safely; read the
other one if you need to override a default or see the manual six-step
recipe the README's "Scaling up" section documents.

Runnable WITHOUT any API key: ``FakeProvider`` below is the same tiny,
in-memory double ``examples/fleet_quickstart.py`` and the test suite use
(``backend(messages) -> (reply, tokens)``, ``budget() -> Budget``) — nothing
here talks to a real socket.

The story, in order:

  1. Construct ``AgentFleet(providers, agents=[...])`` -- ONE call. It builds
     the router and the coordinator together (same clock, coordinator wired
     to the router's own budget) and gives each agent its own, isolated
     estimator.
  2. Drive every step through ``fleet.call(agent_id, messages)`` -- the
     estimate -> ask -> call -> reconcile -> record dance collapses into one
     method. On a non-"continue" decision nothing is called and nothing is
     recorded; you still see the same WAIT/CHECKPOINT signal
     ``coordinator.request()`` would have given you by hand.
  3. Two agents draw down a SHARED, shrinking pool: watch a lower-priority
     agent get turned away once the pool gets too thin for its own predicted
     cost, while a higher-priority one keeps going.
  4. Power-user access: ``fleet.router``/``fleet.coordinator`` are the real,
     live objects -- drop to ``fleet.coordinator.arbitrate(...)`` for the
     batch/contention case exactly as if you had built these by hand.

What this does NOT do (see ``AgentFleet``'s own docstring for the full
scope note): no checkpoint/resume for fleets. If you need that per agent,
pair this with your own ``StateStore`` per ``agent_id``, exactly like
single-agent usage already does elsewhere in the README.

Run:  python examples/fleet_facade_quickstart.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agentpause import AgentFleet
from agentpause.risk import Budget


class FakeProvider:
    """Mimics an adapter: a token bucket that drains as you call it.

    Same shape as examples/fleet_quickstart.py's FakeProvider: a real
    adapter (LiteLLMAdapter, OpenAICompatAdapter, AnthropicAdapter, ...)
    drops in here unchanged -- AgentFleet only needs .backend()/.budget().
    """

    def __init__(self, name: str, window: int) -> None:
        self.name = name
        self.remaining = window
        self.window = window

    def backend(self, messages):
        cost = 200 + 50 * len(messages)   # cost grows with the conversation
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
    print("=" * 78)
    print("  AgentFleet: the plug-and-play way (one call, not six)")
    print("=" * 78)

    # -- step 1: ONE call. Router + coordinator built together, two agents
    # registered (priorities set here matter for arbitrate(), step 4 below —
    # the streaming .call() loop in step 2 has no priority ordering of its
    # own: whichever agent asks first in your code gets served first against
    # whatever the shared pool has left at that instant).
    print("\n-- (1) AgentFleet(providers, agents=...) -- the entire setup --")
    groq_like = FakeProvider("groq-ish", window=1_500)
    openai_like = FakeProvider("openai-ish", window=2_000)
    fleet = AgentFleet(
        [("groq-ish", groq_like), ("openai-ish", openai_like)],
        agents=[("researcher", 1), ("summarizer", 0)],   # (agent_id, priority)
    )
    print(f"  router providers: groq-ish (window={groq_like.window}), "
          f"openai-ish (window={openai_like.window})")
    print("  agents: researcher (priority=1), summarizer (priority=0)")

    # -- step 2 & 3: drive every step through .call(). Two agents draw down
    # the SAME shared pool as the conversation grows; watch BOTH get a clean
    # checkpoint the moment the pool can no longer fit either one's own
    # predicted next-step cost — the whole point of the shared-pool
    # accounting: nobody sneaks past the window into a real 429.
    print("\n-- (2)+(3) fleet.call(agent_id, messages) each step --")
    messages = {
        "researcher": [{"role": "user", "content": "map the field of LLM scheduling"}],
        "summarizer": [{"role": "user", "content": "summarize what we have so far"}],
    }
    for step in range(1, 9):
        for agent in ("researcher", "summarizer"):
            d, reply, used = fleet.call(agent, messages[agent])
            if d.action != "continue":
                print(f"  step {step} | {agent:<11} -> {d.action.upper()} "
                      f"(pool has {d.budget.remaining_tokens} tok, "
                      f"needs ~{d.estimated}+{d.sigma:.0f})")
                continue
            messages[agent].append({"role": "assistant", "content": reply})
            via = reply.split(']')[0][1:]
            print(f"  step {step} | {agent:<11} -> ok, {used} tok via {via} "
                  f"(samples so far: {fleet.estimator_for(agent).samples})")

    # -- step 4: power-user access -- the real objects, not copies. The
    # streaming loop above never used priority (there's nothing to arbitrate
    # one call at a time); a SIMULTANEOUS batch is where priority=1 vs
    # priority=0 actually decides who wins. Fresh, smaller pool so it can fit
    # exactly one of the two contenders, not both:
    print("\n-- (4) power-user access: fleet.router / fleet.coordinator --")
    print(f"  fleet.router.last_route = {fleet.router.last_route!r}")

    class FixedProvider:
        """A single provider with a fixed remaining pool, just for this batch."""
        def backend(self, messages):
            return "[fixed] reply", 10
        def budget(self) -> Budget:
            return Budget(remaining_tokens=400, reset_seconds=60.0)

    priority_demo = AgentFleet([FixedProvider()],
                                agents=[("researcher", 1), ("summarizer", 0)])
    batch = priority_demo.coordinator.arbitrate([("researcher", 300, 20.0),
                                                  ("summarizer", 300, 20.0)])
    print(f"  400 tok left, both agents want ~300+20 each (fits only one):")
    print(f"  arbitrate(...) -> {{k: v.action for k, v in batch.items()}} =",
          {k: v.action for k, v in batch.items()})
    print("  researcher (priority=1) wins the shared pool; summarizer "
          "(priority=0) is told to wait/checkpoint instead of overrunning it.")

    print("\n" + "=" * 78)
    print("Same shared-pool safety as the manual six-step recipe in the README's")
    print("'Scaling up' section and examples/fleet_quickstart.py -- just one")
    print("constructor call and one method per step to get there.")
    print("=" * 78)


if __name__ == "__main__":
    main()
