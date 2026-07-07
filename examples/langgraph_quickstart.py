"""agentpause + LangGraph: predictive suspension inside a graph.

Needs:  pip install agentpause[langgraph]
No API key required — the "LLM" is simulated so you can watch the mechanism.

What happens:
  1. the graph runs with a tiny budget → the guard interrupts BEFORE the
     call that would have hit the rate limit; LangGraph checkpoints the run
  2. we simulate the window reset (budget back up)
  3. we resume the SAME thread with Command(resume=True) → it completes
"""

from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from agentpause.adapters.langgraph import AgentPauseGuard


# -- a fake provider: replace with litellm/openai/anthropic in real use ------

class FakeProvider:
    def __init__(self, remaining):
        self.remaining = remaining          # tokens left in the rate window

    def telemetry(self):
        return self.remaining

    def call(self, messages):
        reply = f"reply to {len(messages)} messages"
        self.remaining -= 400               # each call costs ~400 tokens
        return reply, 400


provider = FakeProvider(remaining=500)      # tiny budget: won't fit 3 calls
guard = AgentPauseGuard(telemetry=provider.telemetry)


# -- the graph ----------------------------------------------------------------

class State(TypedDict):
    messages: list
    replies: list


def agent_node(state: State) -> State:
    guard.check(state["messages"])                    # ← the predictive gate
    reply, used = provider.call(state["messages"])
    guard.record(state["messages"], used)             # teach the estimator
    return {"messages": state["messages"] + [{"role": "assistant", "content": reply}],
            "replies": state["replies"] + [reply]}


def more_work(state: State) -> str:
    return "agent" if len(state["replies"]) < 3 else END


g = StateGraph(State)
g.add_node("agent", agent_node)
g.add_edge(START, "agent")
g.add_conditional_edges("agent", more_work)
app = g.compile(checkpointer=MemorySaver())

cfg = {"configurable": {"thread_id": "demo"}}
out = app.invoke({"messages": [{"role": "user", "content": "start " * 50}],
                  "replies": []}, cfg)

if "__interrupt__" in out:
    info = out["__interrupt__"][0].value
    print(f"[pause] predictive interrupt: remaining={info['remaining']}, "
          f"estimated={info['estimated']} (+{info['safety_k']}sigma)")
    print("[pause] the checkpointer holds the run; simulating window reset...")
    provider.remaining = 100_000
    out = app.invoke(Command(resume=True), cfg)       # same thread, no work lost

print(f"[done] {len(out['replies'])} steps completed: {out['replies']}")
