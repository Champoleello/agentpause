"""Behavioral contract for the LangGraph adapter.

The adapter is a *guard* you drop into any LangGraph node that calls an LLM:

    guard = AgentPauseGuard(telemetry=...)

    def agent_node(state):
        guard.check(state["messages"])         # may interrupt the graph here
        reply = llm.invoke(state["messages"])
        guard.record(state["messages"], used)  # teach the estimator
        ...

``check()`` applies the predictive rule; when the next call would not fit the
remaining budget it calls LangGraph's ``interrupt()``, which pauses the graph
through the configured checkpointer. On resume the node re-executes, telemetry
is read FRESH (never trusted from the checkpoint), and the run proceeds only
if the budget now fits.

These tests run offline: ``interrupt_fn`` is injected, so the base contract
needs no ``langgraph`` install. A final integration test uses the real
library when available (skipped otherwise).
"""

import pytest

from agentpause import Estimator
from agentpause.adapters.langgraph import AgentPauseGuard


# -- test doubles -----------------------------------------------------------

class Telemetry:
    """Mutable telemetry stub; counts reads to prove freshness."""

    def __init__(self, remaining):
        self.remaining = remaining
        self.reads = 0

    def __call__(self):
        self.reads += 1
        return self.remaining


class FakeInterrupt(Exception):
    """Stands in for langgraph's internal GraphInterrupt."""

    def __init__(self, payload):
        self.payload = payload


def raising_interrupt(payload):
    """First-pass behavior of langgraph's interrupt(): abort the node."""
    raise FakeInterrupt(payload)


MESSAGES = [{"role": "user", "content": "a question " * 30}]


# -- check(): the predictive gate ---------------------------------------------

def test_check_passes_with_generous_budget():
    guard = AgentPauseGuard(telemetry=Telemetry(100_000),
                            interrupt_fn=raising_interrupt)
    guard.check(MESSAGES)  # must not raise


def test_check_interrupts_when_budget_low():
    guard = AgentPauseGuard(telemetry=Telemetry(100),
                            interrupt_fn=raising_interrupt)
    with pytest.raises(FakeInterrupt) as exc:
        guard.check(MESSAGES)
    payload = exc.value.payload
    assert payload["reason"] == "predicted_rate_limit"
    assert payload["remaining"] == 100
    assert payload["estimated"] > 0
    assert payload["sigma"] >= 0
    assert payload["safety_k"] == guard.safety_k


def test_check_rereads_telemetry_after_resume():
    """On resume interrupt() *returns*; check() must then re-read the budget."""
    tele = Telemetry(100)
    resumed = []

    def resuming_interrupt(payload):
        # simulate the resume pass: interrupt() returns instead of raising,
        # and meanwhile the provider window has reset
        resumed.append(payload)
        tele.remaining = 100_000
        return True

    guard = AgentPauseGuard(telemetry=tele, interrupt_fn=resuming_interrupt)
    guard.check(MESSAGES)          # low → interrupt → "resume" → fresh read → pass
    assert len(resumed) == 1
    assert tele.reads >= 2         # one stale read, one fresh read


def test_check_interrupts_again_if_budget_still_low():
    """If after a resume the budget is STILL low, the guard must not proceed."""
    tele = Telemetry(100)
    calls = []

    def interrupt_twice(payload):
        calls.append(payload)
        if len(calls) == 1:
            return True            # first resume: budget unchanged
        raise FakeInterrupt(payload)  # second interrupt: abort again

    guard = AgentPauseGuard(telemetry=tele, interrupt_fn=interrupt_twice)
    with pytest.raises(FakeInterrupt):
        guard.check(MESSAGES)
    assert len(calls) == 2


# -- record(): learning -------------------------------------------------------

def test_record_feeds_the_estimator():
    est = Estimator()
    guard = AgentPauseGuard(telemetry=Telemetry(100_000),
                            estimator=est, interrupt_fn=raising_interrupt)
    assert est.samples == 0
    guard.record(MESSAGES, used=450)
    assert est.samples == 1


def test_estimate_adapts_after_records():
    est = Estimator()
    guard = AgentPauseGuard(telemetry=Telemetry(100_000),
                            estimator=est, interrupt_fn=raising_interrupt)
    before = est.estimate(1000)
    for _ in range(6):
        guard.record(MESSAGES, used=5000)   # consistently above the base estimate
    assert est.estimate(1000) > before      # epsilon pushed the estimate up


# -- integration with the real library (optional) ------------------------------

def test_real_graph_interrupts_and_resumes(tmp_path):
    pytest.importorskip("langgraph", reason="langgraph not installed")
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import StateGraph, END, START
    from langgraph.types import Command
    from typing_extensions import TypedDict

    tele = Telemetry(100)   # too low: first run must pause
    guard = AgentPauseGuard(telemetry=tele)   # real langgraph interrupt()

    class S(TypedDict):
        messages: list
        reply: str

    def agent_node(state: S) -> S:
        guard.check(state["messages"])
        return {"messages": state["messages"], "reply": "done"}

    g = StateGraph(S)
    g.add_node("agent", agent_node)
    g.add_edge(START, "agent")
    g.add_edge("agent", END)
    app = g.compile(checkpointer=MemorySaver())

    cfg = {"configurable": {"thread_id": "t1"}}
    out = app.invoke({"messages": MESSAGES, "reply": ""}, cfg)
    assert "__interrupt__" in out                     # paused, not crashed
    payload = out["__interrupt__"][0].value
    assert payload["reason"] == "predicted_rate_limit"

    tele.remaining = 100_000                          # window reset
    out = app.invoke(Command(resume=True), cfg)       # resume the same thread
    assert out["reply"] == "done"
