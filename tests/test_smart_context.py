"""Behavioral contract for the smart-context batch (Fase 10).

* F10.5 — useful waits: the LangGraph guard can hand wait time to the app
  (`while_waiting`) instead of sleeping it away: indexing, memory
  compression, prompt prep — work that needs no LLM.
* F10.2 — preventive compaction: don't wait for the §8.6 wall. When the
  context crosses a configured fraction of the window, old history shrinks
  BEFORE the decision — amortized, no emergencies.
* F10.1 — real summarization: `summarize_with(fn)` replaces old messages
  with one summary produced by an injected callable (typically a CHEAP
  model — possibly a different provider, working while the saturated one
  rests). Truncation stays as the model-free fallback.
"""

import pytest

from agentpause import Budget, Checkpoint, PredictiveScheduler, StateStore
from agentpause.adapters.langgraph import AgentPauseGuard


# -- F10.5: useful waits -------------------------------------------------------

def test_guard_hands_wait_time_to_the_app():
    budget = Budget(remaining_tokens=100, reset_seconds=4.0)
    worked = []

    def do_useful_work(seconds):
        worked.append(seconds)
        budget.remaining_tokens = 100_000     # meanwhile the window refills

    guard = AgentPauseGuard(telemetry=lambda: budget,
                            interrupt_fn=lambda p: None,
                            while_waiting=do_useful_work,
                            sleep_fn=lambda s: pytest.fail("must not idle-sleep"))
    guard.check([{"role": "user", "content": "q " * 50}])
    assert len(worked) == 1                   # the app used the wait


# -- F10.2: preventive compaction ------------------------------------------------

def big_history_session(sched):
    s = sched.session("t")
    s.add_system("task definition")
    for _ in range(8):
        s.add_message("assistant", "verbose old reasoning " * 100)  # ~2.2k chars each
    s.add_user("next question")
    return s


def test_context_pressure_triggers_compaction_before_the_decision(tmp_path):
    events = []
    sched = PredictiveScheduler(
        backend=lambda m: ("ok", 100),
        telemetry=lambda: 1_000_000,
        store=StateStore(str(tmp_path)),
        context_window=8000,                   # tokens
        compact_at=0.5,                        # shrink at 50% pressure
        on_event=lambda n, i: events.append((n, i)),
    )
    s = big_history_session(sched)
    before = sum(len(m["content"]) for m in s.messages)
    assert s.next_action().action == "continue"
    after = sum(len(m["content"]) for m in s.messages)
    assert after < before * 0.5                # history actually shrank
    assert any(n == "compacted" for n, _ in events)
    assert s.messages[0]["content"] == "task definition"       # system intact
    assert s.messages[-1]["content"] == "next question"        # tail intact


def test_no_context_window_means_no_compaction(tmp_path):
    sched = PredictiveScheduler(
        backend=lambda m: ("ok", 100),
        telemetry=lambda: 1_000_000,
        store=StateStore(str(tmp_path)),
    )
    s = big_history_session(sched)
    before = sum(len(m["content"]) for m in s.messages)
    s.next_action()
    assert sum(len(m["content"]) for m in s.messages) == before


# -- F10.1: real summarization ------------------------------------------------------

def test_summarize_with_replaces_old_history_with_one_summary():
    cp = Checkpoint(session_id="t", messages=(
        [{"role": "system", "content": "task"}] +
        [{"role": "assistant", "content": f"finding number {i}: " + "x" * 300}
         for i in range(6)] +
        [{"role": "user", "content": "latest question"}]
    ))

    def cheap_model_summary(text):
        assert "finding number 0" in text        # the old content reaches the fn
        return "six findings about x"

    saved = cp.summarize_with(cheap_model_summary, keep_last=2)
    assert saved > 0
    roles = [m["role"] for m in cp.messages]
    assert roles[0] == "system"                              # task definition kept
    assert "six findings about x" in cp.messages[1]["content"]
    assert cp.messages[-1]["content"] == "latest question"   # tail kept
    assert len(cp.messages) == 4                             # system+summary+tail(2)


def test_summarize_with_noop_when_history_is_short():
    cp = Checkpoint(session_id="t",
                    messages=[{"role": "user", "content": "hi"}])
    assert cp.summarize_with(lambda t: "s", keep_last=4) == 0
    assert cp.messages == [{"role": "user", "content": "hi"}]
