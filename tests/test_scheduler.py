"""Behavioral contract for the high-level PredictiveScheduler / Session API.

The scheduler is provider-agnostic: it takes two injected callables,
``backend`` (performs the LLM call) and ``telemetry`` (reads remaining tokens),
so it is fully testable without any real provider.
"""

import pytest
from agentpause import PredictiveScheduler, StateStore


# -- test doubles -----------------------------------------------------------

def make_backend(reply="ok", tokens=300):
    """A backend that returns a fixed reply and reports a fixed token cost."""
    def backend(messages):
        return reply, tokens
    return backend


class Telemetry:
    """A mutable telemetry stub whose value tests can change between reads."""
    def __init__(self, remaining):
        self.remaining = remaining
        self.reads = 0

    def __call__(self):
        self.reads += 1
        return self.remaining


# -- tests ------------------------------------------------------------------

def test_runs_to_completion_with_generous_budget(tmp_path):
    sched = PredictiveScheduler(
        backend=make_backend(tokens=300),
        telemetry=Telemetry(100_000),
        store=StateStore(str(tmp_path)),
    )
    with sched.session("t1") as s:
        for q in ("q1", "q2", "q3"):
            s.add_user(q)
            assert s.should_suspend() is False
            s.call()
    assert s.step == 3
    # 3 user + 3 assistant messages
    assert len(s.messages) == 6
    assert s.total_tokens_used == 900


def test_should_suspend_true_when_budget_low(tmp_path):
    sched = PredictiveScheduler(
        backend=make_backend(tokens=300),
        telemetry=Telemetry(100),  # tiny window
        store=StateStore(str(tmp_path)),
    )
    with sched.session("t2") as s:
        s.add_user("a long question " * 20)
        assert s.should_suspend() is True


def test_should_suspend_reads_fresh_telemetry_each_time(tmp_path):
    tele = Telemetry(100)
    sched = PredictiveScheduler(
        backend=make_backend(),
        telemetry=tele,
        store=StateStore(str(tmp_path)),
    )
    with sched.session("t3") as s:
        s.add_user("hello")
        assert s.should_suspend() is True      # low budget
        tele.remaining = 100_000               # window "resets"
        assert s.should_suspend() is False     # fresh read reflects it
    assert tele.reads >= 2                      # never cached


def test_resume_restores_step_and_messages(tmp_path):
    store = StateStore(str(tmp_path))
    # first session: run two steps then checkpoint
    s1 = PredictiveScheduler(make_backend(), Telemetry(100_000), store=store)
    with s1.session("job") as a:
        a.add_user("q1"); a.call()
        a.add_user("q2"); a.call()
        a.checkpoint()
    # second session, same id + store: must resume from step 2
    s2 = PredictiveScheduler(make_backend(), Telemetry(100_000), store=store)
    with s2.session("job") as b:
        assert b.resumed is True
        assert b.step == 2
        assert [m["content"] for m in b.messages if m["role"] == "user"] == ["q1", "q2"]


def test_call_records_consumption(tmp_path):
    sched = PredictiveScheduler(make_backend(tokens=250), Telemetry(100_000),
                                store=StateStore(str(tmp_path)))
    with sched.session("t4") as s:
        s.add_user("x")
        s.call()
        assert s.total_tokens_used == 250


def test_complete_clears_checkpoint(tmp_path):
    store = StateStore(str(tmp_path))
    sched = PredictiveScheduler(make_backend(), Telemetry(100_000), store=store)
    with sched.session("done") as s:
        s.add_user("x"); s.call()
        s.checkpoint()
        s.complete()
    # a fresh session with the same id starts clean
    with sched.session("done") as s2:
        assert s2.resumed is False
        assert s2.step == 0


def test_session_is_a_context_manager(tmp_path):
    sched = PredictiveScheduler(make_backend(), Telemetry(100_000),
                                store=StateStore(str(tmp_path)))
    sess = sched.session("cm")
    with sess as s:
        assert s is sess
