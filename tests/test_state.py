"""Behavioral contract for the logical checkpoint store."""

from agentpause import Checkpoint, StateStore


def test_compact_shrinks_old_history_keeps_tail_and_system():
    cp = Checkpoint(session_id="t", messages=(
        [{"role": "system", "content": "S" * 500}] +
        [{"role": "assistant", "content": "X" * 1000} for _ in range(6)] +
        [{"role": "user", "content": "recent " * 50}]
    ))
    saved = cp.compact(keep_last=2, max_chars=100)
    assert saved > 0
    assert len(cp.messages[0]["content"]) == 500          # system kept intact
    assert all(len(m["content"]) <= 100 for m in cp.messages[1:-2])
    assert len(cp.messages[-1]["content"]) > 100          # tail kept intact


def test_compact_survives_a_save_load_roundtrip(tmp_path):
    """The §8.6 offline-compaction flow: load a suspended checkpoint,
    compact it, save it back — the resume sees the smaller history."""
    store = StateStore(str(tmp_path))
    cp = Checkpoint(session_id="t",
                    messages=[{"role": "assistant", "content": "Y" * 900}
                              for _ in range(8)])
    cp.compact(keep_last=2, max_chars=50)
    store.save(cp)
    again = store.load("t")
    assert all(len(m["content"]) <= 50 for m in again.messages[:-2])


def test_save_then_load_roundtrip(tmp_path):
    store = StateStore(directory=str(tmp_path))
    cp = Checkpoint(session_id="task-1", step=3,
                    messages=[{"role": "user", "content": "hi"}],
                    total_tokens_used=1234)
    store.save(cp)

    loaded = store.load("task-1")
    assert loaded is not None
    assert loaded.step == 3
    assert loaded.total_tokens_used == 1234
    assert loaded.messages == [{"role": "user", "content": "hi"}]


def test_load_missing_returns_none(tmp_path):
    store = StateStore(directory=str(tmp_path))
    assert store.load("does-not-exist") is None


def test_clear_removes_checkpoint(tmp_path):
    store = StateStore(directory=str(tmp_path))
    store.save(Checkpoint(session_id="task-2"))
    assert store.load("task-2") is not None
    store.clear("task-2")
    assert store.load("task-2") is None


def test_idempotency_keys_survive_roundtrip(tmp_path):
    store = StateStore(directory=str(tmp_path))
    cp = Checkpoint(session_id="task-3")
    key = cp.new_idempotency_key("send_email")
    store.save(cp)

    loaded = store.load("task-3")
    assert loaded.has_run(key) is True
    assert loaded.has_run("never_ran") is False


def test_save_is_atomic_no_tmp_left_behind(tmp_path):
    store = StateStore(directory=str(tmp_path))
    store.save(Checkpoint(session_id="task-4"))
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


def test_overwrite_keeps_latest(tmp_path):
    store = StateStore(directory=str(tmp_path))
    store.save(Checkpoint(session_id="s", step=1))
    store.save(Checkpoint(session_id="s", step=2))
    assert store.load("s").step == 2
