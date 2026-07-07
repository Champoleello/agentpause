"""Behavioral contract for the logical checkpoint store."""

from agentpause import Checkpoint, StateStore


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
