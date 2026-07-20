"""Behavioral contract for TRUE (KV-cache) warm start, composed with FORK
and MIGRATION (F11.2).

Uses a dict-backed ``FakeSlots`` test double (same interface as
:class:`~agentpause.llamacpp_kv.LlamaCppSlots`: ``fingerprint``/``save``/
``restore``) so every test is deterministic and needs no real llama-server.
"""

from __future__ import annotations

import os

import pytest

from agentpause import Checkpoint, StateStore
from agentpause.errors import KVError
from agentpause.llamacpp_kv import KVStateStore


class FakeSlots:
    """In-memory double for LlamaCppSlots: no network, fully deterministic.

    Mimics the REAL llama-server's semantics precisely (this is what a live
    test against a real server caught missing before): the server resolves
    the ``filename`` it receives against its OWN configured save directory
    (``--slot-save-path``), so a well-behaved caller must send a BARE
    filename, never one prefixed with the local ``kv_dir``. ``save_dir``
    below stands in for that server-side ``--slot-save-path`` -- construct
    this fake with the SAME directory as the ``KVStateStore``'s ``kv_dir``,
    exactly as a real deployment must, and any accidental path-prefixing
    regresses to a clear failure (dir-inside-dir) instead of silently passing.

    ``model`` is the fingerprint ``/props`` would report; change it between
    calls to simulate a model swap. ``fail`` forces the next call to raise
    KVError, to simulate the server being unreachable.
    """

    def __init__(self, save_dir: str, model: str = "models/llama-3.1-8b.gguf") -> None:
        self.save_dir = save_dir
        self.model = model
        self.fail = False
        self.save_calls = 0
        self.restore_calls = 0

    def _resolve(self, filename: str) -> str:
        # a bare filename resolves fine; anything path-like (e.g. accidentally
        # prefixed with kv_dir again) produces a nonexistent nested directory,
        # surfacing the exact bug a real server would 400 on.
        return os.path.join(self.save_dir, filename)

    def fingerprint(self, base_url: str) -> str:
        if self.fail:
            raise KVError("simulated: server unreachable")
        return self.model

    def save(self, base_url: str, id_slot: int, filename: str) -> int:
        if self.fail:
            raise KVError("simulated: save failed")
        assert os.sep not in filename and (os.altsep is None or os.altsep not in filename), (
            f"save() must receive a BARE filename (server resolves it against "
            f"its own --slot-save-path), got a path-like value: {filename!r}"
        )
        self.save_calls += 1
        os.makedirs(self.save_dir, exist_ok=True)
        with open(self._resolve(filename), "wb") as f:
            f.write(b"kv-cells")
        return 42

    def restore(self, base_url: str, id_slot: int, filename: str) -> int:
        if self.fail:
            raise KVError("simulated: restore failed")
        assert os.sep not in filename and (os.altsep is None or os.altsep not in filename), (
            f"restore() must receive a BARE filename, got a path-like value: {filename!r}"
        )
        self.restore_calls += 1
        return 42


def make_kv_store(tmp_path, slots=None, subdir="store"):
    store = StateStore(str(tmp_path / subdir))
    kv_dir = str(tmp_path / f"{subdir}-kv")
    slots = slots if slots is not None else FakeSlots(save_dir=kv_dir)
    kv_store = KVStateStore(store, slots=slots, base_url="http://fake:8080",
                            id_slot=0, kv_dir=kv_dir)
    return kv_store, store, slots


# -- 1. save_with_kv writes a blob + stashes extra['kv'], stays plain-loadable --

def test_save_with_kv_writes_blob_and_stashes_extra(tmp_path):
    kv_store, store, slots = make_kv_store(tmp_path)
    cp = Checkpoint(session_id="mission", step=1,
                    messages=[{"role": "user", "content": "hi"}])

    kv_store.save_with_kv(cp)

    kv = cp.extra["kv"]
    assert kv["n_saved"] == 42
    assert kv["model_fingerprint"] == slots.model
    blob_path = os.path.join(kv_store.kv_dir, kv["file"])
    assert os.path.exists(blob_path)

    # a plain StateStore (no KV awareness at all) can still load it: extra
    # is just data.
    plain = store.load("mission")
    assert plain is not None
    assert plain.extra["kv"]["file"] == kv["file"]
    assert plain.step == 1
    assert plain.messages == [{"role": "user", "content": "hi"}]


# -- 2. load_with_kv on matching fingerprint: true restore --------------------

def test_load_with_kv_matching_fingerprint_restores(tmp_path):
    kv_store, store, slots = make_kv_store(tmp_path)
    cp = Checkpoint(session_id="mission", step=3,
                    messages=[{"role": "user", "content": "a"},
                              {"role": "assistant", "content": "b"}])
    kv_store.save_with_kv(cp)

    fresh_kv_store, _, _ = make_kv_store(tmp_path)
    # reuse same underlying dirs but a fresh KVStateStore instance
    fresh_kv_store = KVStateStore(StateStore(str(tmp_path / "store")), slots=slots,
                                  base_url="http://fake:8080", id_slot=0,
                                  kv_dir=str(tmp_path / "store-kv"))

    loaded, info = fresh_kv_store.load_with_kv("mission")

    assert info["kv_restored"] is True
    assert info["n_restored"] == 42
    assert loaded is not None
    assert loaded.step == 3
    assert loaded.messages == cp.messages
    assert slots.restore_calls == 1


# -- 3. fingerprint mismatch: graceful logical warm start ---------------------

def test_load_with_kv_model_mismatch_degrades_gracefully(tmp_path):
    kv_store, store, slots = make_kv_store(tmp_path)
    cp = Checkpoint(session_id="mission", step=5,
                    messages=[{"role": "user", "content": "hello"}])
    kv_store.save_with_kv(cp)

    slots.model = "models/a-totally-different-model.gguf"  # simulate model swap

    loaded, info = kv_store.load_with_kv("mission")

    assert info["kv_restored"] is False
    assert info["reason"] == "model_mismatch"
    assert loaded is not None
    assert loaded.step == 5                      # logical state intact
    assert loaded.messages == [{"role": "user", "content": "hello"}]
    assert "kv" not in loaded.extra               # stale reference cleared

    # the stale blob file was discarded
    old_kv_dir_files = os.listdir(kv_store.kv_dir)
    assert old_kv_dir_files == []


# -- 4. missing blob file (post-migration): graceful logical warm start ------

def test_load_with_kv_missing_blob_file_degrades_gracefully(tmp_path):
    kv_store, store, slots = make_kv_store(tmp_path)
    cp = Checkpoint(session_id="mission", step=2,
                    messages=[{"role": "user", "content": "migrated"}])
    kv_store.save_with_kv(cp)

    # simulate migration to a new machine: blob directory never traveled
    blob_path = os.path.join(kv_store.kv_dir, cp.extra["kv"]["file"])
    os.remove(blob_path)

    loaded, info = kv_store.load_with_kv("mission")

    assert info["kv_restored"] is False
    assert info["reason"] == "kv_file_missing"
    assert loaded is not None
    assert loaded.step == 2
    assert loaded.messages == [{"role": "user", "content": "migrated"}]
    assert "kv" not in loaded.extra


# -- 5. transactional ordering: save failure never corrupts prior state ------

def test_save_with_kv_failure_leaves_previous_checkpoint_untouched(tmp_path):
    kv_store, store, slots = make_kv_store(tmp_path)
    good = Checkpoint(session_id="mission", step=1,
                      messages=[{"role": "user", "content": "good state"}])
    kv_store.save_with_kv(good)

    slots.fail = True  # next slots.save() raises
    bad = Checkpoint(session_id="mission", step=2,
                     messages=[{"role": "user", "content": "never committed"}])

    with pytest.raises(KVError):
        kv_store.save_with_kv(bad)

    # the wrapped store's .load() directly proves the previous checkpoint
    # survives untouched -- the KV failure never reached the logical commit.
    reloaded = store.load("mission")
    assert reloaded is not None
    assert reloaded.step == 1
    assert reloaded.messages == [{"role": "user", "content": "good state"}]


# -- 6. fork_with_kv: independent blob per branch -----------------------------

def test_fork_with_kv_gives_each_branch_its_own_blob(tmp_path):
    kv_store, store, slots = make_kv_store(tmp_path)
    parent = Checkpoint(session_id="mission", step=4,
                        messages=[{"role": "user", "content": "shared past"}])
    kv_store.save_with_kv(parent)
    parent_file = parent.extra["kv"]["file"]

    cautious = kv_store.fork_with_kv("mission", "mission-cautious")
    bold = kv_store.fork_with_kv("mission", "mission-bold")

    cautious_file = cautious.extra["kv"]["file"]
    bold_file = bold.extra["kv"]["file"]

    assert len({parent_file, cautious_file, bold_file}) == 3  # all distinct

    for fname in (parent_file, cautious_file, bold_file):
        assert os.path.exists(os.path.join(kv_store.kv_dir, fname))

    # deleting one branch's blob never affects a sibling's
    os.remove(os.path.join(kv_store.kv_dir, cautious_file))
    assert os.path.exists(os.path.join(kv_store.kv_dir, parent_file))
    assert os.path.exists(os.path.join(kv_store.kv_dir, bold_file))

    # both children restore correctly and independently
    loaded_bold, info_bold = kv_store.load_with_kv("mission-bold")
    assert info_bold["kv_restored"] is True
    assert loaded_bold.step == 4

    loaded_cautious, info_cautious = kv_store.load_with_kv("mission-cautious")
    assert info_cautious["kv_restored"] is False
    assert info_cautious["reason"] == "kv_file_missing"
    assert loaded_cautious.step == 4  # logical state still intact


# -- 7. GC: consumed blobs and orphans ----------------------------------------

def test_gc_consumed_deletes_blob_after_marking(tmp_path):
    kv_store, store, slots = make_kv_store(tmp_path)
    cp = Checkpoint(session_id="mission", step=1, messages=[])
    kv_store.save_with_kv(cp)
    blob_path = os.path.join(kv_store.kv_dir, cp.extra["kv"]["file"])
    assert os.path.exists(blob_path)

    _, info = kv_store.load_with_kv("mission")
    assert info["kv_restored"] is True
    assert os.path.exists(blob_path)  # not yet GC'd -- caller hasn't confirmed

    removed = kv_store.gc_consumed("mission")
    assert removed == 1
    assert not os.path.exists(blob_path)

    # a second call is a no-op (nothing left tracked for this session)
    assert kv_store.gc_consumed("mission") == 0


def test_gc_orphans_sweeps_unreferenced_files(tmp_path):
    kv_store, store, slots = make_kv_store(tmp_path)
    cp = Checkpoint(session_id="mission", step=1, messages=[])
    kv_store.save_with_kv(cp)
    live_file = cp.extra["kv"]["file"]

    # drop an orphan file with no checkpoint referencing it
    orphan_path = os.path.join(kv_store.kv_dir, "orphan_nobody_references.bin")
    with open(orphan_path, "wb") as f:
        f.write(b"stale")

    removed = kv_store.gc_orphans()

    assert removed == 1
    assert not os.path.exists(orphan_path)
    assert os.path.exists(os.path.join(kv_store.kv_dir, live_file))
