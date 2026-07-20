"""Behavioral contract for FORK and MIGRATION (F11.2).

A suspended checkpoint is inert data — a frozen process image. These tests
pin down the two OS-like operations that follow from that framing:

* FORK: one suspended past, N independent continuations (deep-copied state,
  collision-free idempotency, estimator calibration inherited — F9.3 symmetry).
* MIGRATION: the checkpoint is a portable, json-serializable bundle that can
  be installed into a *different* StateStore (a different machine/directory)
  and resumed from the exact step, no work redone.
"""

from __future__ import annotations

import json

import pytest

from agentpause import Checkpoint, PredictiveScheduler, StateStore
from agentpause.errors import CheckpointError


# -- 1. Checkpoint.fork(): independent deep copy -----------------------------

def test_fork_is_independent_deep_copy():
    parent = Checkpoint(
        session_id="parent",
        step=3,
        messages=[{"role": "user", "content": "hi"}],
        total_tokens_used=42,
        extra={"estimator": {"n": 3, "mean": 100.0}},
    )
    clone = parent.fork("child")

    assert clone.session_id == "child"
    assert clone.step == parent.step
    assert clone.total_tokens_used == parent.total_tokens_used
    assert clone.messages == parent.messages
    assert clone.extra == parent.extra

    # mutate the clone: parent must stay untouched
    clone.messages.append({"role": "assistant", "content": "clone only"})
    clone.messages[0]["content"] = "mutated"
    clone.extra["estimator"]["mean"] = 999.0
    clone.extra["new_key"] = "clone only"

    assert len(parent.messages) == 1
    assert parent.messages[0]["content"] == "hi"
    assert parent.extra["estimator"]["mean"] == 100.0
    assert "new_key" not in parent.extra


# -- 2. StateStore.fork(): persistence + guardrails --------------------------

def test_store_fork_saves_clone(tmp_path):
    store = StateStore(str(tmp_path))
    store.save(Checkpoint(session_id="parent", step=2,
                          messages=[{"role": "user", "content": "x"}]))

    clone = store.fork("parent", "child")
    assert clone.session_id == "child"
    assert clone.step == 2

    reloaded = store.load("child")
    assert reloaded is not None
    assert reloaded.step == 2
    assert reloaded.messages == [{"role": "user", "content": "x"}]


def test_store_fork_refuses_existing_target(tmp_path):
    store = StateStore(str(tmp_path))
    store.save(Checkpoint(session_id="parent"))
    store.save(Checkpoint(session_id="child"))

    with pytest.raises(CheckpointError):
        store.fork("parent", "child")


def test_store_fork_missing_parent_raises(tmp_path):
    store = StateStore(str(tmp_path))
    with pytest.raises(CheckpointError):
        store.fork("nobody-here", "child")


# -- 3. Idempotency independence ---------------------------------------------

def test_forked_session_does_not_dedupe_against_parent():
    parent = Checkpoint(session_id="parent")
    key_before_fork = parent.new_idempotency_key("step-0")  # shared past

    child = parent.fork("child")

    # the key minted before the fork is shared history: both branches
    # correctly recognize it as already-done work
    assert parent.has_run(key_before_fork)
    assert child.has_run(key_before_fork)

    # now each branch mints its OWN key for the "same" logical action
    # (re-executing the step index that follows the fork point)
    parent_key = parent.new_idempotency_key("step-1")
    child_key = child.new_idempotency_key("step-1")

    assert parent_key != child_key           # never collide
    assert not child.has_run(parent_key)      # child doesn't see parent's key
    assert not parent.has_run(child_key)      # parent doesn't see child's key
    # namespacing comes from the session_id embedded in the key
    assert parent_key.startswith("parent:")
    assert child_key.startswith("child:")


# -- 4. export/import roundtrip across two stores ----------------------------

def test_export_import_roundtrip_across_stores(tmp_path):
    store_a = StateStore(str(tmp_path / "a"))
    cp = Checkpoint(session_id="mission", step=3,
                    messages=[{"role": "user", "content": "hi"},
                              {"role": "assistant", "content": "hey"}],
                    total_tokens_used=555,
                    extra={"estimator": {"n": 3}})
    store_a.save(cp)

    bundle = store_a.export_bundle("mission")
    assert bundle["format"] == StateStore.BUNDLE_FORMAT

    # must survive a real json round trip (the wire)
    wire = json.dumps(bundle)
    bundle_over_wire = json.loads(wire)

    store_b = StateStore(str(tmp_path / "b"))
    imported = store_b.import_bundle(bundle_over_wire)

    assert imported.session_id == "mission"
    assert imported.step == 3
    assert imported.messages == cp.messages
    assert imported.total_tokens_used == 555
    assert imported.extra == cp.extra

    reloaded = store_b.load("mission")
    assert reloaded is not None
    assert reloaded.step == 3
    assert reloaded.messages == cp.messages
    assert reloaded.total_tokens_used == 555
    assert reloaded.extra == cp.extra


def test_import_bundle_refuses_overwrite_without_flag(tmp_path):
    store_a = StateStore(str(tmp_path / "a"))
    store_a.save(Checkpoint(session_id="mission", step=1))
    bundle = store_a.export_bundle("mission")

    store_b = StateStore(str(tmp_path / "b"))
    store_b.save(Checkpoint(session_id="mission", step=99))  # local progress

    with pytest.raises(CheckpointError):
        store_b.import_bundle(bundle)

    # local progress untouched
    assert store_b.load("mission").step == 99

    # explicit overwrite works
    imported = store_b.import_bundle(bundle, overwrite=True)
    assert imported.step == 1
    assert store_b.load("mission").step == 1


def test_import_bundle_rejects_wrong_format(tmp_path):
    store = StateStore(str(tmp_path))
    with pytest.raises(CheckpointError):
        store.import_bundle({"format": "not-a-real-format", "state": {}})


# -- 5. End-to-end: scheduler migration across stores ------------------------

def fake_backend(messages):
    n = sum(1 for m in messages if m["role"] == "assistant") + 1
    return f"reply-{n}", 100


def test_scheduler_migration_end_to_end(tmp_path):
    store_a = StateStore(str(tmp_path / "a"))
    sched_a = PredictiveScheduler(backend=fake_backend,
                                  telemetry=lambda: 1_000_000,
                                  store=store_a)

    with sched_a.session("mission") as s:
        s.add_user("first")
        s.call()
        s.add_user("second")
        s.call()
        s.checkpoint()
        assert s.step == 2

    bundle = store_a.export_bundle("mission")
    wire = json.loads(json.dumps(bundle))  # prove it's json-portable

    store_b = StateStore(str(tmp_path / "b"))
    store_b.import_bundle(wire)

    sched_b = PredictiveScheduler(backend=fake_backend,
                                  telemetry=lambda: 1_000_000,
                                  store=store_b)
    with sched_b.session("mission") as s:
        assert s.resumed is True
        assert s.step == 2
        s.add_user("third")
        s.call()
        s.complete()
        assert s.step == 3

    assert store_b.load("mission") is None  # completed: checkpoint cleared
