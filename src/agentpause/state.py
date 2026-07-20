"""Logical checkpoint store: the source of truth for suspend/resume.

This is the part that works on *any* provider, cloud or local, because it only
serializes application-level state — messages, step counter, and idempotency
keys — never model internals. (True KV-cache warm start lives in an optional
plugin and is not required here.)

Writes are atomic: the manifest is written to a temporary file and renamed, so a
crash mid-write can never corrupt a previous checkpoint.

Because a suspended checkpoint is inert data — a frozen process image — two
OS-like operations come naturally (F11.2): FORK (one suspended past, N
independent continuations; see :meth:`Checkpoint.fork`) and MIGRATION (the
checkpoint directory is the process image, so moving the state moves the
agent; see :meth:`StateStore.export_bundle` / :meth:`StateStore.import_bundle`).
"""

from __future__ import annotations

import copy
import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from .errors import CheckpointError


@dataclass
class Checkpoint:
    """A resumable snapshot of an agent session."""

    session_id: str
    step: int = 0
    messages: List[Dict[str, str]] = field(default_factory=list)
    idempotency_keys: List[str] = field(default_factory=list)
    total_tokens_used: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def new_idempotency_key(self, action: str) -> str:
        """Register and return a fresh idempotency key for a side-effecting action.

        Keys are namespaced by ``session_id``: two sessions can never mint the
        same key. This is what keeps forked runs honest — a clone (which gets
        its own id, see :meth:`fork`) builds its own dedup namespace, so its
        new side effects are never confused with a sibling's, while the keys
        inherited from the shared past keep protecting every branch against
        re-running work done before the fork.
        """
        key = f"{self.session_id}:{action}:{uuid.uuid4().hex[:8]}"
        self.idempotency_keys.append(key)
        return key

    def has_run(self, key: str) -> bool:
        """Whether an action key was already executed before a checkpoint."""
        return key in self.idempotency_keys

    def fork(self, new_session_id: str) -> "Checkpoint":
        """FORK (F11.2): one suspended past, N independent futures.

        A suspended checkpoint is inert data — a frozen process image — so
        cloning it is as natural as an OS ``fork()``: take the same expensive
        history and continue it several ways (explore different strategies
        from the same prefix without re-paying for it).

        The clone is fully INDEPENDENT of the parent:

        * ``messages`` and ``extra`` are deep-copied — mutating one branch
          never touches another;
        * it carries its own ``session_id``, and since idempotency keys embed
          the session id (see :meth:`new_idempotency_key`), keys minted after
          the fork can never collide with a sibling's — a forked run is never
          deduplicated against another branch's side effects. Keys inherited
          from the shared past ride along on purpose, so side effects
          performed *before* the fork are not re-executed by any branch;
        * the estimator statistics in ``extra['estimator']`` ride along ON
          PURPOSE (the F9.3 symmetry): a fork starts calibrated with what its
          parent already learned, instead of re-learning from scratch.
        """
        return Checkpoint(
            session_id=new_session_id,
            step=self.step,
            messages=copy.deepcopy(self.messages),
            idempotency_keys=list(self.idempotency_keys),
            total_tokens_used=self.total_tokens_used,
            extra=copy.deepcopy(self.extra),
        )

    def compact(self, keep_last: int = 4, max_chars: int = 200) -> int:
        """Overflow policy (§8.6/§5.2 of the research): shrink old history.

        When the next call cannot fit even a FULL rate window, no amount of
        waiting or suspending helps — the context itself must shrink. This
        truncates the content of older messages (keeping a leading system
        message and the last ``keep_last`` messages intact) and returns the
        number of characters saved.

        Works on a live session (``session.compact()``) or OFFLINE on a
        suspended checkpoint — compression is useful work that needs no LLM,
        perfect for the time an agent spends suspended.
        """
        start = 1 if self.messages and self.messages[0].get("role") == "system" else 0
        saved = 0
        for m in self.messages[start:len(self.messages) - keep_last]:
            content = m.get("content") or ""
            if len(content) > max_chars:
                saved += len(content) - max_chars
                m["content"] = content[:max_chars - 1] + "…"
        return saved

    def summarize_with(self, summarizer, keep_last: int = 4) -> int:
        """Semantic compression (§5.2): replace old history with ONE summary.

        ``summarizer`` is any ``text -> summary`` callable — typically a cheap
        model, ideally on a DIFFERENT provider so the work happens while the
        saturated one rests. Quality-wise this beats :meth:`compact`'s blind
        truncation; use compact as the model-free fallback.

        Keeps a leading system message and the last ``keep_last`` messages
        intact. Returns the number of characters saved (0 if history is too
        short to bother).
        """
        start = 1 if self.messages and self.messages[0].get("role") == "system" else 0
        old = self.messages[start:len(self.messages) - keep_last]
        if not old:
            return 0
        text = "\n".join(f"{m.get('role')}: {m.get('content') or ''}" for m in old)
        summary = summarizer(text)
        head = self.messages[:start]
        tail = self.messages[len(self.messages) - keep_last:]
        note = {"role": "system",
                "content": f"[Summary of {len(old)} earlier messages] {summary}"}
        self.messages[:] = head + [note] + tail
        return max(0, len(text) - len(note["content"]))


class StateStore:
    """Persists and restores :class:`Checkpoint` objects as JSON files."""

    def __init__(self, directory: str = ".agentpause") -> None:
        self.directory = directory
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as exc:
            raise CheckpointError(
                f"Cannot create checkpoint directory '{directory}': {exc}"
            ) from exc

    def _path(self, session_id: str) -> str:
        safe = session_id.replace(os.sep, "_")
        return os.path.join(self.directory, f"{safe}.json")

    def save(self, cp: Checkpoint) -> str:
        """Atomically persist a checkpoint; returns the file path."""
        path = self._path(cp.session_id)
        payload = {
            "checkpoint_id": uuid.uuid4().hex,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "state": asdict(cp),
        }
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            os.replace(tmp, path)  # atomic on POSIX and Windows
        except OSError as exc:
            raise CheckpointError(
                f"Cannot write checkpoint '{path}': {exc}"
            ) from exc
        return path

    def load(self, session_id: str) -> Optional[Checkpoint]:
        """Restore a checkpoint, or return None if none exists."""
        path = self._path(session_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return Checkpoint(**data["state"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise CheckpointError(
                f"Cannot read checkpoint '{path}' (corrupted?): {exc}"
            ) from exc

    def clear(self, session_id: str) -> None:
        """Delete a session's checkpoint (e.g., after successful completion)."""
        path = self._path(session_id)
        if os.path.exists(path):
            os.remove(path)

    # -- fork & migration (F11.2) -------------------------------------------

    #: Version tag for :meth:`export_bundle` payloads. Bump on breaking
    #: changes to the bundle layout so old importers fail loudly, not subtly.
    BUNDLE_FORMAT = "agentpause-checkpoint/1"

    def fork(self, session_id: str, new_session_id: str) -> Checkpoint:
        """Clone a stored checkpoint under a new id and persist the clone.

        The store-level counterpart of :meth:`Checkpoint.fork`: load the
        parent, clone it deep and independent, save the clone atomically,
        return it ready to resume. A missing parent is an error (there is no
        frozen past to continue from), and an existing checkpoint under
        ``new_session_id`` is never silently overwritten — losing a sibling's
        progress to a name clash would defeat the whole point of forking.
        """
        parent = self.load(session_id)
        if parent is None:
            raise CheckpointError(
                f"Cannot fork '{session_id}': no checkpoint found in "
                f"'{self.directory}'."
            )
        if os.path.exists(self._path(new_session_id)):
            raise CheckpointError(
                f"Cannot fork into '{new_session_id}': a checkpoint already "
                f"exists under that id in '{self.directory}' — refusing to "
                "overwrite."
            )
        clone = parent.fork(new_session_id)
        self.save(clone)
        return clone

    def export_bundle(self, session_id: str) -> Dict[str, Any]:
        """MIGRATION, step one (F11.2): the checkpoint as a portable dict.

        The checkpoint directory IS the process image: moving the state moves
        the agent. This returns a plain json-serializable dict — versioned via
        its ``format`` field — that survives any transport (a file, a message
        queue, an HTTP body) and installs into a DIFFERENT StateStore on a
        different machine via :meth:`import_bundle`. Machine A suspends and
        exports; machine B imports and resumes at the exact step.

        What deliberately does NOT migrate: machine-local KV-cache blobs
        (present only in self-hosted setups, via plugins). They are
        accelerators tied to the machine that produced them, not logical
        state — after a migration the resume degrades gracefully to a logical
        warm start: full history, calibrated estimator, cold caches.
        """
        cp = self.load(session_id)
        if cp is None:
            raise CheckpointError(
                f"Cannot export '{session_id}': no checkpoint found in "
                f"'{self.directory}'."
            )
        return {
            "format": self.BUNDLE_FORMAT,
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "state": asdict(cp),
        }

    def import_bundle(self, bundle: Dict[str, Any],
                      overwrite: bool = False) -> Checkpoint:
        """MIGRATION, step two: install an exported bundle into THIS store.

        Validates the ``format`` version and the state layout before touching
        disk, then saves atomically. An existing session under the same id is
        never clobbered unless ``overwrite=True`` — the caller must say out
        loud that the incoming image should replace local progress.
        """
        fmt = bundle.get("format") if isinstance(bundle, dict) else None
        if fmt != self.BUNDLE_FORMAT:
            raise CheckpointError(
                f"Not an agentpause checkpoint bundle: format={fmt!r} "
                f"(expected {self.BUNDLE_FORMAT!r})."
            )
        try:
            cp = Checkpoint(**bundle["state"])
        except (KeyError, TypeError) as exc:
            raise CheckpointError(
                f"Malformed checkpoint bundle: {exc}"
            ) from exc
        if not overwrite and os.path.exists(self._path(cp.session_id)):
            raise CheckpointError(
                f"Session '{cp.session_id}' already exists in "
                f"'{self.directory}' — pass overwrite=True to replace it."
            )
        self.save(cp)
        return cp
