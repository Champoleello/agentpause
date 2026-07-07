"""Logical checkpoint store: the source of truth for suspend/resume.

This is the part that works on *any* provider, cloud or local, because it only
serializes application-level state — messages, step counter, and idempotency
keys — never model internals. (True KV-cache warm start lives in an optional
plugin and is not required here.)

Writes are atomic: the manifest is written to a temporary file and renamed, so a
crash mid-write can never corrupt a previous checkpoint.
"""

from __future__ import annotations

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
        """Register and return a fresh idempotency key for a side-effecting action."""
        key = f"{action}:{uuid.uuid4().hex[:8]}"
        self.idempotency_keys.append(key)
        return key

    def has_run(self, key: str) -> bool:
        """Whether an action key was already executed before a checkpoint."""
        return key in self.idempotency_keys


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
