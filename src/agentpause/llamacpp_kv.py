"""TRUE warm start for llama.cpp: KV-cache save/restore, composed with FORK
and MIGRATION (F11.2).

``state.py`` promises: "True KV-cache warm start lives in an optional plugin
and is not required here." This module is that plugin.

Why a plugin and not part of ``state.py``: the logical checkpoint (messages,
step, idempotency keys) works on ANY provider, cloud or local. The KV-cache
is a machine- and model-local accelerator that only exists for self-hosted
llama.cpp runtimes exposing the ``/slots`` and ``/props`` HTTP endpoints — a
narrower, optional dependency that has no business being in the core module.

Design, in one paragraph: :class:`KVStateStore` WRAPS a :class:`~agentpause.state.StateStore`
(composition, not inheritance — the wrapped store's ``save``/``load``/``fork``/
``export_bundle``/``import_bundle`` are used as-is, never overridden) and adds
exactly two things a KV-cache needs that plain logical state doesn't:
a transactional save order (KV blob to disk FIRST, logical checkpoint commit
SECOND — so a KV failure can never corrupt a previously-valid checkpoint) and
a fingerprint-gated restore that degrades gracefully — never crashes — when
the blob is unusable (wrong model, missing file after a migration, or the
llama.cpp server being unreachable).

    from agentpause.llamacpp_kv import LlamaCppSlots, KVStateStore
    from agentpause import StateStore

    kv_store = KVStateStore(StateStore(".agentpause"), slots=LlamaCppSlots(),
                            base_url="http://127.0.0.1:8080", id_slot=0,
                            kv_dir="kv_cache")
    kv_store.save_with_kv(checkpoint)
    cp, info = kv_store.load_with_kv("mission")
    if info["kv_restored"]:
        ...  # no re-prefill needed
    else:
        ...  # logical warm start only; info["reason"] says why
"""

from __future__ import annotations

import os
import shutil
import uuid
from typing import Any, Callable, Dict, Optional, Tuple

from .errors import CheckpointError, KVError
from .state import Checkpoint, StateStore

__all__ = ["LlamaCppSlots", "KVStateStore"]


# -- transport -----------------------------------------------------------------

def _default_get(url: str) -> Dict[str, Any]:
    """Default GET transport; imports httpx lazily so importing this module
    (or agentpause itself) never requires it — only calling the real
    transport does."""
    import httpx
    r = httpx.get(url, timeout=30.0)
    r.raise_for_status()
    return r.json()


def _default_post(url: str, json: Dict[str, Any]) -> Dict[str, Any]:
    """Default POST transport, same lazy-import discipline as ``_default_get``
    (mirrors ``adapters/openai_compat.py``'s ``_default_post``). KV saves of a
    large context can take a while, hence the generous timeout."""
    import httpx
    r = httpx.post(url, json=json, timeout=1800.0)
    r.raise_for_status()
    return r.json()


class LlamaCppSlots:
    """Thin, injectable client for the 3 llama.cpp server endpoints this
    plugin needs.

    * ``GET /props`` -> ``model_path`` is used as the model fingerprint: a
      KV blob is NOT portable between models, so every restore is gated on
      this matching what was saved.
    * ``POST /slots/{id}?action=save`` with ``{"filename": ...}`` -> response
      carries ``n_saved`` (KV cells written).
    * ``POST /slots/{id}?action=restore`` with ``{"filename": ...}`` ->
      response carries ``n_restored`` (KV cells read back).
    * ``get_props``/``get_slot`` expose the raw ``GET /props`` /
      ``GET /slots`` bodies for callers that need more than the model
      fingerprint -- e.g. :class:`agentpause.adapters.local_resources.LlamaCppContextBudget`,
      which reads the configured context size and a slot's current token
      usage to build a REAL local context :class:`~agentpause.risk.Budget`
      instead of a made-up ``fallback_remaining``.

    ``get_fn``/``post_fn`` are injectable for tests (``get_fn(url) -> dict``,
    ``post_fn(url, json) -> dict``); the defaults import ``httpx`` lazily
    inside the function body, so this module has no hard dependency on it.
    Any transport failure raises :class:`~agentpause.errors.KVError` — the
    caller (:class:`KVStateStore`) decides whether to degrade or propagate.
    """

    def __init__(
        self,
        get_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
        post_fn: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        self._get = get_fn if get_fn is not None else _default_get
        self._post = post_fn if post_fn is not None else _default_post

    def get_props(self, base_url: str) -> Dict[str, Any]:
        """Raw ``GET /props`` response body, as the server sent it.

        Factored out of :meth:`fingerprint` so callers needing more than the
        model path (e.g. the configured context size, under
        ``default_generation_settings.n_ctx`` -- see
        :mod:`agentpause.adapters.local_resources`) don't have to issue a
        second HTTP round-trip for the same endpoint. ``fingerprint``'s own
        observable behavior is unchanged: it still returns exactly the same
        string it always did, just via this method internally.
        """
        try:
            return self._get(f"{base_url.rstrip('/')}/props")
        except Exception as exc:
            raise KVError(
                f"Cannot read /props from llama.cpp server at {base_url}: {exc}"
            ) from exc

    def fingerprint(self, base_url: str) -> str:
        """The model currently loaded by the server, used to gate restores."""
        body = self.get_props(base_url)
        model = body.get("model_path") or (
            body.get("default_generation_settings", {}) or {}
        ).get("model") or ""
        return str(model)

    def get_slot(self, base_url: str, id_slot: int) -> Dict[str, Any]:
        """Raw ``GET /slots`` entry for ``id_slot`` (dict, server's own shape).

        ``GET /slots`` returns a JSON array, one object per slot; this finds
        the entry whose ``"id"`` matches ``id_slot``. Raises
        :class:`~agentpause.errors.KVError` -- same style as
        :meth:`fingerprint`/:meth:`save`/:meth:`restore` -- when the server is
        unreachable, returns something that isn't a list (some deployments
        nest it under a ``"slots"`` key; that shape is tolerated), or simply
        has no slot with that id.
        """
        url = f"{base_url.rstrip('/')}/slots"
        try:
            body = self._get(url)
        except Exception as exc:
            raise KVError(
                f"Cannot read /slots from llama.cpp server at {base_url}: {exc}"
            ) from exc
        if isinstance(body, dict):
            body = body.get("slots", body)  # tolerate a {"slots": [...]} wrapper
        if not isinstance(body, list):
            raise KVError(
                f"Unexpected /slots response shape from {base_url}: "
                f"expected a list, got {type(body).__name__}"
            )
        for slot in body:
            if isinstance(slot, dict) and slot.get("id") == id_slot:
                return slot
        raise KVError(
            f"No slot with id_slot={id_slot} in /slots response from {base_url} "
            f"({len(body)} slot(s) reported)"
        )

    def save(self, base_url: str, id_slot: int, filename: str) -> int:
        """Save slot ``id_slot``'s KV-cache to ``filename``; returns ``n_saved``."""
        url = f"{base_url.rstrip('/')}/slots/{id_slot}?action=save"
        try:
            body = self._post(url, {"filename": filename})
        except Exception as exc:
            raise KVError(
                f"KV save failed for slot {id_slot} at {base_url}: {exc}"
            ) from exc
        return int(body.get("n_saved", 0))

    def restore(self, base_url: str, id_slot: int, filename: str) -> int:
        """Restore slot ``id_slot``'s KV-cache from ``filename``; returns ``n_restored``."""
        url = f"{base_url.rstrip('/')}/slots/{id_slot}?action=restore"
        try:
            body = self._post(url, {"filename": filename})
        except Exception as exc:
            raise KVError(
                f"KV restore failed for slot {id_slot} at {base_url}: {exc}"
            ) from exc
        return int(body.get("n_restored", 0))


class KVStateStore:
    """Wraps a :class:`~agentpause.state.StateStore` and adds TRUE (KV-cache)
    warm start on top of the logical one, composed with FORK and MIGRATION.

    Args:
        store: the wrapped logical store. Its ``save``/``load``/``fork``/
            ``export_bundle``/``import_bundle`` are used unmodified.
        slots: the llama.cpp HTTP client (:class:`LlamaCppSlots`, or a fake
            for tests — anything with the same 3-method interface).
        base_url: the llama-server root, e.g. ``http://127.0.0.1:8080``.
        id_slot: which server slot to save/restore (default 0).
        kv_dir: directory KV blob files are written to/read from. MUST be the
            same directory the llama-server was started with via
            ``--slot-save-path`` (e.g. ``llama-server --slot-save-path
            ./kv_cache ...`` <-> ``KVStateStore(..., kv_dir="kv_cache")``).
            The server resolves the ``filename`` it receives against its own
            ``--slot-save-path``, so this plugin always sends it a BARE
            filename (never ``kv_dir``-prefixed) and only prepends ``kv_dir``
            for its OWN local disk bookkeeping (existence checks, copies for
            :meth:`fork_with_kv`, GC). If the two directories diverge, either
            the server 400s outright (can't resolve the path), or — the more
            insidious case — the server saves successfully into its OWN real
            ``--slot-save-path`` (a valid ``n_saved`` comes back) while our
            ``kv_dir`` is some other directory: :meth:`save_with_kv` now
            catches exactly this case immediately, checking that the blob
            landed under ``kv_dir`` right after the save call and before
            anything is committed, instead of letting it surface later, far
            from the cause, as a ``reason="kv_file_missing"`` at the next
            :meth:`load_with_kv`.
        min_free_bytes: optional disk-space guard for :meth:`save_with_kv`.
            When ``None`` (the default), NO check is performed at all --
            behavior is bit-for-bit identical to before this parameter
            existed. When set, :meth:`save_with_kv` reads
            ``disk_usage_fn(kv_dir).free`` BEFORE calling ``slots.save()``
            (the HTTP POST that tells the llama-server to actually write the
            blob) and raises :class:`~agentpause.errors.KVError` up front,
            before touching ``cp`` or the network, if free space is below
            this threshold.

            Why checking ``kv_dir`` is a meaningful proxy even though the
            llama-server -- a separate C++ process -- is the one physically
            writing the blob's bytes, not this Python process: this class's
            own contract (see ``kv_dir`` above) already REQUIRES ``kv_dir``
            to be the exact same physical directory as the server's
            ``--slot-save-path``. Every other piece of local bookkeeping this
            class does (existence checks in :meth:`load_with_kv`, the copies
            in :meth:`fork_with_kv`, both GC methods) already leans on that
            same assumption being true. Given that assumption holds, free
            space on ``kv_dir`` as seen from Python IS free space on the
            filesystem the server will write to -- same disk, same
            filesystem, same free-space number -- so it's a legitimate
            (if racy: a large enough concurrent write between the check and
            the real POST could still exhaust it) early signal that the
            server-side save is likely to fail for lack of room. Worth
            checking cheaply up front because a real KV save's POST can be
            configured with a timeout of up to 1800 seconds (see
            ``_default_post``) -- a nearly-full disk is a common, foreseeable
            way to pay that entire cost only to fail anyway.
        prune_oldest: if ``True`` and free space is found below
            ``min_free_bytes``, :meth:`save_with_kv` tries to reclaim space
            itself before giving up: first :meth:`gc_orphans` (blobs no
            checkpoint references at all -- always safe to delete), then
            :meth:`gc_consumed` (blobs already confirmed unneeded by a
            completed resume), then rechecks free space exactly once (never
            a retry loop). Only raises :class:`~agentpause.errors.KVError` if
            space is still short after that single recheck. Default
            ``False``: no pruning is attempted, matching today's behavior.
        disk_usage_fn: injectable in place of ``shutil.disk_usage`` (its
            default), so the disk-space guard is testable offline without
            touching the real filesystem. Called as ``disk_usage_fn(kv_dir)``
            and expected to return an object with a ``.free`` attribute in
            bytes, exactly like ``shutil.disk_usage``'s return value. Ignored
            entirely when ``min_free_bytes`` is ``None``.

    The KV blob is "semi-temporary" memory: at most one live blob per
    session (a new save garbage-collects the previous one), a restored blob
    is marked *consumed* and is only actually deleted once the caller
    confirms — via :meth:`gc_consumed` — that at least one post-resume step
    has succeeded without needing it again. Blobs left behind by crashes or
    superseded checkpoints are swept by :meth:`gc_orphans`.
    """

    def __init__(
        self,
        store: StateStore,
        slots: LlamaCppSlots,
        base_url: str,
        id_slot: int = 0,
        kv_dir: str = "kv_cache",
        min_free_bytes: Optional[int] = None,
        prune_oldest: bool = False,
        disk_usage_fn: Callable[[str], Any] = shutil.disk_usage,
    ) -> None:
        self.store = store
        self.slots = slots
        self.base_url = base_url
        self.id_slot = id_slot
        self.kv_dir = kv_dir
        self.min_free_bytes = min_free_bytes
        self.prune_oldest = prune_oldest
        self.disk_usage_fn = disk_usage_fn
        try:
            os.makedirs(kv_dir, exist_ok=True)
        except OSError as exc:
            raise CheckpointError(f"Cannot create KV directory '{kv_dir}': {exc}") from exc
        # session_id -> blob filename, tracked between a successful restore
        # and the caller's gc_consumed() call (see load_with_kv/gc_consumed).
        self._consumed: Dict[str, str] = {}

    def _kv_path(self, filename: str) -> str:
        return os.path.join(self.kv_dir, filename)

    def _delete_blob(self, filename: str) -> None:
        if not filename:
            return
        path = self._kv_path(filename)
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass  # best-effort GC: never let cleanup crash the caller

    # -- save --------------------------------------------------------------

    def save_with_kv(self, cp: Checkpoint) -> Checkpoint:
        """Transactional save: KV blob to disk FIRST, logical commit SECOND.

        This ordering is the whole guarantee. ``slots.save`` (and the
        fingerprint read right after it) happen BEFORE ``cp`` is mutated or
        handed to the wrapped store's (already-atomic) ``.save()``. If the KV
        save fails, the exception propagates immediately: nothing about
        ``cp`` has changed yet, and whatever checkpoint was previously on
        disk for this session_id is untouched — a KV failure can never
        corrupt a prior, valid logical checkpoint.

        When ``min_free_bytes`` is set (see the class docstring), an even
        earlier check runs before any of that: free space on ``kv_dir`` is
        read and, if short, ``prune_oldest`` is given one chance to reclaim
        some before we give up. This guard is the FIRST thing in this method
        that can fail, strictly before ``slots.save``'s HTTP POST — so it
        preserves the exact same transactional guarantee (nothing about
        ``cp`` or the wrapped store has changed if it raises).

        Fail-fast on a silent kv_dir/--slot-save-path mismatch: right after
        ``slots.save`` reports success (before reading the fingerprint, before
        touching ``cp`` at all), this method checks that the blob actually
        landed at ``self._kv_path(filename)``. A real llama-server can report
        a perfectly valid ``n_saved`` while writing the file into a directory
        we never look in, if ``kv_dir`` doesn't match the server's own
        ``--slot-save-path`` — without this check that mismatch would only
        surface much later, at the next :meth:`load_with_kv`, as
        ``reason="kv_file_missing"``, far from the real cause. Raising here
        instead keeps the same transactional guarantee as every other failure
        in this method: nothing about ``cp`` or a previously-valid checkpoint
        has changed.

        Only once the blob is safely on disk do we stash
        ``cp.extra['kv'] = {"file", "model_fingerprint", "n_saved", "consumed"}``
        and call the wrapped store's ``.save(cp)``. On success, the PREVIOUS
        blob this session_id pointed at (if any, and if different from the
        new one) is garbage-collected — at most one live blob per session.
        """
        previous = self.store.load(cp.session_id)
        prev_kv = (previous.extra or {}).get("kv") if previous is not None else None

        filename = f"{cp.session_id}_{uuid.uuid4().hex[:8]}.bin"

        if self.min_free_bytes is not None:
            free = self.disk_usage_fn(self.kv_dir).free
            pruned = False
            if free < self.min_free_bytes and self.prune_oldest:
                pruned = True
                self.gc_orphans()
                self.gc_consumed()
                free = self.disk_usage_fn(self.kv_dir).free
            if free < self.min_free_bytes:
                prune_note = (
                    "pruning was attempted (gc_orphans() + gc_consumed()) but did not "
                    "free enough space"
                    if pruned
                    else "pruning was not attempted (prune_oldest=False)"
                )
                raise KVError(
                    f"Refusing to save KV blob for session '{cp.session_id}' in "
                    f"kv_dir='{self.kv_dir}': {free} byte(s) free, "
                    f"{self.min_free_bytes} byte(s) required ({prune_note})."
                )

        # NOTE: the server resolves `filename` against its OWN configured
        # save directory (llama-server's --slot-save-path), so it must be a
        # BARE filename, never kv_dir-prefixed -- kv_dir is understood to be
        # the SAME directory as --slot-save-path on disk for a local server;
        # see the class docstring.
        n_saved = self.slots.save(self.base_url, self.id_slot, filename)  # may raise: nothing committed yet

        # Fail fast, right here, if the file the server just claimed to write
        # never actually appears where WE expect it. Without this check the
        # method below would happily read the fingerprint, stash extra['kv'],
        # and commit the logical checkpoint -- n_saved looked fine, nothing
        # here raised -- and the missing blob would only surface much later,
        # at the NEXT load_with_kv, as reason='kv_file_missing', far away in
        # time from the real cause. The near-universal real cause: kv_dir (our
        # local bookkeeping directory) does not actually match the
        # --slot-save-path the server was started with, so the server writes
        # the blob into a directory we never look in. Checked BEFORE the
        # fingerprint read and BEFORE any mutation of cp/commit, so the exact
        # same transactional guarantee as every other failure in this method
        # holds: if this raises, nothing about a previously-valid checkpoint
        # changes.
        if not os.path.exists(self._kv_path(filename)):
            raise KVError(
                f"KV save for session '{cp.session_id}' reported success "
                f"(n_saved={n_saved}) but no file appeared at "
                f"{self._kv_path(filename)!r}. This almost certainly means "
                f"kv_dir={self.kv_dir!r} does not match the real "
                f"--slot-save-path the llama-server was started with -- the "
                f"server resolved 'filename' against ITS OWN save directory "
                f"and wrote the blob there instead. Check that kv_dir here "
                f"is the exact same directory as --slot-save-path on the "
                f"server's command line."
            )

        fingerprint = self.slots.fingerprint(self.base_url)  # may raise: still nothing committed

        cp.extra["kv"] = {
            "file": filename,
            "model_fingerprint": fingerprint,
            "n_saved": n_saved,
            "consumed": False,
        }
        self.store.save(cp)  # atomic logical commit -- only after the blob is safe

        if prev_kv and prev_kv.get("file") and prev_kv["file"] != filename:
            self._delete_blob(prev_kv["file"])
        return cp

    # -- load ----------------------------------------------------------------

    def load_with_kv(self, session_id: str) -> Tuple[Optional[Checkpoint], Dict[str, Any]]:
        """Load via the wrapped store; restore the KV blob if it is usable.

        Degrades gracefully (never raises) in every case where the blob
        cannot be trusted:

        * no checkpoint, or a checkpoint with no ``extra['kv']`` at all
          (never had a KV-warm-started save) -> ``kv_restored=False``.
        * ``extra['kv']['file']`` not found under ``kv_dir`` -- exactly what
          happens after :meth:`~agentpause.state.StateStore.import_bundle`
          moves logical state to a new machine without its KV blob (blobs
          are machine-local, they never migrate) -> ``reason="kv_file_missing"``.
        * the llama.cpp server is unreachable, or its currently loaded model
          fingerprint does not match what was saved (a KV blob is not
          portable between models) -> ``reason="unreachable"`` /
          ``"model_mismatch"``. On a model mismatch the stale blob file is
          also deleted (it can never be used again).

        In every degrade path the stale ``extra['kv']`` reference is cleared
        off the returned checkpoint (in memory only -- not re-persisted here;
        the next :meth:`save_with_kv` naturally overwrites it, and until then
        a repeat load simply re-detects the same, still-graceful outcome).

        On success, the blob is marked "consumed": it stays on disk until the
        caller invokes :meth:`gc_consumed` once resumed and past the first
        post-resume step.
        """
        cp = self.store.load(session_id)
        if cp is None:
            return None, {"kv_restored": False, "reason": "no_checkpoint"}

        kv = (cp.extra or {}).get("kv")
        if not kv:
            return cp, {"kv_restored": False, "reason": "no_kv"}

        path = self._kv_path(kv["file"])
        if not os.path.exists(path):
            del cp.extra["kv"]
            return cp, {"kv_restored": False, "reason": "kv_file_missing"}

        try:
            current_fingerprint = self.slots.fingerprint(self.base_url)
        except KVError:
            del cp.extra["kv"]
            return cp, {"kv_restored": False, "reason": "unreachable"}

        if current_fingerprint != kv.get("model_fingerprint"):
            self._delete_blob(kv["file"])
            del cp.extra["kv"]
            return cp, {"kv_restored": False, "reason": "model_mismatch"}

        try:
            # bare filename to the server, same reasoning as in save_with_kv;
            # `path` (kv_dir-prefixed) is only for OUR local disk checks above.
            n_restored = self.slots.restore(self.base_url, self.id_slot, kv["file"])
        except KVError:
            del cp.extra["kv"]
            return cp, {"kv_restored": False, "reason": "unreachable"}

        cp.extra["kv"]["consumed"] = True
        self._consumed[session_id] = kv["file"]
        return cp, {"kv_restored": True, "n_restored": n_restored}

    # -- fork (F11.2) ----------------------------------------------------------

    def fork_with_kv(self, session_id: str, new_session_id: str) -> Checkpoint:
        """FORK, extended to the KV blob: give each branch its OWN copy.

        Calls the wrapped store's ``.fork()`` first to get the independent
        logical clone (unchanged F11.2 behavior — messages/extra deep-copied,
        its own session_id namespace). If the parent's checkpoint carried
        ``extra['kv']``, that reference would otherwise point both parent and
        child at the SAME blob file on disk, and whichever branch consumes or
        garbage-collects it first would break the other's warm start. So: the
        blob file is COPIED to a new filename, and the clone's
        ``extra['kv']['file']`` is rewritten to point at the copy.

        Tradeoff, by design: N branches forked from one suspended past each
        get their own true (KV) warm start, at the cost of N copies of the
        blob on disk. Acceptable for local self-hosted use; no attempt is
        made at copy-on-write sharing here.

        If the parent's blob is already gone (consumed and GC'd, or never
        existed), the clone simply gets no ``extra['kv']`` -- a plain logical
        fork, no different from before F11.2's KV extension.
        """
        clone = self.store.fork(session_id, new_session_id)
        kv = (clone.extra or {}).get("kv")
        if not kv or not kv.get("file"):
            return clone

        src = self._kv_path(kv["file"])
        if not os.path.exists(src):
            del clone.extra["kv"]
            self.store.save(clone)
            return clone

        new_filename = f"{new_session_id}_{uuid.uuid4().hex[:8]}.bin"
        dst = self._kv_path(new_filename)
        shutil.copyfile(src, dst)
        clone.extra["kv"] = dict(kv)
        clone.extra["kv"]["file"] = new_filename
        clone.extra["kv"]["consumed"] = False
        self.store.save(clone)
        return clone

    # -- GC --------------------------------------------------------------------

    def gc_consumed(self, session_id: Optional[str] = None) -> int:
        """Delete blob(s) marked consumed by a prior :meth:`load_with_kv`.

        Call once the session has resumed and completed its first
        post-resume step successfully -- the point at which the restored KV
        state has proven itself and the on-disk blob is no longer needed.
        With ``session_id=None`` (default) GCs every blob currently tracked
        as consumed; pass a specific id to GC just that one. Returns the
        number of blobs removed.
        """
        targets = [session_id] if session_id is not None else list(self._consumed.keys())
        removed = 0
        for sid in targets:
            filename = self._consumed.pop(sid, None)
            if filename:
                self._delete_blob(filename)
                removed += 1
        return removed

    def gc_orphans(self) -> int:
        """Delete blob files under ``kv_dir`` not referenced by ANY
        checkpoint currently in the wrapped store's directory.

        Scans the store's directory the same way :class:`StateStore` names
        its files (``<session_id>.json``), collects every
        ``extra['kv']['file']`` still referenced, and removes anything else
        found in ``kv_dir``. Returns the number of files removed.
        """
        referenced = set()
        directory = self.store.directory
        if os.path.isdir(directory):
            for name in os.listdir(directory):
                if not name.endswith(".json"):
                    continue
                session_id = name[: -len(".json")]
                try:
                    cp = self.store.load(session_id)
                except CheckpointError:
                    continue
                if cp is not None:
                    kv = (cp.extra or {}).get("kv")
                    if kv and kv.get("file"):
                        referenced.add(kv["file"])

        removed = 0
        if os.path.isdir(self.kv_dir):
            for filename in os.listdir(self.kv_dir):
                if filename not in referenced:
                    self._delete_blob(filename)
                    removed += 1
        return removed
