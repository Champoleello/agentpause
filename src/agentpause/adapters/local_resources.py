"""Real local-resource budget for self-hosted llama.cpp servers.

On a cloud provider, :class:`~agentpause.risk.Budget` is built from
rate-limit response headers (``x-ratelimit-remaining-tokens`` and friends --
see ``adapters/openai_compat.py`` / ``adapters/litellm.py``). A self-hosted
llama.cpp server (``llama-server``; see :mod:`agentpause.llamacpp_kv`) sends
none of those headers, because there is no rate limit to respect: nobody is
throttling requests-per-minute against a process running on your own
machine. What DOES exist, and does run out, is context: the fixed-size KV
cache configured via ``--ctx-size`` (per slot). Passing a made-up
``fallback_remaining=N`` to an adapter papers over this with a number that
never reflects reality and never depletes -- exactly the gap this module
closes. :class:`LlamaCppContextBudget` reads the REAL configured context
size and the REAL number of tokens currently resident in a slot's KV cache
straight from the server (``GET /props`` and ``GET /slots``, via
:class:`agentpause.llamacpp_kv.LlamaCppSlots`) and turns that into a
:class:`~agentpause.risk.Budget` the rest of the library already
understands -- wire it in as a ``telemetry`` callable exactly like any
adapter's ``budget()``::

    from agentpause.llamacpp_kv import LlamaCppSlots
    from agentpause.adapters.local_resources import LlamaCppContextBudget

    slots = LlamaCppSlots()
    telemetry = LlamaCppContextBudget(slots, base_url="http://127.0.0.1:8080",
                                       id_slot=0, safety_margin_tokens=256)
    sched = PredictiveScheduler(backend=..., telemetry=telemetry)

Sources checked (2026-07-21) and what remains to verify against a real server
-------------------------------------------------------------------------------
``tools/server/README.md`` on ``ggml-org/llama.cpp`` (master branch, fetched
directly from ``raw.githubusercontent.com``) documents:

* ``GET /props`` -> a top-level ``default_generation_settings`` object that
  the README's own example payload CONFIRMS carries ``n_ctx`` (the context
  size), plus a top-level ``total_slots``. There is no top-level ``n_ctx``
  in the documented example, only the nested one.
* ``GET /slots`` -> a JSON ARRAY, one object per slot, each CONFIRMED (by the
  same README's example payload) to carry ``id`` (matches ``id_slot``),
  ``id_task``, ``n_ctx``, ``is_processing``, ``params``, and ``next_token``
  (with ``n_decoded``: tokens DECODED so far for the slot's current
  generation task).

What is NOT confirmed by that README: a field giving the TOTAL number of
tokens currently resident in a slot's KV cache (prompt + generation so far)
-- the exact thing this module needs as "used". The documented ``/slots``
example does not show ``cache_tokens``, ``n_past``, ``prompt``, or ``tokens``
in its response body, even though other llama.cpp-ecosystem discussion (e.g.
GitHub discussion ggml-org/llama.cpp#13606 on KV cache reuse) references
``n_past`` as the server's internal counter for exactly this quantity --
server LOGS show it, but the currently documented ``/slots`` JSON body does
not. Given that ambiguity, :func:`_default_used_field` tries, IN ORDER:
``cache_tokens`` (list length, or int value if already a count),
``n_past``, ``prompt`` (length of), ``tokens`` (list length), and only as a
last resort ``next_token.n_decoded`` (an admittedly IMPERFECT proxy: it only
counts this task's OUTPUT tokens, not the full resident prompt+generation
context). ANYONE ADOPTING THIS CLASS SHOULD fire a real ``GET /slots`` at
their own server, inspect the raw JSON, and pass an explicit ``used_field=``
if the guessed default doesn't match their build -- exactly why it is an
injectable callable and not a hard-coded key, the same defensive pattern
``remaining_header=`` already uses in ``adapters/litellm.py``.

No "reset" for local context
-----------------------------
Unlike TPM/RPM windows, which refill on a clock (see
:class:`agentpause.risk.Budget`'s ``reset_seconds``), a llama.cpp context
window never refills on its own -- there is nothing to wait out. The only
ways to get tokens back are compressing history
(:meth:`agentpause.state.Checkpoint.compact` /
:meth:`agentpause.state.Checkpoint.summarize_with`) or starting a fresh
session. :class:`LlamaCppContextBudget` never sets ``reset_seconds`` (it
stays at its default, ``None``), so :func:`agentpause.risk.decide` can never
be misled into recommending ``wait`` for a local context shortfall -- with no
reset info it falls through to ``checkpoint``, which is the only sane
outcome here.

A second, INDEPENDENT local signal: free VRAM
----------------------------------------------
:class:`GPUMemoryBudget` reads a second, genuinely different real signal:
how much GPU memory is free RIGHT NOW, straight from the hardware via NVML
(NVIDIA's Management Library), not estimated or configured. This is
DELIBERATELY NVIDIA-only today, via the official `pynvml` bindings
(PyPI package ``nvidia-ml-py`` -- verified 2026-07-21 to be the package
NVIDIA Corporation itself publishes and actively releases; the separate
``pynvml`` PyPI name has historically been an unofficial/abandoned fork and
is NOT what this module depends on, even though the *import* is still
``import pynvml``, since that's the module name NVIDIA's own package
installs). AMD/ROCm GPUs are NOT supported here: ROCm exposes its own
management interface (``rocm-smi`` / ``amdsmi``), not NVML, and would need a
genuinely different reader -- no attempt is made to fake or silently
downgrade that support.

Context vs. VRAM -- an important difference in KIND, not just source:
:class:`LlamaCppContextBudget` measures something scoped to THIS
conversation (the slot's own KV cache). Free VRAM is scoped to the WHOLE
GPU: every other process resident on the same card -- another model, another
agent, a desktop compositor -- eats into the same number. It is a more
"external"/shared signal than the context budget, and :class:`GPUMemoryBudget`
documents that explicitly rather than presenting it as if it belonged to
this session alone.

VRAM bytes are not tokens: a KV-cache token's footprint in bytes depends on
the model (hidden size, layer count, quantization, GQA/MQA head count) --
there is no universal constant. :class:`GPUMemoryBudget` therefore requires
an explicit ``bytes_per_token`` to report a token budget at all; see its
docstring for how to calibrate that number from a real KV save
(:meth:`agentpause.llamacpp_kv.LlamaCppSlots.save`'s ``n_saved`` plus
``os.path.getsize`` on the blob it writes).

A third, much lighter piece: turning local throughput into a price
---------------------------------------------------------------------
:class:`LlamaCppContextBudget` and :class:`GPUMemoryBudget` are full
``telemetry`` callables -- each one replaces an entire axis of what
:class:`~agentpause.scheduler.PredictiveScheduler` reads on every decision.
:func:`estimate_local_price_per_1k_tokens`,
:func:`estimate_hourly_cost_from_power`, and
:func:`price_per_1k_tokens_from_estimator` are NOT a fourth callable in that
family -- they are plain, pure functions that introduce no new mechanism
at all. ``PredictiveScheduler`` already accepts ``price_per_1k_tokens`` and
``money_budget``, and ``Session.next_action()`` already turns those into a
hard "checkpoint, never wait" decision the instant projected cost exceeds
what is left (money never refills by waiting, unlike a token window) --
this already works identically in local and cloud use. The only real gap
locally is that nobody hands you an invoice to read a per-1k-token price
off of. These functions close that gap by deriving a price from an assumed
hourly cost (cloud GPU rental, or your own electricity bill) and a measured
throughput, with the throughput itself pulled straight from an
already-calibrated :class:`~agentpause.estimator.Estimator` /
:class:`~agentpause.regression.FeatureEstimator` instead of a separate
benchmark::

    from agentpause.adapters.local_resources import (
        estimate_hourly_cost_from_power,
        price_per_1k_tokens_from_estimator,
    )

    hourly_cost = estimate_hourly_cost_from_power(watts=350, price_per_kwh=0.28)
    price = price_per_1k_tokens_from_estimator(
        my_estimator, input_tokens=2000, hourly_cost=hourly_cost,
    )
    if price is not None:
        sched = PredictiveScheduler(..., price_per_1k_tokens=price, money_budget=5.0)
    # else: not enough history yet to know real throughput (a plain
    # Estimator has no estimate_latency at all, or a FeatureEstimator hasn't
    # recorded enough steps for it to return anything but None) -- perfectly
    # normal early in a run; just skip setting price_per_1k_tokens this step
    # and try again once more steps have been recorded.

This is deliberately the LIGHTEST of the local pieces in this module.
Unlike :class:`LlamaCppContextBudget`, :class:`GPUMemoryBudget`, and
:class:`KVAwareTimeBudget` -- none of which the scheduler knew how to do on
its own -- ``price_per_1k_tokens``/``money_budget`` already existed and
already worked before this addition. All three functions below do is make
it easy to populate those two pre-existing parameters with an honest,
derived number instead of either a hand-picked guess or nothing at all.

A fourth piece: composing several local signals into ONE telemetry callable
------------------------------------------------------------------------------
``PredictiveScheduler`` accepts exactly one ``telemetry=`` callable, but a
real self-hosted deployment can easily have MORE than one hard local
ceiling at once -- a llama.cpp slot's own context window
(:class:`LlamaCppContextBudget`) AND the GPU's free VRAM
(:class:`GPUMemoryBudget`) are two genuinely independent resources that can
each run out first, unpredictably, depending on the workload and on what
else is resident on the same card. :class:`CompositeLocalBudget` calls every
wrapped callable on each read and reports whichever ``remaining_tokens`` is
LOWEST -- the same failure mode a real deployment has: the agent gets
stopped by whichever resource runs out FIRST, not by an average of the two.
See its own docstring for the exact merge rule for the other ``Budget``
fields, and for why a :class:`~agentpause.errors.TelemetryError` from any one
wrapped source is left to propagate rather than being swallowed.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional, Tuple

from ..errors import GPUError, KVError, TelemetryError
from ..llamacpp_kv import LlamaCppSlots
from ..risk import Budget

__all__ = [
    "LlamaCppContextBudget",
    "GPUMemoryBudget",
    "KVAwareTimeBudget",
    "CompositeLocalBudget",
    "estimate_local_price_per_1k_tokens",
    "estimate_hourly_cost_from_power",
    "price_per_1k_tokens_from_estimator",
]


def _default_context_field(props: Dict[str, Any]) -> int:
    """Best-effort extraction of the configured context size from ``/props``.

    Tries ``default_generation_settings.n_ctx`` first (CONFIRMED present in
    the ggml-org/llama.cpp server README's own example payload, checked
    2026-07-21), then a top-level ``n_ctx`` as a fallback (some proxies/forks
    have been observed flattening it there). Raises ``KeyError`` if neither
    is found -- callers should verify their server's actual ``/props``
    response and pass ``context_field=...`` if it differs.
    """
    dgs = props.get("default_generation_settings")
    if isinstance(dgs, dict) and "n_ctx" in dgs:
        return int(dgs["n_ctx"])
    if "n_ctx" in props:
        return int(props["n_ctx"])
    raise KeyError(
        "neither 'default_generation_settings.n_ctx' nor a top-level "
        "'n_ctx' found in the /props response; pass context_field=..."
    )


def _default_used_field(slot: Dict[str, Any]) -> int:
    """Best-effort extraction of tokens currently resident in a slot's KV cache.

    NOT confirmed against the current official README (see the module
    docstring): tries, in descending order of plausibility, ``cache_tokens``,
    ``n_past``, ``prompt``, ``tokens``, and finally ``next_token.n_decoded``
    (an imperfect proxy -- output tokens only). Raises ``KeyError`` if none of
    these keys is present -- callers should inspect a real ``GET /slots``
    response and pass ``used_field=...`` if the guess is wrong for their
    server/build.
    """
    if "cache_tokens" in slot:
        v = slot["cache_tokens"]
        return len(v) if isinstance(v, (list, tuple)) else int(v)
    if "n_past" in slot:
        return int(slot["n_past"])
    if "prompt" in slot:
        v = slot["prompt"]
        return len(v) if isinstance(v, (list, tuple, str)) else int(v)
    if "tokens" in slot:
        v = slot["tokens"]
        return len(v) if isinstance(v, (list, tuple)) else int(v)
    next_token = slot.get("next_token")
    if isinstance(next_token, dict) and "n_decoded" in next_token:
        return int(next_token["n_decoded"])
    raise KeyError(
        "none of cache_tokens/n_past/prompt/tokens/next_token.n_decoded "
        "found in the /slots entry; pass used_field=..."
    )


class LlamaCppContextBudget:
    """Turns a llama.cpp server's REAL context usage into a :class:`~agentpause.risk.Budget`.

    Reads ``GET /props`` (configured context size) and ``GET /slots`` (tokens
    currently used by ``id_slot``) via an injected :class:`~agentpause.llamacpp_kv.LlamaCppSlots`
    and computes ``remaining = max(0, context_size - used - safety_margin_tokens)``.
    Call it wherever a ``telemetry`` callable is expected
    (``PredictiveScheduler(telemetry=LlamaCppContextBudget(...))``): it takes
    no per-call arguments, matching the ``Callable[[], int | Budget]`` shape
    the scheduler expects.

    Args:
        slots: the llama.cpp HTTP client (:class:`~agentpause.llamacpp_kv.LlamaCppSlots`,
            or any fake with the same ``get_props``/``get_slot`` interface --
            tests inject one that returns fixed dicts, no network involved).
        base_url: the llama-server root, e.g. ``http://127.0.0.1:8080``.
        id_slot: which server slot to read (default 0).
        safety_margin_tokens: tokens subtracted from the raw remaining count
            before it is reported, as a cushion against the next step's own
            prompt overhead or estimation error (default 0, no cushion).
        context_field: ``dict -> int`` extractor over the raw ``/props``
            body; defaults to :func:`_default_context_field`. Override this
            if your server's field names differ -- see the module docstring
            for exactly what was and wasn't verified.
        used_field: ``dict -> int`` extractor over the raw ``/slots`` entry
            for ``id_slot``; defaults to :func:`_default_used_field`. Same
            override mechanism, same caveat.

    IMPORTANT -- no reset, ever: unlike a cloud TPM/RPM window, a local
    context window does not refill on any clock. This class never populates
    ``Budget.reset_seconds`` (it stays ``None``), so the three-valued
    ``decide()`` rule can never conclude ``wait`` for a context shortfall
    reported here -- with no reset information it degrades straight to
    ``checkpoint``, which is the only correct behavior: the only way to free
    local context is to compress the conversation
    (``Checkpoint.compact()`` / ``Checkpoint.summarize_with()`` in
    :mod:`agentpause.state`) or start a new session, never to wait.
    """

    def __init__(
        self,
        slots: LlamaCppSlots,
        base_url: str,
        id_slot: int = 0,
        safety_margin_tokens: int = 0,
        context_field: Optional[Callable[[Dict[str, Any]], int]] = None,
        used_field: Optional[Callable[[Dict[str, Any]], int]] = None,
    ) -> None:
        self.slots = slots
        self.base_url = base_url
        self.id_slot = id_slot
        self.safety_margin_tokens = safety_margin_tokens
        self._context_field = (
            context_field if context_field is not None else _default_context_field
        )
        self._used_field = used_field if used_field is not None else _default_used_field

    def __call__(self) -> Budget:
        """Read the server, compute the remaining local-context budget.

        Raises :class:`~agentpause.errors.TelemetryError` -- never a raw
        :class:`~agentpause.errors.KVError` -- if the server is unreachable,
        the slot doesn't exist, or the configured/default field extractors
        can't find their expected keys in the response. The message always
        says explicitly that this is a LOCAL CONTEXT signal failure, not a
        rate limit, so it isn't mistaken for a 429-style condition.
        """
        try:
            props = self.slots.get_props(self.base_url)
            slot = self.slots.get_slot(self.base_url, self.id_slot)
        except KVError as exc:
            raise TelemetryError(
                f"Local context signal unavailable: llama.cpp server at "
                f"{self.base_url!r} is unreachable, or slot {self.id_slot} "
                f"does not exist. This is a CONTEXT-window read failure, not "
                f"a rate limit -- there is no rate limit on a local server. "
                f"Underlying error: {exc}"
            ) from exc

        try:
            context_size = int(self._context_field(props))
            used = int(self._used_field(slot))
        except (KeyError, TypeError, ValueError) as exc:
            raise TelemetryError(
                f"Could not read context/used token counts from the "
                f"llama.cpp server's /props or /slots response at "
                f"{self.base_url!r} -- the field names this class guesses by "
                f"default may not match your server/build. Pass "
                f"context_field=... / used_field=... explicitly. This is a "
                f"CONTEXT-window read failure, not a rate limit. Underlying "
                f"error: {exc}"
            ) from exc

        remaining = max(0, context_size - used - self.safety_margin_tokens)
        return Budget(remaining_tokens=remaining, limit_tokens=context_size)


# -- GPU VRAM -----------------------------------------------------------------

def _default_vram_reader(device_index: int) -> Tuple[int, int]:
    """Default VRAM reader: NVIDIA only, via `pynvml` (the `nvidia-ml-py` package).

    Imports `pynvml` LAZILY, inside the function body -- same discipline as
    `llamacpp_kv._default_get`'s lazy `httpx` import -- so importing this
    module, or agentpause itself, never requires `pynvml` to be installed;
    only actually calling this function (i.e. using :class:`GPUMemoryBudget`
    with no ``reader_fn`` override) does.

    Package note (verified 2026-07-21, see the module docstring): the PyPI
    package is ``nvidia-ml-py``, published and actively released by NVIDIA
    Corporation itself (current releases track NVML/driver versions); it
    installs as the `pynvml` importable module. A separate, older PyPI
    project literally named ``pynvml`` exists too and has a history of being
    an unofficial/unmaintained fork -- this module depends on ``nvidia-ml-py``
    (see the ``gpu`` extra in ``pyproject.toml``), not that one.

    NVIDIA-only, by design, today: this reads VRAM via NVML (NVIDIA
    Management Library). AMD/ROCm GPUs are NOT supported by this function --
    ROCm exposes its own management interface (`rocm-smi` / `amdsmi`), a
    genuinely different API this module has not implemented or tested
    against. A caller on ROCm hardware must supply their own ``reader_fn``;
    nothing here pretends to cover that case.

    Raises :class:`~agentpause.errors.GPUError` if `pynvml` is not
    installed, `nvmlInit` fails (no NVIDIA driver/GPU found), the device
    index does not exist, or any other NVML call fails.
    """
    try:
        import pynvml
    except ImportError as exc:
        raise GPUError(
            "pynvml is not installed -- required to read real VRAM from an "
            "NVIDIA GPU. Install the optional extra (`pip install "
            "'agentpause[gpu]'`, which pulls in the official `nvidia-ml-py` "
            "package) or pass reader_fn=... to supply VRAM numbers from "
            "elsewhere."
        ) from exc

    try:
        pynvml.nvmlInit()
    except Exception as exc:
        raise GPUError(
            f"pynvml.nvmlInit() failed -- no NVIDIA driver/GPU found, or "
            f"NVML could not be initialized: {exc}"
        ) from exc

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return int(info.free), int(info.total)
    except Exception as exc:
        raise GPUError(
            f"NVML query failed for GPU device_index={device_index}: {exc}"
        ) from exc
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass  # best-effort shutdown; never mask the real error above


class GPUMemoryBudget:
    """Turns REAL free GPU VRAM into a :class:`~agentpause.risk.Budget` (NVIDIA-only).

    Reads ``(free_bytes, total_bytes)`` for one GPU device -- straight from
    the hardware via NVML (see :func:`_default_vram_reader`), never
    estimated -- and, IF ``bytes_per_token`` is configured, turns the free
    byte count into a token budget the rest of the library already
    understands. Call it wherever a ``telemetry`` callable is expected
    (``PredictiveScheduler(telemetry=GPUMemoryBudget(...))``): it takes no
    per-call arguments, matching the ``Callable[[], int | Budget]`` shape
    the scheduler expects.

    IMPORTANT -- this is an EXTERNAL/SHARED signal, unlike context: free
    VRAM reflects EVERYTHING resident on the GPU at that instant, including
    other processes that have nothing to do with agentpause (another model,
    another agent, a desktop compositor). :class:`LlamaCppContextBudget`, by
    contrast, only ever reports what THIS session's own slot is using.
    Don't treat a VRAM budget as private to your process -- it isn't.

    Args:
        device_index: which GPU to read (as NVML enumerates them, matching
            ``nvidia-smi``'s device numbering). Default 0 (the first GPU).
        bytes_per_token: average VRAM bytes consumed by ONE token of
            KV-cache for the model actually being served. THERE IS NO
            UNIVERSAL CONSTANT for this -- it depends on hidden size, layer
            count, attention head count (GQA/MQA change it sharply vs. plain
            MHA), and quantization. If you don't pass it, this class cannot
            honestly convert bytes into tokens (see ``__call__`` below).

            How to calibrate it from a REAL KV save you've already made:
            :meth:`agentpause.llamacpp_kv.LlamaCppSlots.save` returns
            ``n_saved`` (KV cells actually written for that save), and the
            blob it writes to disk has a size in bytes readable with
            ``os.path.getsize``. For a model whose KV cache was fully
            resident in VRAM at save time::

                n_saved = slots.save(base_url, id_slot, filename)
                blob_bytes = os.path.getsize(os.path.join(kv_dir, filename))
                bytes_per_token = blob_bytes / n_saved

            This class does not compute that ratio itself (that would
            couple it to :class:`~agentpause.llamacpp_kv.KVStateStore` more
            tightly than a VRAM reader needs); it only documents how to
            arrive at the number. Treat it as model- and
            build-specific -- recalibrate whenever the served model changes.
        safety_margin_bytes: bytes subtracted from the raw free-VRAM count
            before conversion, as a cushion against other processes growing
            their own allocation between this read and the next call
            (default 0, no cushion).
        reader_fn: ``device_index -> (free_bytes, total_bytes)``. Defaults to
            :func:`_default_vram_reader` (real `pynvml`/NVML). Tests inject a
            fake returning fixed numbers -- no GPU, no `pynvml` install,
            required for the suite.
    """

    def __init__(
        self,
        device_index: int = 0,
        bytes_per_token: Optional[float] = None,
        safety_margin_bytes: int = 0,
        reader_fn: Optional[Callable[[int], Tuple[int, int]]] = None,
    ) -> None:
        self.device_index = device_index
        self.bytes_per_token = bytes_per_token
        self.safety_margin_bytes = safety_margin_bytes
        self._reader = reader_fn if reader_fn is not None else _default_vram_reader

    def __call__(self) -> Budget:
        """Read the GPU, compute the remaining VRAM (and, if possible, token) budget.

        Raises :class:`~agentpause.errors.TelemetryError` in two distinct
        cases, both worded to make clear this is NOT a rate limit:

        1. ``reader_fn`` (default or injected) raised ANYTHING -- driver
           missing, `pynvml` not installed, bad device index, a failed NVML
           call, or any other reader failure. The original exception (often
           a :class:`~agentpause.errors.GPUError` from the default reader)
           is chained via ``__cause__``.
        2. ``bytes_per_token`` was never set. The VRAM read itself may have
           succeeded perfectly -- ``free_bytes``/``total_bytes`` are known --
           but converting bytes into a token count would mean inventing a
           model-specific constant this class has no way to know. Rather
           than guess (which would silently misinform every downstream
           `decide()` call), this fails loudly and tells the caller exactly
           what to pass and how to derive it. Prefer this over a
           conservative-but-fabricated heuristic: a wrong "I don't know" is
           obvious and gets fixed; a wrong number that LOOKS plausible does
           not.
        """
        try:
            free_bytes, total_bytes = self._reader(self.device_index)
        except Exception as exc:
            raise TelemetryError(
                f"VRAM read failed for GPU device_index={self.device_index}: "
                f"{exc}. This is a VRAM-read failure, not a rate limit -- "
                f"there is no rate limit on local hardware telemetry."
            ) from exc

        if self.bytes_per_token is None:
            raise TelemetryError(
                "GPUMemoryBudget read the GPU successfully "
                f"(free={free_bytes}, total={total_bytes} bytes) but has no "
                "bytes_per_token configured, so free VRAM bytes cannot be "
                "honestly converted into a token count -- pass "
                "bytes_per_token=... (see the GPUMemoryBudget class "
                "docstring for how to calibrate it from a real KV save) to "
                "use this class as a token budget. Refusing to invent a "
                "number: this is not a rate limit, it's a missing "
                "conversion factor."
            )

        remaining_bytes = max(0, free_bytes - self.safety_margin_bytes)
        remaining_tokens = int(remaining_bytes / self.bytes_per_token)
        limit_tokens = int(total_bytes / self.bytes_per_token)
        return Budget(remaining_tokens=remaining_tokens, limit_tokens=limit_tokens)


# -- KV-aware wall-clock budget -------------------------------------------------

class KVAwareTimeBudget:
    """Wraps another ``telemetry`` callable, taking DIRECT ownership of the
    wall-clock (``remaining_seconds``) dimension of the reported
    :class:`~agentpause.risk.Budget`, INSTEAD OF relying on
    :class:`~agentpause.scheduler.PredictiveScheduler`'s own ``time_budget_s``
    bookkeeping.

    Why this exists at all -- read before using it
    -----------------------------------------------
    ``PredictiveScheduler`` can already turn a wall-clock deadline into
    ``Budget.remaining_seconds`` on its own: ``Session.__init__`` records
    ``self._started_at = scheduler.clock()``, and ``Session.next_action()``
    later does::

        if self._sched.time_budget_s is not None and budget.remaining_seconds is None:
            elapsed = self._sched.clock() - self._started_at
            budget.remaining_seconds = self._sched.time_budget_s - elapsed

    That mechanism lives ENTIRELY inside ``Session`` -- it uses the
    scheduler's own clock and the session's own ``_started_at``, neither of
    which an adapter/telemetry callable can see or influence from the
    outside. So a wrapper that wants to reserve time upfront (e.g. for
    writing a KV-cache blob to disk before the deadline hits) cannot just
    read ``inner_telemetry()``'s ``remaining_seconds`` and subtract a
    reserve from it: for exactly the local adapters in this module
    (:class:`LlamaCppContextBudget`, :class:`GPUMemoryBudget`),
    ``remaining_seconds`` comes back ``None`` -- it is the scheduler,
    downstream of this callable, that fills it in later, using state this
    class has no access to.

    ``KVAwareTimeBudget`` sidesteps that by doing its OWN wall-clock
    accounting, independent of ``PredictiveScheduler.time_budget_s``: it
    keeps its own ``_started_at`` (set in ``__init__``, using an injectable
    ``clock``) and computes ``remaining_seconds`` itself on every call,
    overwriting whatever ``inner_telemetry()`` may have already set.

    Reserving time for a KV-cache save
    -----------------------------------
    The whole point of doing this ourselves (rather than just duplicating
    ``PredictiveScheduler``'s own math) is to carve out a reserve for
    saving the KV-cache blob before the deadline actually hits, so
    ``decide()`` starts recommending ``checkpoint`` while there is still
    enough time left to actually perform the save. The reserve
    (``reserve_s``) is computed, in order of precedence:

    1. ``estimated_kv_save_s``, if given explicitly -- used as-is.
    2. Otherwise, if BOTH ``save_throughput_bytes_per_s`` and
       ``expected_blob_bytes`` are given: ``reserve_s = expected_blob_bytes
       / save_throughput_bytes_per_s``.
    3. Otherwise: ``reserve_s = 0.0``.

    Be honest with yourself about option 3: with no reserve estimate at
    all, this class's accounting is IDENTICAL to what
    ``PredictiveScheduler.time_budget_s`` already does for you -- same
    deadline, same elapsed-time subtraction, zero extra margin. The entire
    added value of ``KVAwareTimeBudget`` over the scheduler's own
    ``time_budget_s`` lives in ``reserve_s`` being nonzero; if you don't
    have a real number for the KV save cost, this wrapper buys you nothing
    you didn't already have, and it is fine to skip it and just pass
    ``time_budget_s=`` directly to ``PredictiveScheduler``.

    Don't set ``time_budget_s`` on ``PredictiveScheduler`` too: if you use
    this wrapper, there is no reason to ALSO pass ``time_budget_s=`` to
    ``PredictiveScheduler`` for the same deadline. ``__call__`` below
    always sets ``budget.remaining_seconds`` (never leaves it ``None``), so
    the scheduler's own ``if budget.remaining_seconds is None:`` check will
    always be False and its ``time_budget_s`` logic becomes a permanent
    no-op. Setting it too isn't wrong -- it just does nothing, and is
    confusing to a future reader. Recommended usage is
    ``PredictiveScheduler(time_budget_s=None, telemetry=KVAwareTimeBudget(...))``,
    with the deadline supplied ONLY here, via this class's own
    ``time_budget_s`` argument.

    The ``_started_at`` approximation
    -----------------------------------
    ``self._started_at = clock()`` is set in ``__init__``, at CONSTRUCTION
    time -- not at the first ``__call__``, and not by reading anything off
    the :class:`~agentpause.scheduler.Session` (this class has no reference
    to one; it is a plain telemetry callable). This is an approximation of
    "when the session started", and it is only accurate if you construct
    this object IMMEDIATELY BEFORE creating the ``Session``
    (``sched.session(...)``) that will use it. If you build a
    ``KVAwareTimeBudget`` long before the session it's wired into actually
    starts running steps, ``elapsed`` (measured from THIS constructor call)
    will overstate the real working time, and ``remaining_seconds`` will be
    reported smaller than it should be -- eating into your deadline for no
    real reason. The intended usage is to construct it right before
    ``sched.session(...)``, exactly like ``Session`` itself does with its
    own ``_started_at``.

    Args:
        inner_telemetry: the underlying ``() -> int | Budget`` telemetry
            callable this wraps (e.g. :class:`LlamaCppContextBudget` or
            :class:`GPUMemoryBudget`) -- supplies every field of the
            ``Budget`` OTHER than ``remaining_seconds``, which this class
            always overwrites.
        time_budget_s: the wall-clock deadline for the whole run, in
            seconds, measured from THIS object's construction (see the
            ``_started_at`` approximation above).
        estimated_kv_save_s: an explicit estimate, in seconds, of how long
            saving the KV-cache blob takes. Highest precedence for
            ``reserve_s`` when given.
        save_throughput_bytes_per_s: measured/assumed disk (or network)
            throughput for writing the KV blob, in bytes/second. Used with
            ``expected_blob_bytes`` to compute ``reserve_s`` ONLY when
            ``estimated_kv_save_s`` is not given.
        expected_blob_bytes: expected size, in bytes, of the KV-cache blob
            that will need saving. Used with
            ``save_throughput_bytes_per_s`` as above.
        clock: monotonic time source, injectable for tests (default
            ``time.monotonic``). Tests inject a fake returning a fixed,
            pre-scripted sequence of increasing values -- never real
            ``time.monotonic`` -- to keep timing assertions deterministic.
    """

    def __init__(
        self,
        inner_telemetry: Callable[[], "int | Budget"],
        time_budget_s: float,
        estimated_kv_save_s: Optional[float] = None,
        save_throughput_bytes_per_s: Optional[float] = None,
        expected_blob_bytes: Optional[float] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.inner_telemetry = inner_telemetry
        self.time_budget_s = time_budget_s
        self.estimated_kv_save_s = estimated_kv_save_s
        self.save_throughput_bytes_per_s = save_throughput_bytes_per_s
        self.expected_blob_bytes = expected_blob_bytes
        self.clock = clock
        self._started_at = clock()

        if estimated_kv_save_s is not None:
            self.reserve_s = float(estimated_kv_save_s)
        elif save_throughput_bytes_per_s is not None and expected_blob_bytes is not None:
            self.reserve_s = expected_blob_bytes / save_throughput_bytes_per_s
        else:
            self.reserve_s = 0.0

    def __call__(self) -> Budget:
        """Read ``inner_telemetry()``, then set ``remaining_seconds`` ourselves.

        Normalizes the inner reading EXACTLY the way
        ``agentpause.scheduler.Session._read_budget`` does (a bare ``int`` is
        legacy shorthand for ``Budget(remaining_tokens=int(raw))``; a
        :class:`~agentpause.risk.Budget` is used as-is) -- this class reuses
        that logic rather than inventing a second, subtly different
        normalization rule.

        ``remaining_seconds`` is then ALWAYS set to
        ``time_budget_s - elapsed - reserve_s``, unconditionally overwriting
        anything ``inner_telemetry()`` put there. This is deliberate: once
        you wrap a telemetry callable in ``KVAwareTimeBudget``, THIS class is
        the source of truth for the time dimension, not the inner callable
        and not the scheduler's own ``time_budget_s`` mechanism.

        The result can be NEGATIVE (when ``elapsed + reserve_s`` has already
        exceeded ``time_budget_s``) and is NOT floored at zero here:
        ``risk.decide()`` treats ``remaining_seconds`` as a plain deadline
        comparison (``estimated_latency <= budget.remaining_seconds``), so a
        negative value already reads as "no time left" and forces
        ``checkpoint`` -- flooring to 0 would not change that outcome, and
        would throw away the (potentially useful, for logging/diagnostics)
        information of exactly how far past the deadline the run is.
        """
        raw = self.inner_telemetry()
        budget = raw if isinstance(raw, Budget) else Budget(remaining_tokens=int(raw))

        elapsed = self.clock() - self._started_at
        budget.remaining_seconds = self.time_budget_s - elapsed - self.reserve_s
        return budget


# -- composing multiple local signals into one ---------------------------------

class CompositeLocalBudget:
    """Composes several ``telemetry`` callables into one, most-restrictive
    :class:`~agentpause.risk.Budget` reading.

    Why this exists
    -----------------
    :class:`~agentpause.scheduler.PredictiveScheduler` accepts exactly ONE
    ``telemetry=`` callable, but a self-hosted setup routinely has MORE than
    one real, independent ceiling at once -- e.g. a llama.cpp slot's own
    context window (:class:`LlamaCppContextBudget`) and the GPU's free VRAM
    (:class:`GPUMemoryBudget`), each a full ``telemetry`` callable on its
    own, each capable of being the one that runs out first depending on the
    workload and on what else is resident on the same card. This class calls
    every wrapped callable on each read and reports whichever
    ``remaining_tokens`` is LOWEST -- exactly the failure mode a real
    deployment has: the agent is stopped by whichever resource runs out
    FIRST, never by an average of the two.

    Normalization
    --------------
    Each raw reading is normalized EXACTLY the way
    :meth:`agentpause.scheduler.Session._read_budget` already does -- a bare
    ``int`` becomes ``Budget(remaining_tokens=int(raw))``, a
    :class:`~agentpause.risk.Budget` is used as-is -- so this class does not
    invent a second, subtly different normalization rule.

    Merging the OTHER Budget fields
    ---------------------------------
    ``remaining_tokens`` (and, riding along with it, ``reset_seconds`` /
    ``reset_requests_seconds`` / ``refill_regime``) come from the single
    winning (most restrictive) ``Budget``. None of the adapters in this
    module ever populate those three riding-along fields (a local context
    window has no refill clock -- see :class:`LlamaCppContextBudget`'s
    docstring), so there is nothing today to merge for them; they are simply
    whatever the winning budget happens to carry (``None`` in every case
    observed in this module).

    FOUR fields are instead merged ACROSS ALL sources, conservatively,
    because the source that wins on ``remaining_tokens`` is not necessarily
    the tightest on every other axis too -- e.g. the GPU budget could be the
    binding constraint on tokens while an independently-wrapped
    :class:`KVAwareTimeBudget` is, separately, the binding constraint on wall
    clock:

    * ``remaining_seconds`` -- the MINIMUM across every source that sets it
      (the tightest deadline wins); the lone value if only one source sets
      it; ``None`` if none do.
    * ``remaining_requests`` -- the same rule: fewest requests left wins.
      It is a depleting counter exactly like ``remaining_tokens``, so the
      same "lowest wins" logic applies for the same reason.
    * ``limit_tokens`` -- the same rule. This one is a capacity, not a
      "remaining" count, so "most restrictive" is a slightly looser fit --
      but reporting the SMALLEST capacity any source states, rather than the
      largest, keeps this class from ever describing the composite as
      roomier than the tightest single source actually says it is.
    * ``remaining_input_tokens`` / ``remaining_output_tokens`` -- the same
      min-across-sources rule, for consistency. No adapter in this module
      sets either field today, but a future local adapter that splits
      input/output would get correct behavior here with no changes needed.

    Error propagation: PROPAGATE, don't swallow
    ----------------------------------------------
    If any wrapped callable raises :class:`~agentpause.errors.TelemetryError`,
    it is left to PROPAGATE immediately -- this class does NOT catch it and
    quietly continue with whatever other sources still answered. A missing
    signal is not the same thing as "that resource has no limit": if, say,
    the GPU reader fails because NVML could not be reached this instant,
    silently falling back to context-only would let the scheduler answer
    ``continue`` while VRAM could in fact already be exhausted -- the
    failure would stay invisible until the process actually OOMs. This
    matches the stance the rest of this module already takes
    (:class:`GPUMemoryBudget` refuses to guess a token count rather than
    silently report a wrong one) and the library's general rule of never
    failing silently. Catch-and-continue with the remaining sources CAN be
    the right call in a different design -- e.g. if one signal is explicitly
    advisory rather than a hard ceiling -- but nothing wrapped by this class
    is advisory: every adapter in this module represents a real resource
    ceiling, so losing one without noticing is a bug waiting to happen, not
    a graceful degradation.

    Args:
        *telemetry_callables: one or more ``() -> int | Budget`` callables
            (e.g. :class:`LlamaCppContextBudget`, :class:`GPUMemoryBudget`,
            :class:`KVAwareTimeBudget`, or even a cloud adapter's own
            ``.budget``/``.telemetry`` -- this class does not require its
            inputs to be "local", it only composes whatever it is given).

    Raises:
        ValueError: if constructed with zero callables -- there would be
            nothing to compose, and a scheduler wired to an empty composite
            would silently see no real budget at all instead of a clear
            setup error.
    """

    def __init__(self, *telemetry_callables: Callable[[], "int | Budget"]) -> None:
        if not telemetry_callables:
            raise ValueError(
                "CompositeLocalBudget needs at least one telemetry callable"
            )
        self.telemetry_callables = telemetry_callables

    def __call__(self) -> Budget:
        """Read every wrapped callable; return the most restrictive Budget.

        Calls each wrapped callable in order given to ``__init__``. Any
        :class:`~agentpause.errors.TelemetryError` raised by a source
        propagates immediately (see the class docstring for why). Otherwise
        normalizes every reading, picks the ``Budget`` with the lowest
        ``remaining_tokens`` as the winner, and merges
        ``remaining_seconds``/``remaining_requests``/``limit_tokens``/
        ``remaining_input_tokens``/``remaining_output_tokens`` across ALL
        sources by taking the minimum of whichever ones set each field
        (``None`` if none do).
        """
        budgets = []
        for telemetry in self.telemetry_callables:
            raw = telemetry()
            budgets.append(raw if isinstance(raw, Budget) else Budget(remaining_tokens=int(raw)))

        winner = min(budgets, key=lambda b: b.remaining_tokens)

        def _min_across(attr: str) -> Optional[float]:
            values = [getattr(b, attr) for b in budgets if getattr(b, attr) is not None]
            return min(values) if values else None

        return Budget(
            remaining_tokens=winner.remaining_tokens,
            remaining_requests=_min_across("remaining_requests"),
            reset_seconds=winner.reset_seconds,
            reset_requests_seconds=winner.reset_requests_seconds,
            remaining_input_tokens=_min_across("remaining_input_tokens"),
            remaining_output_tokens=_min_across("remaining_output_tokens"),
            limit_tokens=_min_across("limit_tokens"),
            refill_regime=winner.refill_regime,
            remaining_seconds=_min_across("remaining_seconds"),
        )


# -- price-per-1k-tokens, derived locally --------------------------------------

def estimate_local_price_per_1k_tokens(tokens_per_second: float, hourly_cost: float) -> float:
    """Derive a ``price_per_1k_tokens`` for :class:`~agentpause.scheduler.PredictiveScheduler`
    from a measured local throughput and an assumed hourly cost.

    Pure arithmetic, no I/O: ``tokens_per_second`` scales up to tokens/hour,
    ``hourly_cost`` divided by that gives a cost per token, and the result is
    scaled to a per-1000-token price -- the same unit
    ``PredictiveScheduler(price_per_1k_tokens=...)`` already expects, exactly
    as if it had come from a cloud provider's price sheet::

        tokens_per_hour = tokens_per_second * 3600
        cost_per_token = hourly_cost / tokens_per_hour
        price_per_1k_tokens = cost_per_token * 1000

    ``hourly_cost`` is whatever the caller assumes it costs, per hour, to run
    the hardware generating those tokens -- a cloud GPU rental rate, or the
    output of :func:`estimate_hourly_cost_from_power` for real electricity
    cost. This function does not care which; it only does the unit
    conversion.

    Raises:
        ValueError: if ``tokens_per_second <= 0`` (a zero or negative
            throughput cannot be converted into a price -- it would mean
            dividing by zero or inverting the sign of the result), or if
            ``hourly_cost < 0`` (a negative cost is not a meaningful input
            here).
    """
    if tokens_per_second <= 0:
        raise ValueError(
            f"tokens_per_second must be positive, got {tokens_per_second!r} -- "
            "a zero or negative throughput cannot be converted into a price."
        )
    if hourly_cost < 0:
        raise ValueError(
            f"hourly_cost must not be negative, got {hourly_cost!r}."
        )
    tokens_per_hour = tokens_per_second * 3600.0
    cost_per_token = hourly_cost / tokens_per_hour
    return cost_per_token * 1000.0


def estimate_hourly_cost_from_power(watts: float, price_per_kwh: float) -> float:
    """Convert a power draw (Watts) and an energy price ($/kWh) into an hourly cost.

    For anyone who wants ``hourly_cost`` (as consumed by
    :func:`estimate_local_price_per_1k_tokens`) to reflect real electricity
    spend instead of an assumed cloud GPU rental rate. Pure arithmetic, no
    I/O::

        hourly_cost = watts / 1000.0 * price_per_kwh

    (``watts / 1000`` converts to kW; one hour of that draw is exactly
    ``kW * price_per_kwh`` dollars.)

    Raises:
        ValueError: if ``watts < 0`` or ``price_per_kwh < 0`` -- neither a
            negative power draw nor a negative energy price is a meaningful
            input here.
    """
    if watts < 0:
        raise ValueError(f"watts must not be negative, got {watts!r}.")
    if price_per_kwh < 0:
        raise ValueError(f"price_per_kwh must not be negative, got {price_per_kwh!r}.")
    return watts / 1000.0 * price_per_kwh


def price_per_1k_tokens_from_estimator(
    estimator: Any, input_tokens: int, hourly_cost: float,
) -> Optional[float]:
    """Derive ``price_per_1k_tokens`` from an already-calibrated ``Estimator``, if possible.

    The whole point of this function is to avoid a separate throughput
    benchmark: a :class:`~agentpause.regression.FeatureEstimator` already
    passed to ``PredictiveScheduler(estimator=...)`` has, purely as a side
    effect of doing its normal job, learned both how many output tokens the
    model tends to produce for a given input
    (:meth:`~agentpause.estimator.Estimator.estimate`) and how many seconds
    that takes (``estimate_latency``, if the estimator implements it --
    see :class:`~agentpause.regression.FeatureEstimator.estimate_latency`).
    Dividing one by the other IS the real, measured tokens/second throughput
    of this exact model on this exact hardware -- no separate benchmark
    needed.

    Calls ``estimator.estimate(input_tokens)`` for the predicted output
    token count, and reads ``estimate_latency`` off the estimator via
    ``getattr(estimator, "estimate_latency", None)`` -- the same defensive
    pattern :mod:`agentpause.scheduler` itself uses (``next_action()``),
    because the base :class:`~agentpause.estimator.Estimator` does not
    implement ``estimate_latency`` at all, while
    :class:`~agentpause.regression.FeatureEstimator` does.

    Returns ``None`` -- deliberately, not an exception -- in either of two
    normal, expected situations where a throughput cannot honestly be
    computed:

    1. ``estimator`` has no callable ``estimate_latency`` at all (e.g. a
       plain :class:`~agentpause.estimator.Estimator`, which never tracks
       latency).
    2. ``estimate_latency`` exists but returns ``None`` (or a
       non-positive value) -- e.g. a fresh
       :class:`~agentpause.regression.FeatureEstimator` that has not yet
       recorded enough steps (``estimate_latency`` returns ``None`` until
       ``min_samples`` realized steps exist) to have a latency regression.

    Neither case is an error: both are ordinary states, especially early in
    a run before any history has accumulated, so callers should treat
    ``None`` as "not derivable yet" and simply skip setting
    ``price_per_1k_tokens`` for this step rather than treating it as a
    failure.

    When a positive predicted latency IS available, computes
    ``tokens_per_second = predicted_output_tokens / predicted_latency_s``
    and delegates to :func:`estimate_local_price_per_1k_tokens` with that
    throughput and the given ``hourly_cost`` -- so an invalid
    ``hourly_cost`` (negative) still raises ``ValueError`` exactly as
    calling that function directly would.
    """
    output_tokens = estimator.estimate(input_tokens)
    predict_latency = getattr(estimator, "estimate_latency", None)
    if not callable(predict_latency):
        return None
    est_latency = predict_latency(input_tokens)
    if est_latency is None or est_latency <= 0:
        return None
    tokens_per_second = output_tokens / est_latency
    return estimate_local_price_per_1k_tokens(tokens_per_second, hourly_cost)
