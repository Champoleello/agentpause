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
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from ..errors import GPUError, KVError, TelemetryError
from ..llamacpp_kv import LlamaCppSlots
from ..risk import Budget

__all__ = ["LlamaCppContextBudget", "GPUMemoryBudget"]


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
