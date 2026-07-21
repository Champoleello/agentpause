# Changelog

All notable changes to agentpause. Follows [Keep a Changelog](https://keepachangelog.com)
and semantic versioning.

## [Unreleased]

### Added

- **`AgentFleet`: one-call facade over `BudgetRouter` + `MultiAgentCoordinator`
  + a per-agent `Estimator`** (`agentpause.fleet.AgentFleet`, exported
  top-level). The README's "Scaling up" section previously showed the only
  way to combine these three pieces: six manual steps per call (build a
  router, build a coordinator around `router.budget`, register each agent,
  build a separate estimator, compute `estimated`/`sigma` from it, call
  `coordinator.request(...)`, check `.action`, call `router.backend(...)`,
  call `coordinator.complete(...)`, call `est.record(...)`). `AgentFleet`
  collapses construction to one constructor call (`providers=`, optional
  `agents=`) and the per-step dance to one `.call(agent_id, messages)`,
  mirroring the plug-and-play treatment `LocalResourceBudget` already gave
  local-resource budgets. A fresh, isolated `Estimator` (or an injected
  `estimator_factory`, e.g. `FeatureEstimator`) is created per `agent_id` —
  one agent's learned history never leaks into another's. Unregistered
  agent ids auto-register at `priority=0`, mirroring
  `MultiAgentCoordinator`'s own behavior; re-registering an already-known
  agent id updates its priority without resetting its learned estimator
  state. `self.router`/`self.coordinator` stay the real, live objects for
  power-user access (`coordinator.arbitrate(...)` for the batch/contention
  case). Deliberate scope boundary, stated plainly in the docstring: this
  does **not** add checkpoint/resume for fleets — `Session.next_action()`
  computes its estimate from its own private estimator and calls `decide()`
  directly, and there is no existing seam to redirect that decision through
  the coordinator instead (the coordinator/router are intentionally separate,
  adapter-level constructs, not wired into `Session`/`PredictiveScheduler`).
  Pair `AgentFleet` with your own `StateStore` per `agent_id` if per-agent
  resumability is needed. New runnable example,
  `examples/fleet_facade_quickstart.py`: two agents sharing two providers
  through one `AgentFleet`, a shared pool draining to a clean simultaneous
  `checkpoint` for both agents, and a separate, tightly-scoped scenario
  showing `coordinator.arbitrate()` actually using agent priority (the
  streaming `.call()` loop does not — priority only matters in `arbitrate()`'s
  batch call, documented explicitly after this was checked by running the
  example, not assumed). `examples/fleet_quickstart.py` (the original manual
  composition) is kept as the "under the hood" fine-control demo.

## [0.5.0] — 2026-07-21

### Added

**Local resource budgets (self-hosted llama.cpp):** on a self-hosted
`llama-server` there are no rate-limit headers to build a `Budget` from —
there is no traffic limit to respect — so `should_suspend()` has nothing real
to evaluate locally unless something reads an ACTUAL resource. The seven
entries below (five real signals/guards, the composer that ties them
together, and a one-line plug-and-play facade over all of it) close that
gap; see the README's "Local mode: what's actually being controlled" section
and `examples/local_resources_quickstart.py` for the honest, end-to-end
story.

- **Optional disk-space guard for KV blob saves** (`llamacpp_kv.KVStateStore`):
  `save_with_kv` can now check free space on `kv_dir` before calling
  `slots.save()` (the HTTP POST that can run up to 1800s on a large context,
  see `_default_post`), avoiding paying that cost on a save that's almost
  certainly going to fail for lack of disk. Two new constructor parameters,
  both opt-in and appended after the existing ones so no existing call site
  changes behavior: `min_free_bytes` (default `None` — no check at all,
  identical to before) and `prune_oldest` (default `False` — if `True` and
  free space is short, tries `gc_orphans()` then `gc_consumed()` once before
  giving up). A third, `disk_usage_fn` (default `shutil.disk_usage`), makes
  the check injectable so the test suite stays fully offline. Raises
  `KVError` — same style as every other `save_with_kv` failure mode — before
  `cp` is mutated or the network is touched, preserving the existing
  transactional guarantee (KV blob to disk first, logical commit second).
- **Real local-context budget for llama.cpp** (`adapters.local_resources.LlamaCppContextBudget`):
  on a self-hosted `llama-server` there are no rate-limit headers to build a
  `Budget` from — there is no traffic limit to respect — so this reads the
  ACTUAL configured context size (`GET /props`) and the ACTUAL tokens used in
  a slot's KV cache (`GET /slots`), via two new `LlamaCppSlots` methods
  (`get_props`, `get_slot`, both factored to reuse/extend the existing
  `fingerprint`/`save`/`restore` HTTP client without touching their behavior),
  and turns that into a real, depleting `Budget` instead of an inert
  `fallback_remaining=N`. Field extraction (`context_field=`/`used_field=`)
  is injectable, defaulting to `n_prompt_tokens` — CONFIRMED live (2026-07-21)
  against a real `llama-server` serving Qwen3-8B-GGUF Q4_K_M (see the "Fixed"
  entry below for how this was found), with older guesses kept as a fallback
  chain for other llama.cpp-ecosystem forks/proxies that may shape this
  differently. Local context never "resets" like a cloud TPM/RPM window —
  only `compact()`/`summarize_with()` or a fresh session free it up.
- **Real GPU-VRAM budget** (`adapters.local_resources.GPUMemoryBudget`): a
  second, independent local signal — how much GPU memory is free RIGHT NOW,
  read straight from the hardware via NVML (NVIDIA's official `pynvml`
  bindings, PyPI package `nvidia-ml-py`; verified 2026-07-21 to be the
  actively-maintained package NVIDIA itself publishes, not the separate,
  historically unofficial `pynvml` PyPI project). Imported lazily inside an
  injectable `reader_fn: (device_index) -> (free_bytes, total_bytes)`, so
  the test suite stays fully offline — no `pynvml` install, no GPU required.
  NVIDIA-only for now; ROCm/AMD would need a genuinely different reader and
  is explicitly out of scope here, not silently pretended to work. Unlike
  `LlamaCppContextBudget` (scoped to this session's own KV slot), free VRAM
  is an EXTERNAL/shared signal — it reflects every process on the GPU, not
  just agentpause's own usage, and the docstring says so plainly. Converting
  free bytes into a token budget needs a model-specific `bytes_per_token`
  (no universal constant exists); if it isn't supplied, `GPUMemoryBudget`
  raises a `TelemetryError` explaining exactly what to pass and how to
  derive it from a real KV save (`LlamaCppSlots.save()`'s `n_saved` plus
  `os.path.getsize` on the blob) rather than silently guessing a number.
  New `GPUError` exception type for the underlying hardware-query failures
  (driver missing, `pynvml` not installed, bad device index, failed NVML
  call); new optional `gpu` extra (`pip install 'agentpause[gpu]'`).
- **KV-aware wall-clock budget** (`adapters.local_resources.KVAwareTimeBudget`):
  wraps another `telemetry` callable and takes over `Budget.remaining_seconds`
  accounting itself, rather than relying on `PredictiveScheduler.time_budget_s`
  — that mechanism lives entirely inside `Session` (its own clock, its own
  `_started_at`), so an outside wrapper can't read or reserve against it,
  and for local adapters like `LlamaCppContextBudget`/`GPUMemoryBudget`,
  `remaining_seconds` simply comes back `None` for the scheduler to fill in
  later. This class keeps its own `_started_at` and `clock`, and always sets
  `remaining_seconds = time_budget_s - elapsed - reserve_s`, where `reserve_s`
  carves out time to save the KV-cache blob before the deadline actually
  hits — from an explicit `estimated_kv_save_s`, or computed from
  `save_throughput_bytes_per_s` + `expected_blob_bytes`, or `0.0` if neither
  is given (in which case this reduces to exactly what
  `PredictiveScheduler.time_budget_s` already does on its own — the added
  value only exists once a real reserve estimate is supplied). Docstring is
  explicit that pairing this with `PredictiveScheduler(time_budget_s=...)`
  for the same deadline is redundant, not broken: this wrapper always sets
  `remaining_seconds`, so the scheduler's own `if remaining_seconds is None`
  fallback becomes a permanent no-op.
- **Local price-per-1k-tokens helpers** (`adapters.local_resources.estimate_local_price_per_1k_tokens`,
  `estimate_hourly_cost_from_power`, `price_per_1k_tokens_from_estimator`):
  three pure functions (no I/O, no new exception type — just `ValueError` on
  bad input) that derive `PredictiveScheduler`'s existing
  `price_per_1k_tokens`/`money_budget` parameters for a local setup, where
  there is no provider invoice to read a price off of. `price_per_1k_tokens_from_estimator`
  reads the throughput straight off an already-calibrated `Estimator`/
  `FeatureEstimator` (via `estimator.estimate()` and, defensively via
  `getattr` exactly like `scheduler.py` does, `estimate_latency`) instead of
  requiring a separate benchmark — it returns `None`, not an error, whenever
  a throughput genuinely can't be known yet (a plain `Estimator` with no
  `estimate_latency` at all, or a fresh `FeatureEstimator` that hasn't
  recorded enough steps). This is the lightest of the local additions: it
  introduces no new scheduler mechanism, only makes it easy to populate two
  parameters that already existed and already worked identically in local
  and cloud use.
- **Composing multiple local signals into one** (`adapters.local_resources.CompositeLocalBudget`):
  `PredictiveScheduler` accepts exactly one `telemetry=` callable, but a real
  self-hosted deployment routinely has more than one hard local ceiling at
  once — e.g. `LlamaCppContextBudget` AND `GPUMemoryBudget` are independent
  and either can run out first. `CompositeLocalBudget(*telemetry_callables)`
  calls every wrapped callable on each read and reports whichever
  `remaining_tokens` is LOWEST — the same failure mode a real deployment has:
  stopped by whichever resource runs out first, never an average of them.
  `remaining_seconds` / `remaining_requests` / `limit_tokens` /
  `remaining_input_tokens` / `remaining_output_tokens` are merged
  conservatively across ALL sources (the minimum of whichever ones set each
  field); other fields ride along on the winning budget. A
  `TelemetryError` raised by any wrapped source PROPAGATES immediately rather
  than being swallowed — a missing signal is not the same thing as "that
  resource has no limit," and silently falling back to the sources that did
  answer could let the scheduler say `continue` while an unreadable resource
  is in fact already exhausted (see the class docstring for the full
  reasoning, and for why catch-and-continue would be the wrong default
  here). Raises `ValueError` if constructed with zero callables. New runnable
  example, `examples/local_resources_quickstart.py`: a fake, growing
  llama.cpp context budget composed with a fake, shrinking GPU-VRAM budget,
  wrapped in a `KVAwareTimeBudget` reserve, a `price_per_1k_tokens` derived
  from a `FeatureEstimator`'s own throughput, and a real `PredictiveScheduler`
  driven by the composite — checkpointing the moment one local signal (never
  a cloud rate limit) runs out.
- **`LocalResourceBudget`: a one-line, self-calibrating facade over all five
  local pieces above** (`adapters.local_resources.LocalResourceBudget`).
  Using `LlamaCppContextBudget` + `GPUMemoryBudget` + `KVAwareTimeBudget` +
  `CompositeLocalBudget` + `KVStateStore` together used to mean six manual
  pieces, two of which needed a hand calibration step (`bytes_per_token`,
  `estimated_kv_save_s`) most users would never know to perform correctly.
  `LocalResourceBudget(kv_dir=...)` reads `GET /props` immediately (fails
  fast, loudly, if the server isn't reachable — a plug-and-play convenience
  should not hide a broken server behind a lazy retry) and detects GPU
  availability without needing `bytes_per_token` yet; `kv_dir` (which MUST
  match the server's own `--slot-save-path`) is the one thing this class
  cannot discover over HTTP and is therefore the only required argument.
  Calling it as `telemetry=` composes a `LlamaCppContextBudget` alone until
  a GPU is both detected AND calibrated, at which point `GPUMemoryBudget`
  is composed in automatically via `CompositeLocalBudget`; `kv_store(store)`
  returns a `KVStateStore` whose `save_with_kv` is wrapped to calibrate
  `bytes_per_token`, the measured KV-save duration, and (unless the caller
  set an explicit `min_free_bytes`) the disk-guard threshold
  (`max(500_000_000, 5 * blob_bytes)`) from every real save — never from a
  synthetic probe. `suggest_price_per_1k_tokens(...)` delegates to
  `price_per_1k_tokens_from_estimator`, falling back to a constructor-level
  `hourly_cost` and raising `ValueError` only if neither is ever supplied.
  None of the five underlying pieces are removed or deprecated — this is
  purely an additive, opt-in convenience layer for the common single-server
  case.

### Fixed

- **`LlamaCppContextBudget`'s default `used_field`: `n_prompt_tokens`,
  confirmed live**, not the best-effort guess chain originally shipped.
  Found by actually running a real `llama-server` (Qwen3-8B-GGUF Q4_K_M,
  `-c 10240`) and diffing a slot's `GET /slots` entry before and after a real
  completion: `n_prompt_tokens` tracks prompt + generation tokens together
  (measured: 91 prompt + 200 generated → `n_prompt_tokens: 290`, matching
  within rounding) and is simply absent on a slot that has never run a task.
  An idle slot (`is_processing: false`, no `id_task`) now correctly reports 0
  tokens used instead of raising; `next_token` is also handled as the LIST of
  per-attempt dicts a real server actually sends, not the bare dict
  originally assumed. Older guesses (`cache_tokens`, `n_past`, `prompt`,
  `tokens`) are kept as a fallback chain for other forks/proxies, checked
  only after `n_prompt_tokens`.
- **`llamacpp_kv.KVStateStore.save_with_kv`: fail fast on a `kv_dir` /
  `--slot-save-path` mismatch**, instead of discovering it much later.
  Found live (2026-07-21): if `kv_dir` (this plugin's own bookkeeping
  directory) doesn't match the real `--slot-save-path` the server was
  started with, the server can still report a perfectly valid `n_saved` —
  it wrote the blob into its OWN save directory, which this plugin never
  looks in — and `save_with_kv` would "succeed" silently, committing a
  checkpoint that pointed at a KV file that was never actually where
  expected. That mismatch used to surface only much later, at the NEXT
  `load_with_kv`, as `reason="kv_file_missing"` — far away in time from the
  real cause. `save_with_kv` now checks, immediately after `slots.save()`
  reports success and strictly before reading the model fingerprint or
  touching `cp` at all, that the blob actually landed under `kv_dir`; if
  not, it raises `KVError` with a message that explicitly names the likely
  cause (`kv_dir` vs. `--slot-save-path`) and suggests checking that
  correspondence. Same transactional guarantee as every other failure mode
  in this method: nothing about `cp` or a previously-valid checkpoint
  changes when this check fails.

## [0.4.0] — 2026-07-20

### Added
- **ollama-gateway support**: `KNOWN_PROVIDERS["ollama-gateway"]` for
  `OpenAICompatAdapter.for_model("ollama-gateway/<model>")` against a local
  or self-hosted `martinobettucci/ollama-gateway` instance — no code changes
  needed beyond the registry entry, since the gateway already emits
  OpenAI/Groq-style rate-limit headers our adapter reads natively. Confirmed
  live (2026-07-20) against a real gateway instance: `adapter.budget()`
  correctly reads `remaining_tokens`/`remaining_requests`/`reset_seconds` —
  the one requirement is that the API key has a quota configured on the
  gateway side; a key with no quota emits no rate-limit headers at all, and
  agentpause surfaces that as a clear `TelemetryError` rather than a silent
  wrong reading.
- **`quality_ab.py`: VOICE dimension.** Alongside the existing 6-fact recall
  probe, a persona ("Zappy") with two short, literal catchphrases is planted
  in the same conversation, and `voice_score()` counts how many survive
  slimming (same substring-match style as the facts probe). Live result
  (2026-07-20, Groq `llama-3.1-8b-instant`): A facts 6/6 voice 1/2; B
  (`compact()`) facts 0/6 voice 1/2; C (`summarize_with()`) facts 6/6 voice
  0/2 — confirms a Reddit reader's hypothesis (Budget_Ad_5787, r/LLMDevs)
  that semantic summarization trades style for facts, while blind truncation
  keeps short verbatim tics but loses facts scattered earlier in the
  history. No new CLI flag: the voice probe rides along on the existing
  three calls.
- **Checkpoint fork & migration** (F11.2): `Checkpoint.fork()` /
  `StateStore.fork()` — one suspended past, N independent futures. Clones get
  deep-copied `messages`/`extra`, their own `session_id`, and a collision-free
  idempotency namespace (keys embed the session id, so a forked branch is
  never deduplicated against a sibling's side effects), while the learned
  estimator statistics in `extra['estimator']` ride along on purpose (F9.3
  symmetry: a fork starts calibrated). `StateStore.export_bundle()` /
  `import_bundle()` turn the checkpoint into a versioned, json-portable
  process image: suspend on machine A, ship the bundle over any transport,
  resume on machine B at the exact step. Machine-local KV-cache accelerators
  (self-hosted plugins) deliberately do NOT migrate — resume degrades
  gracefully to a logical warm start. Runnable demo:
  `examples/migrate_fork.py`.
- **True warm start for llama.cpp (KV-cache plugin)**: `LlamaCppSlots` (thin,
  injectable HTTP client for `/slots/{id}?action=save|restore` and `/props`)
  and `KVStateStore` (wraps any `StateStore`; transactional save — KV blob to
  disk first, logical checkpoint commit second, so a KV failure never
  corrupts prior state; fingerprint-gated restore that degrades to a logical
  warm start on model mismatch, missing blob, or migration to a new machine;
  `fork_with_kv` gives each forked branch its own independent blob; blob GC
  for consumed/orphaned files). Runnable demo:
  `examples/kv_llamacpp_demo.py`.
- **Task-completion forecast** (`session.forecast(steps_remaining)`): predicts
  total tokens, money, expected waits/suspensions and wall-clock time for the
  rest of the run by simulating the remaining steps against the live window
  and its refill rate. Honest flags: `context_wall` when a single step can
  never fit a whole window (§8.6), `fits_time_budget` against a run deadline.
- **Human attention as a budget** (`HumanAttentionBudget`): the person in the
  loop is a rate-limited window too - N questions per rolling hour, manual
  `available()`/`away(until)`, and `as_budget()` so the standard three-valued
  decision (continue/wait/checkpoint) applies to "may I interrupt the human?"
  exactly as it does to TPM. An absent human with no return time means
  checkpoint, never busy-wait.

### Fixed
- **`llamacpp_kv`: KV save/restore now sends the server a BARE filename.**
  Found by a live test against a real llama-server: `save_with_kv`/
  `load_with_kv` were sending `filename` prefixed with the local `kv_dir`
  (e.g. `kv_cache/mission_ab12cd34.bin`), but llama-server resolves the
  `filename` it receives against its OWN `--slot-save-path`, so the prefixed
  path was resolved a second time and produced a nonexistent nested
  directory — the server correctly rejected it with `400 Bad Request`. The
  bug was invisible in the test suite because the `FakeSlots` double simply
  wrote to whatever path it was given, path-like or not. Fixed by sending a
  bare filename to `slots.save()`/`slots.restore()` always, and documented
  the requirement (`kv_dir` must equal the server's `--slot-save-path`) on
  `KVStateStore`. `FakeSlots` in both the test suite and the runnable demo
  now models the server's real path-resolution behavior instead of accepting
  anything, so this class of bug can't silently regress again.

## [0.3.0] — 2026-07-10

### Added
- **Age-adjusted reset countdowns**: telemetry served from cache (up to
  `max_age_s` old) now returns `reset_seconds` / `reset_requests_seconds`
  minus the reading's age (floor 0) — callers no longer wait on seconds
  that already elapsed. Both direct and LiteLLM adapters.
- **`rpm_margin`** (`decide()` / `PredictiveScheduler`): optional safety
  slack on the request budget — pause while `remaining <= rpm_margin`
  requests instead of riding the window to zero. Default 0 (historical
  behavior); field practice elsewhere (e.g. Hermes agent's pre-emptive
  RPM throttler) keeps 1-2 in reserve against stale telemetry and
  concurrent consumers of the same key.
- **Fleet quickstart** (`examples/fleet_quickstart.py`): runnable without keys —
  BudgetRouter switching providers + MultiAgentCoordinator issuing a predictive
  WAIT under a shared window + FeatureEstimator learning per-tool cost.
- **Feature-based cost estimator** (`FeatureEstimator`, D6): predicts next-step
  tokens AND latency from workload features (context length, tool, model,
  temperature, output length) via a standardized ridge regression, not size
  alone. Drop-in for `Estimator` — with only context length it degrades to a
  linear-in-size model, and it falls back to the base estimator until it has
  data. Feed richer features with `set_context(tool=..., model=...)`;
  categoricals are one-hot encoded. Learned samples persist in the checkpoint.
- **Latency as a budget dimension**: `Budget.remaining_seconds` (a task
  deadline) and `decide(estimated_latency=...)`. Time never refills by waiting,
  so a step that can't finish before the deadline yields `checkpoint`, never
  `wait`. Wire a whole-run deadline via `PredictiveScheduler(time_budget_s=...)`;
  `FeatureEstimator` supplies the per-step latency prediction.
- **Multi-agent shared-budget coordination** (`MultiAgentCoordinator`): one
  rate-limit window, many agents. A granted call reserves its predicted cost
  from the shared pool, so agents can't independently overcommit the same
  window and 429 each other. `arbitrate()` resolves contention for a pool too
  small for all — highest priority first, then longest-waiting — which is where
  priority/deadline/fairness finally apply (meaningless with one agent).
  Composes with `BudgetRouter`: route across providers AND share one across
  agents.
- **Budget-aware multi-provider routing** (`BudgetRouter`): predictive sibling
  of `FallbackBackend`. Reads every provider's telemetry FIRST and sends the
  next call to the window with the most headroom — before anyone hits a 429 —
  instead of switching only after a failure. Duck-types the adapter shape
  (`backend` + `budget`), so it drops into `PredictiveScheduler` in place of a
  single adapter. Rate-limited providers are parked (cooldown) until their
  window resets; selection metric is overridable via `key=` (default:
  `remaining_tokens`).
- **Useful waits** (`AgentPauseGuard(while_waiting=...)`): wait time is handed
  to the app for LLM-free work (indexing, memory compression, prompt prep)
  instead of being slept away.
- **Preventive compaction** (`PredictiveScheduler(context_window=...,
  compact_at=0.6)`): old history shrinks when context pressure crosses the
  threshold — amortized, before the §8.6 wall, with a `compacted` event.
- **Semantic summarization** (`session.summarize_with(fn)` /
  `Checkpoint.summarize_with`): old messages replaced by ONE summary from an
  injected cheap model (ideally a different provider, working while the
  saturated one rests). Truncating `compact()` stays as model-free fallback.
- **Phase-shift detection** (Estimator, on by default): when recent
  estimation errors shift coherently away from older ones (exploration →
  synthesis), stale history is dropped so σ and the quantile track the new
  phase. Straddling windows are rejected (tightness test).
- **Estimator persistence** (automatic): learned statistics (ε, σ, error
  history) ride inside the checkpoint — a resumed session starts calibrated.
  Unlike the budget, statistics do not go stale during a suspension.
- **Direct Anthropic adapter** (`adapters.anthropic.AnthropicAdapter`):
  /v1/messages protocol, split input/output token telemetry, RFC 3339 resets
  — plus prompt-cache breakpoints on the stable prefix (`cache_prompt=True`),
  the cloud analog of the KV warm start, with MEASURED savings
  (`cache_read_tokens`). Caveats documented: minutes-scale TTL; discounts
  price/latency, not the rate-limit count.
- **Tool quotas** (`ToolQuota`): client-side sliding-window budgets for
  rate-limited tools (search APIs, scrapers) that send no headers —
  `ready()`/`wait_seconds()` for predictive checks, `guard(fn)` to pace a
  tool automatically.

## [0.2.2] — 2026-07-08

### Added (hardening from the live stress test)
- **Anti-livelock guard** (§8.6 of the research): when the next call needs
  ~the whole TPM window, waiting can never succeed (telemetry pings nibble
  whatever refills — the bucket hovers a hair below the bar forever). The
  decision is now `checkpoint`: a resume starts against a full, untouched
  window.
- **Overflow policy — mandatory summarize** (`Checkpoint.compact()` /
  `session.compact()`): when the context itself no longer fits a full
  window, old messages are truncated (system head and recent tail kept).
  Works OFFLINE on a suspended checkpoint — compression is useful work
  that needs no LLM, done while the agent sleeps.
- **Sliding-window quantile** (default last 30 steps): agent behavior changes
  phase on long tasks; a monster step from an old phase no longer fattens the
  margin forever. With few samples the bound degrades — declaredly — to the
  max of recent errors.

### Fixed (found live by the stress test)
- **RPM livelock**: when the request budget (RPM) was the binding constraint,
  waits were computed on the TOKEN reset clock — telemetry pings then ate
  every refilled request slot in a self-sustaining loop. Requests now wait on
  their own clock (`Budget.reset_requests_seconds`, parsed from
  `x-ratelimit-reset-requests`).
- **429 on a telemetry ping crashed the decision path.** It is now absorbed
  as telemetry (zero budget + provider's `retry-after`), never raised.

Completes the 0.2.1 batch (0.2.1 was published mid-development and is
superseded — prefer 0.2.2). On top of 0.2.1 as released: regime-detection
sampling fixed (real calls now start the sampling chain, so the detector
actually votes during single-chunk waits), honest telemetry accounting
(`ping_tokens`/`ping_count`, benchmark row), `invalidate()` on both adapters
(fresh telemetry after sleeping out a wait), and `scripts/stress_test.py`
(long irregular multi-window task exercising the full ladder, with
checkpoint/resume across runs).

## [0.2.1] — 2026-07-08 (superseded by 0.2.2)

### Added
- **Refill-aware waiting**: token buckets refill continuously, so the wait is
  now `deficit / refill_rate` (derived from headers alone:
  `(limit - remaining) / reset`), capped at the full reset, with the
  wait-vs-checkpoint threshold compared against the *effective* wait.
  `Budget.limit_tokens` + `Decision.wait_seconds`. Safe on fixed-window
  providers too: every wait is followed by a fresh telemetry read.

- **Refill-regime detection** (`RegimeDetector`): consecutive telemetry pings
  with no real call in between reveal whether the provider refills
  continuously (token bucket → refill-aware waits) or all at once at a
  boundary (fixed window → full-reset waits). Measured online, zero
  configuration; `Budget.refill_regime` carries the verdict.
- **Chunked waiting** (guard `chunk_s`, default 10 s): long waits sleep in
  chunks with a fresh telemetry read in between — resume as soon as the
  bucket actually holds enough, and feed the regime detector for free.

### Changed
- `Estimator.sigma` now measures the spread of estimation RESIDUALS instead
  of raw consumption: context growth no longer inflates the safety margin,
  shortening waits and reducing premature suspensions at equal coverage.
- Benchmark now levels the field between conditions (waits for a full
  window before each), so neither inherits a drained bucket.
- `OpenAICompatAdapter`: direct-HTTP adapter for OpenAI-compatible providers
  (Groq, OpenAI, ...) reading rate-limit headers at the source. Validated
  live on Groq (5-step task, live TPM/RPM telemetry, zero 429).

### Fixed
- LiteLLM adapter: enable `litellm.return_response_headers` automatically.
  Note: litellm currently drops provider headers anyway
  (BerriAI/litellm#11749) — use `OpenAICompatAdapter` for real telemetry
  until that bug is fixed upstream.

## [0.2.0] — 2026-07-07 · robustness & telemetry batch (shipped in the 0.2.0 upload)

### Added
- `FallbackBackend`: ordered model fallback chain — switches on 429/retriable
  failures only, exposes `last_index` and `on_fallback` for downstream
  validation of fallback-model outputs.
- Monetary hard constraint: `price_per_1k_tokens` + `money_budget` on the
  scheduler; a predicted overrun decides `checkpoint` (never `wait`);
  `money_spent` / `money_remaining` tracked per step.
- Observability hook `on_event(name, info)`: `decision`, `step_completed`,
  `rate_limit_hit`, `retry`. Hook exceptions never break the run.
- `CircuitBreaker`: CLOSED→OPEN→HALF_OPEN wrapper for any backend — fails
  fast (`CircuitOpenError`) during outages instead of hammering the provider;
  request errors (4xx) never trip it; composes with `FallbackBackend`.
- Split input/output token budgets (`Budget.remaining_input_tokens` /
  `remaining_output_tokens`): providers like Anthropic limit the two
  dimensions separately, and `decide()` now honors both when reported.
- Anthropic-style headers recognized automatically
  (`anthropic-ratelimit-*-remaining`, RFC 3339 reset timestamps) alongside
  OpenAI/Groq-style names.
- Quantile margins (`PredictiveScheduler(quantile=0.95)`): with enough
  history, the decision uses the empirical q-quantile of past estimation
  errors instead of the `k·sigma` margin — honest with heavy-tailed
  consumption; falls back to `k·sigma` until 8 steps are recorded.
- Jitter (±25% default) in `RetryPolicy` backoff, against thundering herds.
- Retriable-vs-non-retriable error classification: provider 5xx/timeouts map
  to `BackendError(retriable=True)` and are retried; 4xx propagate untouched.

## [0.2.0] — 2026-07-07 · adapters & decision batch

### Added
- **LiteLLM adapter** (`agentpause.adapters.litellm.LiteLLMAdapter`): backend +
  telemetry for 100+ providers; budget read from rate-limit response headers;
  stale readings refreshed with a 1-token telemetry ping.
- **LangGraph adapter** (`agentpause.adapters.langgraph.AgentPauseGuard`):
  predictive gate for graph nodes via LangGraph's native `interrupt()` +
  checkpointer; fresh telemetry on every resume pass.
- **Three-valued decision** (`Budget`, `decide`, `Session.next_action`):
  `continue` / `wait` / `checkpoint`. Considers remaining tokens (TPM),
  remaining requests (RPM), and time-to-reset (wait when the window resets
  within `wait_threshold_s`).
- **Typed errors** (`AgentPauseError`, `RateLimitHit`, `TelemetryError`,
  `CheckpointError`, `BackendError`).
- **Retry with backoff** (`RetryPolicy`): unexpected 429s honored via
  `retry-after` or exponential backoff; failed calls leave session state
  untouched; each hit adaptively bumps `safety_k` (capped at `k_max`).
- **Async support**: `Session.acall()`, `LiteLLMAdapter.abackend/atelemetry/abudget`,
  `AgentPauseGuard.acheck()`.
- **Real tokenizer**: `LiteLLMAdapter.count_tokens` (per-model, heuristic fallback).
- Validation script (`scripts/validate_provider.py`) for any real provider.

## [0.1.0] — 2026-07-03

### Added
- Core components: `Estimator` (ε moving-average correction, σ tracking),
  `should_checkpoint` rule, `RiskModel`, `StateStore`/`Checkpoint` (atomic,
  idempotency keys).
- High-level API: `PredictiveScheduler` / `Session` with injectable backend
  and telemetry.
- Runnable quickstart (no API keys), MIT license, full offline test suite.
