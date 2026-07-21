# Changelog

All notable changes to agentpause. Follows [Keep a Changelog](https://keepachangelog.com)
and semantic versioning.

## [Unreleased]

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
