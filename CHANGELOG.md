# Changelog

All notable changes to agentpause. Follows [Keep a Changelog](https://keepachangelog.com)
and semantic versioning.

## [Unreleased]

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

## [0.2.0] — 2026-07-07

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
