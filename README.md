# agentpause

**Predictive scheduling for autonomous LLM agents.** Suspend an agent gracefully
*before* it hits a provider rate limit, and resume it without redoing work.

The core works on any provider — cloud (OpenAI, Anthropic, Groq) or local — because
it only serializes application-level state. True KV-cache warm start is an optional
plugin for self-hosted runtimes (llama.cpp, vLLM).

## Measured results (from the accompanying research)

| Experiment | Result |
|---|---|
| Simulation, 300 runs/config | reactive baseline crashes up to 100% → predictive **0%** (k=4), token waste −80% |
| Real provider (Groq free, 6K TPM) | multi-window task completed with **zero 429 errors** |
| vs LangGraph's MemorySaver (200 runs) | LangGraph: 7.8 × 429/run, 9,239 tokens wasted; predictive: **0 and 0** |
| End-to-end A/B (thinking models, 4B/8B) | recovery **54×–93× faster**; total task time −19% |

Reproduce them yourself with a free key: `python scripts/benchmark_groq.py`.
Live run (2026-07-08, Groq free tier, 12 steps, ~1k-token context,
refill-aware waiting enabled):

|  | reactive baseline | agentpause |
|---|---|---|
| 429 errors suffered | 7 | **0** |
| steps redone | 7 | **0** |
| tokens re-sent (waste) | 12,520 | **0** |
| wall-clock | 79 s | 145 s* |

\* zero waste costs some waiting. Refill-aware math already cuts most waits
from ~50 s to ~10 s; the residue is a benchmark artifact (condition B starts
right after A has drained the shared TPM window) plus the safety cap when a
late-task call needs nearly the whole bucket. Waste is money on paid tiers —
waiting is free.

> Status: **early alpha (0.1)**. Core components, the high-level
> `PredictiveScheduler` API, the LiteLLM adapter (any provider), the LangGraph
> adapter, runnable examples, and a full test suite are in place; the optional
> KV-cache plugin is next.

## Quick example

```python
from agentpause import PredictiveScheduler

sched = PredictiveScheduler(backend=my_llm_call, telemetry=my_rate_limit_reader)

with sched.session("task-1") as s:      # resumes automatically if interrupted before
    for question in questions[s.step:]:  # skip steps finished before a suspend
        s.add_user(question)
        if s.should_suspend():           # predictive check, *before* the call
            s.checkpoint()
            break                        # stop cleanly; rerun to resume
        reply = s.call()
    else:
        s.complete()                     # task done: drop the checkpoint
```

`backend` is any callable `messages -> (reply, tokens_used)`; `telemetry` is any
callable `() -> remaining_tokens`. See `examples/quickstart.py` for a runnable demo
(no API keys needed) that suspends mid-task and resumes on the next run.

## Real providers via LiteLLM

The LiteLLM adapter supplies both callables for 100+ providers (OpenAI, Groq,
Anthropic, local servers, ...), reading the budget from each response's
rate-limit headers and refreshing stale readings with a tiny telemetry ping:

```python
from agentpause import PredictiveScheduler
from agentpause.adapters.litellm import LiteLLMAdapter

adapter = LiteLLMAdapter(model="groq/llama-3.1-8b-instant")
sched = PredictiveScheduler(backend=adapter.backend, telemetry=adapter.telemetry)
```

Install with `pip install -e ".[litellm]"`; see `examples/litellm_groq.py`.
To validate against your own provider (frontier models included):
`python scripts/validate_provider.py gpt-4o-mini`.

Rate-limit headers by provider (defaults target the OpenAI-style names):

| Provider | remaining tokens | remaining requests | reset |
|---|---|---|---|
| OpenAI | `x-ratelimit-remaining-tokens` | `x-ratelimit-remaining-requests` | `x-ratelimit-reset-tokens` (`1s`, `6m0s`) |
| Groq | `x-ratelimit-remaining-tokens` | `x-ratelimit-remaining-requests` | `x-ratelimit-reset-tokens` (`7.66s`, `2m59.56s`) |
| Anthropic | `anthropic-ratelimit-tokens-remaining` | `anthropic-ratelimit-requests-remaining` | RFC 3339 timestamp — set `remaining_header=` etc. |

Header names differ? Override them: `LiteLLMAdapter(model=..., remaining_header="...", requests_header="...", reset_header="...")`.

## Beyond tokens: RPM and wait-vs-suspend

Telemetry can be richer than a token count. `adapter.budget` reports all
three dimensions providers expose — remaining tokens (TPM), remaining
requests (RPM), and seconds until the window resets — and unlocks the
three-valued decision:

```python
sched = PredictiveScheduler(backend=adapter.backend, telemetry=adapter.budget)

d = session.next_action()   # "continue" | "wait" | "checkpoint"
```

`wait` fires when the budget does not fit **but** the window resets within
`wait_threshold_s` (default 15 s): a short in-place pause is cheaper than a
full suspend/resume cycle. Exhausted requests (RPM = 0) block the call even
with plenty of tokens left. Plain-int telemetry keeps working unchanged.

## Async

Every entry point has an async twin — same rules, same guarantees, never
blocks the event loop:

```python
adapter = LiteLLMAdapter(model="groq/llama-3.1-8b-instant")
sched = PredictiveScheduler(backend=None, async_backend=adapter.abackend,
                            telemetry=adapter.telemetry)
reply = await session.acall()          # retry/backoff via asyncio.sleep
await guard.acheck(state["messages"])  # async LangGraph nodes
```

For sharper input estimates, wire the per-model tokenizer:
`PredictiveScheduler(..., count_tokens=adapter.count_tokens)` (falls back to
the ~4 chars/token heuristic if the tokenizer is unavailable).

## When prediction fails anyway

Estimates are statistical — a 429 can still slip through. agentpause survives
it instead of crashing:

- **typed errors**: everything derives from `AgentPauseError`
  (`RateLimitHit`, `TelemetryError`, `CheckpointError`, `BackendError`);
- **retry with backoff**: unexpected 429s are retried (provider `retry-after`
  honored, else exponential backoff — see `RetryPolicy`);
- **hits are feedback**: each one bumps `safety_k` up (capped at `k_max`),
  so the scheduler grows more cautious on workloads it underestimates;
- **clean failure**: a failed call leaves the session state untouched —
  no phantom steps, resumes stay consistent.

## LangGraph integration

`AgentPauseGuard` adds the predictive gate to any LangGraph node — LangGraph
persists reactively (after nodes), the guard pauses *before* the call that
would hit the rate limit, via LangGraph's own `interrupt()` + checkpointer:

```python
from agentpause.adapters.langgraph import AgentPauseGuard

guard = AgentPauseGuard(telemetry=adapter.telemetry)   # e.g. from LiteLLMAdapter

def agent_node(state):
    guard.check(state["messages"])       # pauses the graph here if needed
    reply = llm.invoke(state["messages"])
    guard.record(state["messages"], reply.usage_metadata["total_tokens"])
    ...
```

Resume the paused thread with `graph.invoke(Command(resume=True), config)`.
On resume the guard re-reads telemetry fresh — never from the checkpoint.
Install with `pip install -e ".[langgraph]"`; see `examples/langgraph_quickstart.py`.

## Why

Current agent frameworks persist state *reactively*: they checkpoint after a step
completes and crash when the provider returns HTTP 429. `agentpause` adds a
*predictive* layer that estimates the next step's cost, compares it against the
remaining rate-limit budget, and suspends cleanly before the error — then resumes
from the exact step.

This library is the engineering counterpart of the research preprint *"A
Resource-Aware Predictive Scheduler for Autonomous LLM Agents"*.

## Components (v0.1)

| Module | Role |
|--------|------|
| `PredictiveScheduler` / `Session` | the high-level API: `session()`, `should_suspend()`, `call()`, `checkpoint()` |
| `Estimator` | predicts next-step token cost with a moving-average error correction (ε) and tracks σ |
| `should_checkpoint` / `RiskModel` | the suspension rule (`remaining < estimated + k·σ`) and a diagnostic risk score |
| `Budget` / `decide` | three-dimensional telemetry (TPM, RPM, reset time) and the three-valued rule: continue / wait / checkpoint |
| `StateStore` / `Checkpoint` | atomic logical checkpointing with idempotency keys — works on any provider |
| `adapters.litellm.LiteLLMAdapter` | backend + telemetry for any LiteLLM-supported provider (headers → budget, stale reading → 1-token ping) |
| `adapters.langgraph.AgentPauseGuard` | predictive gate for LangGraph nodes: `check()` interrupts the graph before the fatal call, `record()` trains the estimator |

## Install (from source, during development)

```bash
git clone https://github.com/<user>/agentpause
cd agentpause
pip install -e ".[dev]"
pytest
```

## Roadmap

- [x] Core components + test suite
- [x] `PredictiveScheduler` high-level API (`session()`, `should_suspend()`, `call()`)
- [x] Runnable quickstart example (no keys)
- [x] LiteLLM adapter (works with any provider)
- [x] LangGraph adapter (interrupt + checkpointer)
- [ ] Optional KV-cache plugin for llama.cpp / vLLM (true warm start)

## License

MIT
