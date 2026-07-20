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
refill-aware chunked waiting, leveled windows):

|  | reactive baseline | agentpause |
|---|---|---|
| 429 errors suffered | 7 | **0** |
| steps redone | 7 | **0** |
| tokens re-sent (waste) | 12,840 | **0** |
| telemetry overhead (pings) | 0 | 259 |
| wall-clock | 82 s | 86 s |

Zero errors and zero waste at wall-clock parity: the sensor costs ~2% of
what crashes waste (259 vs 12,840 tokens) — and waste is money on paid tiers.

**Context slimming vs. answer quality** (`python scripts/quality_ab.py`,
live 2026-07-10, Groq `llama-3.1-8b-instant`): six facts planted early in a
verbose conversation, same final quiz under three histories.

| condition | prompt chars | facts recalled |
|---|---|---|
| A — full history | 8,984 | **6/6** |
| B — `compact()` (blind truncation) | 4,370 | **0/6** — and the model *invented* plausible replacements |
| C — `summarize_with()` (one summary call) | 3,608 | **6/6** |

A 2.2× larger run (`--big`, 19,145 chars — near the free tier's whole TPM
window, the physical ceiling for a single call) repeated the pattern:
A 6/6, B 0/6, C 6/6 at a third of the prompt.

The instructive failure is B's, and it is *erratic*: in one run it answered
confidently with fabricated values (fake codename, fake budget, fake city);
in the larger run it honestly declined. You cannot know in advance which
failure you get — and the hallucinating one is the dangerous one. Blind
truncation is an emergency exit (§8.6 overflow, no LLM available), not a
strategy; semantic summarization recalled everything at a fraction of the
prompt, earning its one extra call.
A 20-step live stress test (2026-07-10) closed the loop: at the §8.6 wall the
scheduler suspended, compacted the checkpoint offline, resumed slim, and
completed 20/20 steps with zero 429s.

**Facts vs. voice** (`python scripts/quality_ab.py`, live 2026-07-20,
Groq `llama-3.1-8b-instant` — the voice probe rides along on the same run,
no extra flag needed): does shrinking a checkpoint also flatten the
*style* it was written in, not just the facts inside it? A persona with two
short verbal tics is planted in the same conversation as the six facts above:

| condition | facts recalled | voice tics preserved |
|---|---|---|
| A — full history | **6/6** | 1/2 |
| B — `compact()` (blind truncation) | 0/6 | 1/2 |
| C — `summarize_with()` (one summary call) | **6/6** | **0/2** |

Truncation and summarization fail in opposite, complementary ways: blind
truncation keeps short, verbatim phrases intact (only long messages get cut)
but loses facts scattered earlier in the history; summarization recovers the
facts but rewrites everything in its own voice, so a persona's way of talking
does not survive being summarized even when what it *said* does. If tone and
persona consistency matter for your agent, `compact()` and `summarize_with()`
are not interchangeable — pick per what you can't afford to lose.

> Status: **0.4.0**. Core scheduler, direct + LiteLLM + Anthropic adapters,
> LangGraph integration, multi-provider routing, multi-agent shared budgets,
> feature-based cost/latency estimation, task-completion forecast, checkpoint
> fork & cross-machine migration, an optional true-warm-start KV-cache plugin
> for llama.cpp, human attention as a budget, runnable examples, and a full
> test suite are all in place.

## Install

```bash
pip install agentpause
```

Optional extras, only if you use them (the core has zero dependencies):

```bash
pip install "agentpause[litellm]"    # LiteLLM adapter (100+ providers)
pip install "agentpause[langgraph]"  # LangGraph adapter
```

Developing on the library itself (clone + editable install + tests):

```bash
git clone https://github.com/Champoleello/agentpause
cd agentpause
pip install -e ".[dev]"
pytest
```

## Getting started

**Step 1 — the smallest possible loop.** `backend` is any callable
`messages -> (reply, tokens_used)`; `telemetry` is any callable
`() -> remaining_tokens`. That's the entire contract — no framework required:

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

Run it now, no API key needed: `python examples/quickstart.py` — it suspends
mid-task on the first run and resumes cleanly the second time you run it.

**Step 2 — the same three calls, wired into YOUR agent loop.** Whatever
framework you use (a hand-rolled ReAct loop, LangGraph, CrewAI, a plain
`while` loop), the integration is always the same three calls in the same
three places:

1. Before every LLM call, ask `should_suspend()` (or `session.next_action()`
   for the richer `continue`/`wait`/`checkpoint` answer). This is a local
   computation — no network call, no cost.
2. If it says stop, call `checkpoint()` and exit the process. Nothing else
   to do: state, messages, and idempotency keys are already on disk.
3. Re-running the same code with the same session id resumes automatically
   from the exact step — `with sched.session(...)` handles the restore.

**Step 3 — pick the adapter for your provider**, which supplies `backend` and
`telemetry` for you instead of writing them by hand:

| You use... | Adapter | Runnable example |
|---|---|---|
| LiteLLM (100+ providers) | `adapters.litellm.LiteLLMAdapter` | `examples/litellm_groq.py` |
| Direct HTTP, OpenAI-compatible (Groq, OpenAI, ...) | `adapters.openai_compat.OpenAICompatAdapter` | `scripts/validate_provider.py` |
| Direct Anthropic Messages API | `adapters.anthropic.AnthropicAdapter` | see "Real providers" below |
| LangGraph | `adapters.langgraph.AgentPauseGuard` | `examples/langgraph_quickstart.py` |
| A CrewAI/AutoGen/custom loop, no adapter yet | none — write `backend`/`telemetry` by hand as in Step 1 | `examples/quickstart.py` |

Details and full code for each adapter are in the sections right below; the
`## Components` table further down is the complete function/class reference
for everything the library exposes.

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

### Self-hosted: ollama-gateway

Running Ollama locally behind
[martinobettucci/ollama-gateway](https://github.com/martinobettucci/ollama-gateway)
(a self-hosted auth + quota proxy for Ollama)? It emits the same
OpenAI/Groq-style `x-ratelimit-*` headers agentpause already reads, so it
works out of the box through the direct adapter — no litellm, no
special-casing. Confirmed live end-to-end (2026-07-20) against a real
gateway instance: `adapter.budget()` correctly reads `remaining_tokens`,
`remaining_requests` (decrementing per call), and `reset_seconds` — **the
one condition is that the API key has a quota configured** (token cap and/or
rate limit); a key with no quota set simply emits no rate-limit headers at
all, which agentpause reports as a clear `TelemetryError` rather than a
silent wrong answer:

```python
from agentpause.adapters.openai_compat import OpenAICompatAdapter

adapter = OpenAICompatAdapter.for_model("ollama-gateway/llama3:8b")
# default base_url is http://127.0.0.1:8787/v1 (dev default); override for
# a custom port or a production HTTPS domain:
# OpenAICompatAdapter.for_model("ollama-gateway/llama3:8b",
#                                base_url="https://your-gateway-domain")
```

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

## Scaling up: many providers, many agents, smarter estimates (v0.3)

The same predictive idea — read the budget first, act before the error —
extends in three directions, and they compose:

```python
from agentpause import BudgetRouter, MultiAgentCoordinator, FeatureEstimator
from agentpause.adapters.openai_compat import OpenAICompatAdapter
from agentpause.adapters.anthropic import AnthropicAdapter

# 1. Route each call to the provider with the most headroom (predictive
#    fallback: switch BEFORE the 429, not after). Providers in cooldown
#    after a real 429 are skipped until their window resets.
router = BudgetRouter(
    ("groq",   OpenAICompatAdapter.for_model("groq/llama-3.1-8b-instant")),
    ("claude", AnthropicAdapter("claude-haiku-4-5")),
)

# 2. Share ONE rate-limit window across a fleet: every granted call
#    reserves its predicted cost (estimate + k·σ) from the shared pool,
#    so agents can't overcommit the window together. Contention is
#    arbitrated by priority, then longest-waiting.
coord = MultiAgentCoordinator(telemetry=router.budget)
coord.register("researcher", priority=1)
coord.register("summarizer")

est = FeatureEstimator()                     # 3. learn cost from features,
est.set_context(tool="web_search")           #    not just context size

d = coord.request("researcher", estimated=est.estimate(1200),
                  sigma=est.sigma(fallback_estimate=1200))
if d.action == "continue":
    reply, used = router.backend(messages)   # router picks the provider
    coord.complete("researcher", actual_tokens=used)
    est.record(1200, used)
```

`FeatureEstimator` is a drop-in for the default estimator
(`PredictiveScheduler(estimator=FeatureEstimator())`): a dependency-free
ridge regression over features you declare (tool, model, temperature, …)
that also learns per-step **latency**, feeding the optional time budget
(`PredictiveScheduler(time_budget_s=...)`) — if the predicted step can't
finish before the deadline, the answer is checkpoint, never wait (time,
unlike tokens, does not refill).

Runnable demo without any key: `python examples/fleet_quickstart.py`
(routing switch + predictive WAIT under a shared window, in 30 lines of loop).

## Plan before you spend, fork your past, and survive a reboot (v0.4)

Four pieces that treat a suspended checkpoint as what it actually is: inert
data you can inspect, clone, move, or accelerate.

```python
from agentpause import HumanAttentionBudget, StateStore

# 1. FORECAST: before committing to a long run, simulate the REMAINING
#    steps against the live window (no network calls) and get an honest
#    estimate of tokens, money, waits/suspensions and wall-clock time.
forecast = session.forecast(steps_remaining=12)
print(forecast)  # flags context_wall / fits_time_budget if either fails

# 2. FORK: one suspended past, N independent futures. Clones are fully
#    independent (deep-copied messages, own idempotency namespace) but
#    inherit the parent's calibrated estimator (F9.3 symmetry).
store = StateStore(".agentpause")
store.fork("research-task", "research-task-cautious")
store.fork("research-task", "research-task-bold")

# 3. MIGRATE: the checkpoint directory IS the process image. Export it on
#    machine A, ship the bundle over any transport, resume on machine B at
#    the exact step.
bundle = store.export_bundle("research-task")
StateStore("/mnt/machine-b/.agentpause").import_bundle(bundle)

# 4. The human in the loop is a rate-limited resource too: N questions per
#    rolling hour, with a manual override for "I'm away".
attention = HumanAttentionBudget(max_asks=3, window_s=3600)
if not attention.ready():
    session.checkpoint()  # an absent human means suspend, not spin
```

For **self-hosted llama.cpp**, checkpoints can go further than logical state:
`KVStateStore` wraps any `StateStore` and additionally saves/restores the
model's KV-cache via `/slots` — a TRUE warm start (no re-prefill at all, not
just no re-work). It degrades gracefully and automatically to a plain logical
warm start whenever the accelerator can't be trusted: a model fingerprint
mismatch, a missing blob, or a resume after `import_bundle` on a different
machine (KV blobs are intentionally machine-local and never migrate).

```python
from agentpause.llamacpp_kv import LlamaCppSlots, KVStateStore

kv_store = KVStateStore(StateStore(".agentpause"),
                        slots=LlamaCppSlots(), base_url="http://127.0.0.1:8080")
kv_store.save_with_kv(checkpoint)               # KV blob first, logical commit second
cp, info = kv_store.load_with_kv("research-task")
print(info)  # {"kv_restored": True, "n_restored": 4096} — or a graceful
             # {"kv_restored": False, "reason": "model_mismatch" | "kv_file_missing"}
```

Runnable demos without any key or server: `python examples/migrate_fork.py`
(fork + migration story) and `python examples/kv_llamacpp_demo.py` (KV
save/restore, model-change degradation, independent forked blobs, migration).

## Why

Current agent frameworks persist state *reactively*: they checkpoint after a step
completes and crash when the provider returns HTTP 429. `agentpause` adds a
*predictive* layer that estimates the next step's cost, compares it against the
remaining rate-limit budget, and suspends cleanly before the error — then resumes
from the exact step.

This library is the engineering counterpart of the research preprint *"A
Resource-Aware Predictive Scheduler for Autonomous LLM Agents"*.

## Components

| Module | Role |
|--------|------|
| `PredictiveScheduler` / `Session` | the high-level API: `session()`, `should_suspend()`, `call()`, `checkpoint()` |
| `Estimator` | predicts next-step token cost with a moving-average error correction (ε) and tracks σ |
| `FeatureEstimator` | drop-in replacement that learns cost *and latency* from declared features (tool, model, …) via dependency-free ridge regression |
| `should_checkpoint` / `RiskModel` | the suspension rule (`remaining < estimated + k·σ`) and a diagnostic risk score |
| `Budget` / `decide` | multi-dimensional telemetry (TPM, RPM, reset time, deadline) and the three-valued rule: continue / wait / checkpoint |
| `StateStore` / `Checkpoint` | atomic logical checkpointing with idempotency keys — works on any provider; `compact()` / `summarize_with()` shrink a suspended checkpoint offline |
| `BudgetRouter` | predictive multi-provider routing: reads every provider's budget first, routes to the most headroom, cools down 429'd providers |
| `MultiAgentCoordinator` | one shared rate-limit window across many agents: granted calls reserve their predicted cost; `arbitrate()` resolves contention by priority + fairness |
| `ToolQuota` | client-side sliding window for rate-limited tools that expose no headers |
| `CircuitBreaker` / `FallbackBackend` | reactive safety nets: fail fast on a broken provider, try the next one in order |
| `adapters.litellm.LiteLLMAdapter` | backend + telemetry for any LiteLLM-supported provider (headers → budget, stale reading → 1-token ping) |
| `adapters.openai_compat.OpenAICompatAdapter` | direct HTTP adapter for OpenAI-compatible APIs (Groq, OpenAI, …) — no litellm dependency |
| `adapters.anthropic.AnthropicAdapter` | direct adapter for the Anthropic Messages API, with `cache_control` prompt caching and measured `cache_read/write_tokens` |
| `adapters.langgraph.AgentPauseGuard` | predictive gate for LangGraph nodes: `check()` interrupts the graph before the fatal call, `record()` trains the estimator |
| `Session.forecast()` | pure, no-network simulation of the remaining steps: predicted tokens, money, waits/suspensions, wall-clock time; flags `context_wall` and `fits_time_budget` |
| `Checkpoint.fork()` / `StateStore.fork()` | one suspended past, N independent futures — deep-copied state, collision-free idempotency, inherited estimator calibration |
| `StateStore.export_bundle()` / `import_bundle()` | the checkpoint as a versioned, json-portable process image: suspend on one machine, resume on another at the exact step |
| `HumanAttentionBudget` | the person in the loop as a rate-limited `Budget`: N asks per rolling window + manual `available()`/`away(until)` override; composes with `decide()` |
| `llamacpp_kv.LlamaCppSlots` / `KVStateStore` | optional plugin: TRUE warm start for self-hosted llama.cpp via KV-cache save/restore, transactional, with automatic graceful degradation |

## Roadmap

- [x] Core components + test suite
- [x] `PredictiveScheduler` high-level API (`session()`, `should_suspend()`, `call()`)
- [x] Runnable quickstart example (no keys)
- [x] LiteLLM adapter (works with any provider)
- [x] LangGraph adapter (interrupt + checkpointer)
- [x] Direct adapters (OpenAI-compatible, Anthropic with prompt caching)
- [x] Predictive multi-provider routing (`BudgetRouter`)
- [x] Shared budget across agents (`MultiAgentCoordinator`)
- [x] Feature-based cost & latency estimator (`FeatureEstimator`)
- [x] Task-completion forecast (`session.forecast()`)
- [x] Checkpoint fork & cross-machine migration
- [x] Optional KV-cache plugin for llama.cpp (true warm start), incl. fork+KV
- [x] Human attention as a rate-limited budget (`HumanAttentionBudget`)
- [ ] CrewAI / AutoGen / LlamaIndex adapters
- [ ] KV-cache plugin for vLLM

## License

MIT
