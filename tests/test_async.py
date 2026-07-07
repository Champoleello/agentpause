"""Behavioral contract for async support and the real tokenizer.

Modern agents are mostly ``asyncio``-based. The async path mirrors the sync
one exactly: same decision rule, same retry semantics, same state guarantees —
just awaitable. Telemetry freshness rules are unchanged.

The tokenizer contract: when litellm is available the adapter can supply a
per-model ``count_tokens``; on any failure it falls back to the ~4 chars/token
heuristic instead of crashing.
"""

import asyncio

import pytest

from agentpause import PredictiveScheduler, RateLimitHit, RetryPolicy, StateStore
from agentpause.adapters.langgraph import AgentPauseGuard
from agentpause.adapters.litellm import LiteLLMAdapter

from tests.test_litellm_adapter import make_response


def run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


# -- Session.acall() ----------------------------------------------------------

def make_async_backend(reply="ok", tokens=300):
    async def backend(messages):
        return reply, tokens
    return backend


def test_acall_records_and_advances(tmp_path):
    sched = PredictiveScheduler(
        backend=None,
        async_backend=make_async_backend(tokens=250),
        telemetry=lambda: 100_000,
        store=StateStore(str(tmp_path)),
    )

    async def main():
        with sched.session("t") as s:
            s.add_user("q")
            assert s.should_suspend() is False
            reply = await s.acall()
            assert reply == "ok"
            assert s.step == 1
            assert s.total_tokens_used == 250

    run(main())


def test_acall_retries_with_async_sleep(tmp_path):
    calls = {"n": 0}
    sleeps = []

    async def flaky(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RateLimitHit(retry_after=3.0)
        return "ok", 100

    async def fake_sleep(s):
        sleeps.append(s)

    sched = PredictiveScheduler(
        backend=None,
        async_backend=flaky,
        telemetry=lambda: 100_000,
        store=StateStore(str(tmp_path)),
        retry=RetryPolicy(async_sleep_fn=fake_sleep),
    )

    async def main():
        with sched.session("t") as s:
            s.add_user("q")
            assert await s.acall() == "ok"

    run(main())
    assert sleeps == [3.0]
    assert sched.rate_limit_hits == 1


def test_acall_gives_up_cleanly(tmp_path):
    async def always_429(messages):
        raise RateLimitHit()

    async def fake_sleep(s):
        pass

    sched = PredictiveScheduler(
        backend=None,
        async_backend=always_429,
        telemetry=lambda: 100_000,
        store=StateStore(str(tmp_path)),
        retry=RetryPolicy(max_retries=1, async_sleep_fn=fake_sleep),
    )

    async def main():
        with sched.session("t") as s:
            s.add_user("q")
            with pytest.raises(RateLimitHit):
                await s.acall()
            assert s.step == 0
            assert s.total_tokens_used == 0

    run(main())


def test_acall_without_async_backend_raises(tmp_path):
    sched = PredictiveScheduler(
        backend=make_async_backend(),  # sync slot filled, async slot empty
        telemetry=lambda: 100_000,
        store=StateStore(str(tmp_path)),
    )

    async def main():
        with sched.session("t") as s:
            s.add_user("q")
            with pytest.raises(RuntimeError):
                await s.acall()

    run(main())


# -- async LiteLLM adapter -------------------------------------------------------

HEADERS = {"x-ratelimit-remaining-tokens": "5000"}


def test_adapter_abackend_and_atelemetry():
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return make_response(reply="hi", total_tokens=42, headers=HEADERS)

    adapter = LiteLLMAdapter(model="m", acompletion_fn=fake_acompletion)

    async def main():
        reply, used = await adapter.abackend([{"role": "user", "content": "x"}])
        assert (reply, used) == ("hi", 42)
        assert await adapter.atelemetry() == 5000   # fresh from the same response
        assert len(calls) == 1                       # no extra ping needed

    run(main())


def test_adapter_aping_when_stale():
    async def fake_acompletion(**kwargs):
        return make_response(headers=HEADERS)

    adapter = LiteLLMAdapter(model="m", acompletion_fn=fake_acompletion)

    async def main():
        b = await adapter.abudget()                  # nothing cached → ping
        assert b.remaining_tokens == 5000

    run(main())


# -- async guard -------------------------------------------------------------------

def test_guard_acheck_waits_then_proceeds():
    from agentpause import Budget

    budget = Budget(remaining_tokens=100, reset_seconds=2.0)
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)
        budget.remaining_tokens = 100_000

    guard = AgentPauseGuard(telemetry=lambda: budget,
                            interrupt_fn=lambda p: None,
                            async_sleep_fn=fake_sleep)

    run(guard.acheck([{"role": "user", "content": "q " * 50}]))
    assert len(sleeps) == 1


# -- real tokenizer with fallback ---------------------------------------------------

def test_count_tokens_uses_injected_counter():
    adapter = LiteLLMAdapter(model="m", completion_fn=lambda **k: None,
                             token_counter_fn=lambda model, text: 7)
    assert adapter.count_tokens("whatever text") == 7


def test_count_tokens_falls_back_on_failure():
    def broken(model, text):
        raise RuntimeError("no tokenizer for this model")

    adapter = LiteLLMAdapter(model="m", completion_fn=lambda **k: None,
                             token_counter_fn=broken)
    assert adapter.count_tokens("x" * 400) == 100    # ~4 chars/token heuristic
