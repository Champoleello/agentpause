"""Offline tests for LlamaCppContextBudget (agentpause.adapters.local_resources).

Entirely offline: LlamaCppSlots is built with a fake get_fn returning fixed
Python dicts standing in for GET /props and GET /slots responses -- no real
HTTP, no httpx, no llama.cpp server involved.
"""

from __future__ import annotations

import pytest

from agentpause.adapters.local_resources import LlamaCppContextBudget
from agentpause.errors import TelemetryError
from agentpause.llamacpp_kv import LlamaCppSlots


def make_slots(props, slots_list):
    """A real LlamaCppSlots wired to a fake, in-memory transport."""

    def fake_get(url):
        if url.endswith("/props"):
            return props
        if url.endswith("/slots"):
            return slots_list
        raise AssertionError(f"unexpected url: {url}")

    return LlamaCppSlots(get_fn=fake_get)


# -- 1. normal case: plenty of headroom left ----------------------------------

def test_normal_case_computes_correct_remaining():
    slots = make_slots(
        props={"default_generation_settings": {"n_ctx": 4096}},
        slots_list=[{"id": 0, "cache_tokens": list(range(1000))}],
    )
    budget = LlamaCppContextBudget(slots, base_url="http://fake:8080", id_slot=0)()
    assert budget.remaining_tokens == 3096
    assert budget.limit_tokens == 4096


# -- 2. nearly full: remaining close to zero, but still correct ---------------

def test_nearly_full_context():
    slots = make_slots(
        props={"default_generation_settings": {"n_ctx": 4096}},
        slots_list=[{"id": 0, "cache_tokens": list(range(4090))}],
    )
    budget = LlamaCppContextBudget(slots, base_url="http://fake:8080", id_slot=0)()
    assert budget.remaining_tokens == 6
    assert budget.limit_tokens == 4096


# -- 3. used > context_size: clamps to 0, never negative ----------------------

def test_used_greater_than_context_clamps_to_zero():
    slots = make_slots(
        props={"default_generation_settings": {"n_ctx": 2048}},
        slots_list=[{"id": 0, "cache_tokens": list(range(5000))}],
    )
    budget = LlamaCppContextBudget(slots, base_url="http://fake:8080", id_slot=0)()
    assert budget.remaining_tokens == 0
    assert budget.limit_tokens == 2048


# -- 4. safety margin is subtracted, and still floors at 0 --------------------

def test_safety_margin_subtracted():
    slots = make_slots(
        props={"default_generation_settings": {"n_ctx": 4096}},
        slots_list=[{"id": 0, "cache_tokens": list(range(3000))}],
    )
    budget = LlamaCppContextBudget(
        slots, base_url="http://fake:8080", id_slot=0, safety_margin_tokens=500,
    )()
    assert budget.remaining_tokens == 596  # 4096 - 3000 - 500


def test_safety_margin_can_also_floor_to_zero():
    slots = make_slots(
        props={"default_generation_settings": {"n_ctx": 4096}},
        slots_list=[{"id": 0, "cache_tokens": list(range(3900))}],
    )
    budget = LlamaCppContextBudget(
        slots, base_url="http://fake:8080", id_slot=0, safety_margin_tokens=500,
    )()
    assert budget.remaining_tokens == 0


# -- 5. server unreachable: TelemetryError, never a raw KVError ---------------

def test_unreachable_server_raises_telemetry_error_not_kverror():
    def fake_get(url):
        raise ConnectionError("connection refused")

    slots = LlamaCppSlots(get_fn=fake_get)
    budget_fn = LlamaCppContextBudget(slots, base_url="http://fake:8080", id_slot=0)
    with pytest.raises(TelemetryError):
        budget_fn()


# -- 6. slot does not exist: TelemetryError -----------------------------------

def test_missing_slot_raises_telemetry_error():
    slots = make_slots(
        props={"default_generation_settings": {"n_ctx": 4096}},
        slots_list=[{"id": 0, "cache_tokens": []}],
    )
    budget_fn = LlamaCppContextBudget(slots, base_url="http://fake:8080", id_slot=7)
    with pytest.raises(TelemetryError):
        budget_fn()


# -- 7. custom context_field / used_field callables are honored ---------------

def test_custom_context_and_used_field_callables():
    slots = make_slots(
        props={"weird_shape": {"ctx_total": 8192}},
        slots_list=[{"id": 0, "my_used_count": 1234}],
    )
    budget = LlamaCppContextBudget(
        slots,
        base_url="http://fake:8080",
        id_slot=0,
        context_field=lambda p: p["weird_shape"]["ctx_total"],
        used_field=lambda s: s["my_used_count"],
    )()
    assert budget.limit_tokens == 8192
    assert budget.remaining_tokens == 8192 - 1234


# -- 8. default field extraction: fallback candidates --------------------------

def test_default_used_field_falls_back_to_n_past():
    slots = make_slots(
        props={"default_generation_settings": {"n_ctx": 4096}},
        slots_list=[{"id": 0, "n_past": 2048}],
    )
    budget = LlamaCppContextBudget(slots, base_url="http://fake:8080", id_slot=0)()
    assert budget.remaining_tokens == 2048


def test_default_used_field_falls_back_to_next_token_n_decoded():
    slots = make_slots(
        props={"default_generation_settings": {"n_ctx": 4096}},
        slots_list=[{"id": 0, "next_token": {"n_decoded": 100}}],
    )
    budget = LlamaCppContextBudget(slots, base_url="http://fake:8080", id_slot=0)()
    assert budget.remaining_tokens == 3996


def test_default_context_field_falls_back_to_top_level_n_ctx():
    slots = make_slots(
        props={"n_ctx": 8192},
        slots_list=[{"id": 0, "cache_tokens": [1, 2, 3]}],
    )
    budget = LlamaCppContextBudget(slots, base_url="http://fake:8080", id_slot=0)()
    assert budget.limit_tokens == 8192
    assert budget.remaining_tokens == 8189


def test_no_recognized_field_raises_telemetry_error():
    slots = make_slots(
        props={"default_generation_settings": {"n_ctx": 4096}},
        slots_list=[{"id": 0}],  # no cache_tokens/n_past/prompt/tokens/next_token
    )
    budget_fn = LlamaCppContextBudget(slots, base_url="http://fake:8080", id_slot=0)
    with pytest.raises(TelemetryError):
        budget_fn()
