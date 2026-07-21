"""Offline tests for LlamaCppContextBudget and GPUMemoryBudget
(agentpause.adapters.local_resources).

Entirely offline: LlamaCppSlots is built with a fake get_fn returning fixed
Python dicts standing in for GET /props and GET /slots responses, and
GPUMemoryBudget is exercised with a fake reader_fn returning fixed
(free_bytes, total_bytes) tuples -- no real HTTP, no httpx, no pynvml, no
GPU or llama.cpp server involved anywhere in this file.
"""

from __future__ import annotations

import pytest

from agentpause.adapters.local_resources import GPUMemoryBudget, LlamaCppContextBudget
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


# ==============================================================================
# GPUMemoryBudget -- entirely offline via a fake reader_fn (no pynvml, no GPU)
# ==============================================================================

class TestGPUMemoryBudget:
    """Offline coverage for GPUMemoryBudget: normal case, the bytes_per_token=None
    design decision (fails loudly rather than guessing), reader failures becoming
    TelemetryError, and near-zero free VRAM."""

    # -- 1. normal case: bytes_per_token set, plenty of headroom -----------------

    def test_normal_case_computes_correct_remaining_tokens(self):
        def fake_reader(device_index):
            assert device_index == 0
            return (8_000_000_000, 24_000_000_000)  # 8 GB free of 24 GB total

        budget_fn = GPUMemoryBudget(
            device_index=0, bytes_per_token=100_000.0, reader_fn=fake_reader,
        )
        budget = budget_fn()
        assert budget.remaining_tokens == 80_000       # 8e9 / 1e5
        assert budget.limit_tokens == 240_000           # 24e9 / 1e5

    def test_safety_margin_bytes_subtracted(self):
        def fake_reader(device_index):
            return (1_000_000, 10_000_000)

        budget_fn = GPUMemoryBudget(
            bytes_per_token=1000.0,
            safety_margin_bytes=200_000,
            reader_fn=fake_reader,
        )
        budget = budget_fn()
        assert budget.remaining_tokens == 800  # (1_000_000 - 200_000) / 1000

    def test_device_index_is_passed_through_to_reader(self):
        seen = {}

        def fake_reader(device_index):
            seen["device_index"] = device_index
            return (5_000, 5_000)

        GPUMemoryBudget(device_index=3, bytes_per_token=1.0, reader_fn=fake_reader)()
        assert seen["device_index"] == 3

    # -- 2. bytes_per_token=None: documented design choice ------------------------
    # Preferred option (a) from the design brief: fail loudly with a clear
    # TelemetryError rather than inventing a conservative-looking token count
    # from a byte reading alone -- consistent with how the rest of the library
    # treats missing conversion information (see TelemetryError's own docstring:
    # "the budget could not be read").

    def test_bytes_per_token_none_raises_telemetry_error_not_a_guess(self):
        def fake_reader(device_index):
            return (8_000_000_000, 24_000_000_000)

        budget_fn = GPUMemoryBudget(bytes_per_token=None, reader_fn=fake_reader)
        with pytest.raises(TelemetryError, match="bytes_per_token"):
            budget_fn()

    # -- 3. reader failure (driver missing / NVML failed / anything) -------------

    def test_reader_exception_becomes_telemetry_error(self):
        def fake_reader(device_index):
            raise RuntimeError("NVML_ERROR_LIBRARY_NOT_FOUND")

        budget_fn = GPUMemoryBudget(bytes_per_token=1000.0, reader_fn=fake_reader)
        with pytest.raises(TelemetryError):
            budget_fn()

    def test_reader_gpu_error_also_becomes_telemetry_error(self):
        from agentpause.errors import GPUError

        def fake_reader(device_index):
            raise GPUError("pynvml is not installed")

        budget_fn = GPUMemoryBudget(bytes_per_token=1000.0, reader_fn=fake_reader)
        with pytest.raises(TelemetryError):
            budget_fn()

    # -- 4. free_bytes near zero: floors at 0 tokens, never negative -------------

    def test_free_bytes_near_zero_floors_to_zero_tokens(self):
        def fake_reader(device_index):
            return (50, 24_000_000_000)  # essentially no free VRAM left

        budget_fn = GPUMemoryBudget(
            bytes_per_token=1000.0, safety_margin_bytes=1000, reader_fn=fake_reader,
        )
        budget = budget_fn()
        assert budget.remaining_tokens == 0

    def test_free_bytes_exactly_zero(self):
        def fake_reader(device_index):
            return (0, 24_000_000_000)

        budget_fn = GPUMemoryBudget(bytes_per_token=1000.0, reader_fn=fake_reader)
        budget = budget_fn()
        assert budget.remaining_tokens == 0
