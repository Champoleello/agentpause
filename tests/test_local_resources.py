"""Offline tests for LlamaCppContextBudget, GPUMemoryBudget, and
LocalResourceBudget (agentpause.adapters.local_resources).

Entirely offline: LlamaCppSlots is built with a fake get_fn (and, for
LocalResourceBudget's kv_store() tests, a fake post_fn too) returning fixed
Python values standing in for GET /props, GET /slots, and POST
/slots/{id}?action=save|restore responses, and GPUMemoryBudget /
LocalResourceBudget's GPU detection are exercised with a fake reader_fn
returning fixed (free_bytes, total_bytes) tuples or raising -- no real HTTP,
no httpx, no pynvml, no GPU or llama.cpp server involved anywhere in this
file. LocalResourceBudget's kv_store() tests DO use a real StateStore and
real files under pytest's tmp_path, since kv_store() wraps a real
KVStateStore that writes real bookkeeping/blob files -- only the HTTP
transport underneath LlamaCppSlots is faked.
"""

from __future__ import annotations

import os

import pytest

from agentpause import Checkpoint, StateStore
from agentpause.adapters.local_resources import (
    CompositeLocalBudget,
    GPUMemoryBudget,
    KVAwareTimeBudget,
    LlamaCppContextBudget,
    LocalResourceBudget,
    estimate_hourly_cost_from_power,
    estimate_local_price_per_1k_tokens,
    price_per_1k_tokens_from_estimator,
)
from agentpause.errors import GPUError, TelemetryError
from agentpause.estimator import Estimator
from agentpause.llamacpp_kv import LlamaCppSlots
from agentpause.risk import Budget


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


def test_default_used_field_prefers_n_prompt_tokens():
    """n_prompt_tokens is CONFIRMED live (2026-07-21, real llama-server) as the
    correct field, and must win over the older, unconfirmed fallback guesses
    even when both happen to be present."""
    slots = make_slots(
        props={"default_generation_settings": {"n_ctx": 4096}},
        slots_list=[{"id": 0, "n_prompt_tokens": 290, "cache_tokens": [1, 2, 3]}],
    )
    budget = LlamaCppContextBudget(slots, base_url="http://fake:8080", id_slot=0)()
    assert budget.remaining_tokens == 4096 - 290


def test_default_used_field_next_token_as_list_of_dicts():
    """The real server reports next_token as a LIST of per-attempt dicts, not
    a bare dict -- confirmed live 2026-07-21."""
    slots = make_slots(
        props={"default_generation_settings": {"n_ctx": 4096}},
        slots_list=[{"id": 0, "next_token": [{"has_next_token": False, "n_decoded": 100}]}],
    )
    budget = LlamaCppContextBudget(slots, base_url="http://fake:8080", id_slot=0)()
    assert budget.remaining_tokens == 3996


def test_idle_slot_with_no_task_reports_zero_used():
    """CONFIRMED live: a freshly started server's idle slot is just
    {"id", "n_ctx", "speculative", "is_processing": false} -- no id_task, no
    n_prompt_tokens. 0 tokens used is the correct answer, not a raised error."""
    slots = make_slots(
        props={"default_generation_settings": {"n_ctx": 10240}},
        slots_list=[{"id": 0, "n_ctx": 10240, "speculative": False, "is_processing": False}],
    )
    budget = LlamaCppContextBudget(slots, base_url="http://fake:8080", id_slot=0)()
    assert budget.remaining_tokens == 10240
    assert budget.limit_tokens == 10240


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


# ==============================================================================
# KVAwareTimeBudget -- entirely offline via a fake clock and a fake
# inner_telemetry callable (no real time.monotonic, no network, no GPU).
# ==============================================================================

def make_fake_clock(values):
    """A fake ``clock`` that returns each of ``values`` in order, one per call.

    ``__init__`` consumes the first value (for ``_started_at``); each
    subsequent ``__call__`` on the wrapper consumes one more (for ``elapsed``).
    Never ``time.monotonic`` -- fully deterministic, pre-scripted increasing
    values, exactly as the task requires.
    """
    it = iter(values)

    def clock() -> float:
        return next(it)

    return clock


class TestKVAwareTimeBudget:
    """Offline coverage for KVAwareTimeBudget: reserve precedence
    (estimated_kv_save_s > throughput*blob_bytes > 0.0), the always-overwrite
    behavior of remaining_seconds, negative remaining_seconds when the
    deadline has already passed (left negative, not floored -- see
    risk.decide's time_fit check), and int-vs-Budget normalization identical
    to Session._read_budget."""

    # -- 1. reserve from an explicit estimated_kv_save_s --------------------------

    def test_reserve_from_explicit_estimated_kv_save_s(self):
        clock = make_fake_clock([0.0, 10.0])  # init -> 0.0, __call__ -> 10.0
        inner = lambda: Budget(remaining_tokens=100)

        budget_fn = KVAwareTimeBudget(
            inner_telemetry=inner,
            time_budget_s=100.0,
            estimated_kv_save_s=20.0,
            clock=clock,
        )
        budget = budget_fn()
        # elapsed = 10.0 - 0.0 = 10.0; reserve_s = 20.0
        # remaining_seconds = 100.0 - 10.0 - 20.0 = 70.0
        assert budget_fn.reserve_s == 20.0
        assert budget.remaining_seconds == 70.0
        assert budget.remaining_tokens == 100

    # -- 2. reserve computed from throughput + expected blob size -----------------

    def test_reserve_computed_from_throughput_and_blob_bytes(self):
        clock = make_fake_clock([0.0, 3.0])
        inner = lambda: Budget(remaining_tokens=50)

        budget_fn = KVAwareTimeBudget(
            inner_telemetry=inner,
            time_budget_s=50.0,
            save_throughput_bytes_per_s=1_000_000.0,
            expected_blob_bytes=5_000_000.0,
            clock=clock,
        )
        # reserve_s = 5_000_000 / 1_000_000 = 5.0
        assert budget_fn.reserve_s == 5.0
        budget = budget_fn()
        # elapsed = 3.0; remaining_seconds = 50.0 - 3.0 - 5.0 = 42.0
        assert budget.remaining_seconds == 42.0

    # -- 3. no reserve info given at all: reserve_s defaults to 0.0 ---------------

    def test_no_reserve_info_defaults_to_zero(self):
        clock = make_fake_clock([0.0, 4.0])
        inner = lambda: Budget(remaining_tokens=10)

        budget_fn = KVAwareTimeBudget(
            inner_telemetry=inner,
            time_budget_s=30.0,
            clock=clock,
        )
        assert budget_fn.reserve_s == 0.0
        budget = budget_fn()
        # remaining_seconds = 30.0 - 4.0 - 0.0 = 26.0 -- identical to what
        # PredictiveScheduler.time_budget_s would compute on its own.
        assert budget.remaining_seconds == 26.0

    def test_only_throughput_without_blob_bytes_defaults_to_zero_reserve(self):
        # Only ONE of the two throughput-calc inputs given: not enough to
        # compute a reserve, so it must fall back to 0.0, not raise or guess.
        clock = make_fake_clock([0.0, 1.0])
        budget_fn = KVAwareTimeBudget(
            inner_telemetry=lambda: Budget(remaining_tokens=1),
            time_budget_s=10.0,
            save_throughput_bytes_per_s=1_000_000.0,
            clock=clock,
        )
        assert budget_fn.reserve_s == 0.0

    # -- 4. estimated_kv_save_s takes precedence when BOTH modes are given --------

    def test_estimated_kv_save_s_takes_precedence_over_throughput_calc(self):
        clock = make_fake_clock([0.0, 2.0])
        inner = lambda: Budget(remaining_tokens=1)

        budget_fn = KVAwareTimeBudget(
            inner_telemetry=inner,
            time_budget_s=50.0,
            estimated_kv_save_s=7.0,
            # if this were used instead, reserve_s would be 100.0 -- must NOT win
            save_throughput_bytes_per_s=1_000.0,
            expected_blob_bytes=100_000.0,
            clock=clock,
        )
        assert budget_fn.reserve_s == 7.0
        budget = budget_fn()
        # remaining_seconds = 50.0 - 2.0 - 7.0 = 41.0
        assert budget.remaining_seconds == 41.0

    # -- 5. remaining_seconds goes negative when the deadline has passed ----------
    # Left negative on purpose: risk.decide()'s time_fit check is
    # `estimated_latency <= budget.remaining_seconds`, which already reads a
    # negative remaining_seconds as "does not fit" -> checkpoint, same as any
    # remaining_seconds smaller than estimated_latency. Flooring at 0 here
    # would not change decide()'s outcome and would throw away the (useful
    # for logging) information of how far past the deadline the run is.

    def test_remaining_seconds_is_negative_and_not_floored_when_expired(self):
        clock = make_fake_clock([0.0, 15.0])
        inner = lambda: Budget(remaining_tokens=1)

        budget_fn = KVAwareTimeBudget(
            inner_telemetry=inner,
            time_budget_s=10.0,
            clock=clock,
        )
        budget = budget_fn()
        # elapsed = 15.0; remaining_seconds = 10.0 - 15.0 - 0.0 = -5.0
        assert budget.remaining_seconds == -5.0

    # -- 6. inner_telemetry returning a bare int is normalized like Session does --

    def test_bare_int_inner_telemetry_is_normalized_to_budget(self):
        clock = make_fake_clock([0.0, 5.0])
        inner = lambda: 777  # legacy shape: plain remaining-tokens int

        budget_fn = KVAwareTimeBudget(
            inner_telemetry=inner,
            time_budget_s=20.0,
            clock=clock,
        )
        budget = budget_fn()
        assert isinstance(budget, Budget)
        assert budget.remaining_tokens == 777
        # remaining_seconds = 20.0 - 5.0 - 0.0 = 15.0
        assert budget.remaining_seconds == 15.0

    # -- 7. remaining_seconds is ALWAYS overwritten, even if inner already set it -

    def test_inner_remaining_seconds_is_always_overwritten(self):
        clock = make_fake_clock([0.0, 6.0])
        # the inner telemetry claims 999s remaining on its own -- this wrapper
        # must ignore that and impose its own accounting unconditionally.
        inner = lambda: Budget(remaining_tokens=1, remaining_seconds=999.0)

        budget_fn = KVAwareTimeBudget(
            inner_telemetry=inner,
            time_budget_s=40.0,
            estimated_kv_save_s=5.0,
            clock=clock,
        )
        budget = budget_fn()
        # remaining_seconds = 40.0 - 6.0 - 5.0 = 29.0, NOT 999.0
        assert budget.remaining_seconds == 29.0


# ==============================================================================
# estimate_local_price_per_1k_tokens / estimate_hourly_cost_from_power /
# price_per_1k_tokens_from_estimator -- pure arithmetic, no I/O, no fake
# hardware needed anywhere in this section.
# ==============================================================================

class TestEstimateLocalPricePer1kTokens:
    """estimate_local_price_per_1k_tokens: normal calculation plus the two
    documented ValueError cases (tokens_per_second<=0, hourly_cost<0)."""

    def test_normal_case(self):
        # 10 tokens/s -> 36,000 tokens/hour; $1.00/hour -> $1/36000 per token
        # -> * 1000 = $0.02777... per 1k tokens
        price = estimate_local_price_per_1k_tokens(tokens_per_second=10.0, hourly_cost=1.0)
        assert price == pytest.approx(1.0 / 36.0)

    def test_another_normal_case_matches_hand_computation(self):
        # 50 tokens/s, $2.00/hour rental.
        # tokens_per_hour = 50 * 3600 = 180,000
        # cost_per_token = 2.0 / 180,000
        # price_per_1k = cost_per_token * 1000 = 2.0 * 1000 / 180,000
        price = estimate_local_price_per_1k_tokens(tokens_per_second=50.0, hourly_cost=2.0)
        assert price == pytest.approx(2.0 * 1000.0 / 180_000.0)

    def test_zero_hourly_cost_gives_zero_price(self):
        assert estimate_local_price_per_1k_tokens(tokens_per_second=10.0, hourly_cost=0.0) == 0.0

    def test_tokens_per_second_zero_raises_value_error(self):
        with pytest.raises(ValueError, match="tokens_per_second"):
            estimate_local_price_per_1k_tokens(tokens_per_second=0.0, hourly_cost=1.0)

    def test_tokens_per_second_negative_raises_value_error(self):
        with pytest.raises(ValueError, match="tokens_per_second"):
            estimate_local_price_per_1k_tokens(tokens_per_second=-5.0, hourly_cost=1.0)

    def test_hourly_cost_negative_raises_value_error(self):
        with pytest.raises(ValueError, match="hourly_cost"):
            estimate_local_price_per_1k_tokens(tokens_per_second=10.0, hourly_cost=-1.0)


class TestEstimateHourlyCostFromPower:
    """estimate_hourly_cost_from_power: a hand-verifiable case, plus the two
    documented ValueError cases (watts<0, price_per_kwh<0)."""

    def test_known_case(self):
        # 350 W draw, $0.30/kWh -> 0.35 kW * 0.30 $/kWh = $0.105/hour
        cost = estimate_hourly_cost_from_power(watts=350.0, price_per_kwh=0.30)
        assert cost == pytest.approx(0.105)

    def test_zero_watts_gives_zero_cost(self):
        assert estimate_hourly_cost_from_power(watts=0.0, price_per_kwh=0.30) == 0.0

    def test_watts_negative_raises_value_error(self):
        with pytest.raises(ValueError, match="watts"):
            estimate_hourly_cost_from_power(watts=-10.0, price_per_kwh=0.30)

    def test_price_per_kwh_negative_raises_value_error(self):
        with pytest.raises(ValueError, match="price_per_kwh"):
            estimate_hourly_cost_from_power(watts=350.0, price_per_kwh=-0.1)


class _FakeEstimatorWithLatency:
    """Fake estimator with a valid estimate_latency -- stands in for a
    FeatureEstimator that has recorded enough steps to predict latency."""

    def __init__(self, output_tokens: int, latency_s):
        self._output_tokens = output_tokens
        self._latency_s = latency_s

    def estimate(self, input_tokens: int) -> int:
        return self._output_tokens

    def estimate_latency(self, input_tokens: int):
        return self._latency_s


class _FakeEstimatorNoAttribute:
    """Fake estimator with NO estimate_latency attribute at all -- stands in
    for the base Estimator class, which never implements it."""

    def __init__(self, output_tokens: int):
        self._output_tokens = output_tokens

    def estimate(self, input_tokens: int) -> int:
        return self._output_tokens


class TestPricePer1kTokensFromEstimator:
    """price_per_1k_tokens_from_estimator's three cases: no estimate_latency
    attribute at all (base Estimator), estimate_latency present but
    returning None (not enough history yet), and the normal case where a
    positive latency is available and the combined calculation runs."""

    # -- 1. normal case: estimate_latency returns a valid, positive float ------

    def test_normal_case_combines_estimate_and_latency(self):
        # 100 output tokens predicted in 2.0s -> 50 tokens/s throughput.
        estimator = _FakeEstimatorWithLatency(output_tokens=100, latency_s=2.0)
        price = price_per_1k_tokens_from_estimator(
            estimator, input_tokens=2000, hourly_cost=1.0,
        )
        expected = estimate_local_price_per_1k_tokens(tokens_per_second=50.0, hourly_cost=1.0)
        assert price == pytest.approx(expected)

    # -- 2. estimate_latency exists but returns None (no history yet) ----------

    def test_estimate_latency_returns_none_yields_none(self):
        estimator = _FakeEstimatorWithLatency(output_tokens=100, latency_s=None)
        price = price_per_1k_tokens_from_estimator(
            estimator, input_tokens=2000, hourly_cost=1.0,
        )
        assert price is None

    def test_estimate_latency_returns_zero_yields_none(self):
        # Non-positive latency can't produce an honest throughput either.
        estimator = _FakeEstimatorWithLatency(output_tokens=100, latency_s=0.0)
        price = price_per_1k_tokens_from_estimator(
            estimator, input_tokens=2000, hourly_cost=1.0,
        )
        assert price is None

    # -- 3. estimator has no estimate_latency attribute at all (base Estimator) -

    def test_estimator_without_estimate_latency_attribute_yields_none(self):
        estimator = _FakeEstimatorNoAttribute(output_tokens=100)
        assert not hasattr(estimator, "estimate_latency")
        price = price_per_1k_tokens_from_estimator(
            estimator, input_tokens=2000, hourly_cost=1.0,
        )
        assert price is None

    def test_real_base_estimator_class_also_yields_none(self):
        # Not a fake: the REAL base Estimator, which genuinely has no
        # estimate_latency method (only FeatureEstimator adds one) --
        # confirms the fake above isn't hiding a mismatch with reality.
        estimator = Estimator()
        assert not hasattr(estimator, "estimate_latency")
        price = price_per_1k_tokens_from_estimator(
            estimator, input_tokens=2000, hourly_cost=1.0,
        )
        assert price is None

    # -- 4. invalid hourly_cost still propagates as ValueError ------------------

    def test_invalid_hourly_cost_still_raises_value_error(self):
        estimator = _FakeEstimatorWithLatency(output_tokens=100, latency_s=2.0)
        with pytest.raises(ValueError, match="hourly_cost"):
            price_per_1k_tokens_from_estimator(
                estimator, input_tokens=2000, hourly_cost=-1.0,
            )


# ==============================================================================
# CompositeLocalBudget -- entirely offline via fake telemetry callables
# (plain lambdas returning fixed Budget/int values -- no server, no GPU).
# ==============================================================================

class TestCompositeLocalBudget:
    """CompositeLocalBudget's contract: the most restrictive remaining_tokens
    wins, remaining_seconds/remaining_requests/limit_tokens are merged as the
    minimum across sources that set them, a TelemetryError from any one
    source propagates rather than being swallowed, a single callable is a
    valid (if trivial) composite, and an empty call raises ValueError."""

    # -- 1. minimum of two fake sources wins on remaining_tokens ------------------

    def test_takes_minimum_remaining_tokens_across_two_sources(self):
        low = lambda: Budget(remaining_tokens=100, limit_tokens=4096)
        high = lambda: Budget(remaining_tokens=9000, limit_tokens=100_000)

        composite = CompositeLocalBudget(low, high)
        budget = composite()
        assert budget.remaining_tokens == 100
        # limit_tokens is merged as the MINIMUM across sources, not just
        # carried along from the winning (lowest remaining_tokens) budget
        assert budget.limit_tokens == 4096

    def test_order_of_callables_does_not_matter(self):
        low = lambda: Budget(remaining_tokens=50)
        high = lambda: Budget(remaining_tokens=5000)

        assert CompositeLocalBudget(low, high)().remaining_tokens == 50
        assert CompositeLocalBudget(high, low)().remaining_tokens == 50

    # -- 2. remaining_seconds: minimum of those that set it, else the lone one,
    #        else None -------------------------------------------------------

    def test_remaining_seconds_takes_minimum_when_both_set_it(self):
        a = lambda: Budget(remaining_tokens=100, remaining_seconds=30.0)
        b = lambda: Budget(remaining_tokens=200, remaining_seconds=5.0)

        budget = CompositeLocalBudget(a, b)()
        # a wins on remaining_tokens (100 < 200), but remaining_seconds is
        # still the tightest of the two (5.0), not just a's own 30.0
        assert budget.remaining_tokens == 100
        assert budget.remaining_seconds == 5.0

    def test_remaining_seconds_uses_lone_value_when_only_one_sets_it(self):
        a = lambda: Budget(remaining_tokens=100, remaining_seconds=42.0)
        b = lambda: Budget(remaining_tokens=200)  # no remaining_seconds

        budget = CompositeLocalBudget(a, b)()
        assert budget.remaining_seconds == 42.0

    def test_remaining_seconds_is_none_when_nobody_sets_it(self):
        a = lambda: Budget(remaining_tokens=100)
        b = lambda: Budget(remaining_tokens=200)

        budget = CompositeLocalBudget(a, b)()
        assert budget.remaining_seconds is None

    # -- 3. remaining_requests merged the same way --------------------------------

    def test_remaining_requests_takes_minimum_across_sources(self):
        a = lambda: Budget(remaining_tokens=100, remaining_requests=10)
        b = lambda: Budget(remaining_tokens=50, remaining_requests=2)

        budget = CompositeLocalBudget(a, b)()
        assert budget.remaining_tokens == 50   # b wins on tokens
        assert budget.remaining_requests == 2  # still the tightest overall

    # -- 4. a TelemetryError from any one source propagates -----------------------

    def test_telemetry_error_from_one_source_propagates(self):
        ok = lambda: Budget(remaining_tokens=100)

        def failing():
            raise TelemetryError("GPU unreachable")

        composite = CompositeLocalBudget(ok, failing)
        with pytest.raises(TelemetryError, match="GPU unreachable"):
            composite()

    def test_telemetry_error_propagates_regardless_of_order(self):
        ok = lambda: Budget(remaining_tokens=100)

        def failing():
            raise TelemetryError("context signal unavailable")

        with pytest.raises(TelemetryError):
            CompositeLocalBudget(failing, ok)()

    # -- 5. a single callable is a valid, if trivial, composite -------------------

    def test_single_callable_passes_through_unchanged(self):
        source = lambda: Budget(remaining_tokens=777, remaining_seconds=12.0,
                                limit_tokens=8192)

        budget = CompositeLocalBudget(source)()
        assert budget.remaining_tokens == 777
        assert budget.remaining_seconds == 12.0
        assert budget.limit_tokens == 8192

    def test_single_callable_returning_bare_int_is_normalized(self):
        source = lambda: 555  # legacy shape: plain remaining-tokens int

        budget = CompositeLocalBudget(source)()
        assert isinstance(budget, Budget)
        assert budget.remaining_tokens == 555

    # -- 6. empty call raises a clear ValueError ----------------------------------

    def test_empty_callables_raises_value_error(self):
        with pytest.raises(ValueError, match="at least one telemetry callable"):
            CompositeLocalBudget()

    # -- 7. three real local adapters composed together (integration-ish, but
    #        still fully offline via fakes) -------------------------------------

    def test_three_real_adapters_composed_together(self):
        # LlamaCppContextBudget: plenty of context left (4096 - 500 = 3596)
        slots = make_slots(
            props={"default_generation_settings": {"n_ctx": 4096}},
            slots_list=[{"id": 0, "cache_tokens": list(range(500))}],
        )
        context_budget = LlamaCppContextBudget(slots, base_url="http://fake:8080", id_slot=0)

        # GPUMemoryBudget: much tighter -- only 100 tokens' worth of VRAM free
        def fake_reader(device_index):
            return (100_000, 24_000_000_000)

        gpu_budget = GPUMemoryBudget(bytes_per_token=1000.0, reader_fn=fake_reader)

        composite = CompositeLocalBudget(context_budget, gpu_budget)
        budget = composite()
        # GPU is the binding constraint: 100_000 / 1000 = 100 tokens
        assert budget.remaining_tokens == 100


# ==============================================================================
# LocalResourceBudget -- the plug-and-play facade.
#
# GET /props and /slots are faked via LlamaCppSlots(get_fn=...); the KV
# save/restore POST is faked via LlamaCppSlots(post_fn=...), writing a real
# file of a scripted size under a real tmp_path directory (sparse-file
# trick via seek+write of one byte, so even a "huge" blob size test doesn't
# actually consume that much real disk). kv_store() itself wraps a REAL
# KVStateStore + real StateStore, both operating on tmp_path -- only the
# HTTP transport underneath is fake.
# ==============================================================================

def make_local_slots(n_ctx=10240, model="models/qwen3-8b.gguf",
                     save_dir=None, n_saved=100, blob_bytes=1000):
    """A real LlamaCppSlots wired to a fake, in-memory transport that
    supports GET /props, GET /slots (idle slot -- 0 used), and POST
    /slots/{id}?action=save|restore, writing a real (possibly sparse) blob
    file of exactly ``blob_bytes`` bytes under ``save_dir`` so
    LocalResourceBudget.kv_store()'s calibration can be verified against a
    hand-computable bytes_per_token = blob_bytes / n_saved.
    """
    calls = {"save": 0, "restore": 0}

    def fake_get(url):
        if url.endswith("/props"):
            return {
                "default_generation_settings": {"n_ctx": n_ctx},
                "model_path": model,
            }
        if url.endswith("/slots"):
            return [{"id": 0, "n_ctx": n_ctx, "is_processing": False}]
        raise AssertionError(f"unexpected GET url: {url}")

    def fake_post(url, json):
        filename = json["filename"]
        assert os.sep not in filename, "must receive a BARE filename"
        if "action=save" in url:
            calls["save"] += 1
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, filename)
            with open(path, "wb") as f:
                if blob_bytes > 0:
                    f.seek(blob_bytes - 1)
                    f.write(b"\0")
            return {"n_saved": n_saved}
        if "action=restore" in url:
            calls["restore"] += 1
            return {"n_restored": n_saved}
        raise AssertionError(f"unexpected POST url: {url}")

    slots = LlamaCppSlots(get_fn=fake_get, post_fn=fake_post)
    slots.calls = calls
    return slots


class TestLocalResourceBudgetConstruction:
    """Construction-time behavior: reads n_ctx immediately, detects GPU
    availability without needing bytes_per_token, never crashes either way."""

    def test_reachable_server_reads_n_ctx_correctly(self, tmp_path, capsys):
        slots = make_local_slots(n_ctx=10240, save_dir=str(tmp_path / "kv"))

        LocalResourceBudget(
            kv_dir=str(tmp_path / "kv"), slots=slots,
            gpu_reader_fn=lambda i: (_ for _ in ()).throw(GPUError("no gpu")),
        )

        captured = capsys.readouterr()
        assert "n_ctx=10240" in captured.out

    def test_gpu_reader_raising_sets_gpu_available_false_no_crash(self, tmp_path):
        slots = make_local_slots(save_dir=str(tmp_path / "kv"))

        def failing_reader(device_index):
            raise GPUError("pynvml not installed")

        local = LocalResourceBudget(
            kv_dir=str(tmp_path / "kv"), slots=slots, gpu_reader_fn=failing_reader,
            verbose=False,
        )
        assert local.gpu_available is False

    def test_gpu_reader_raising_generic_exception_also_sets_false(self, tmp_path):
        slots = make_local_slots(save_dir=str(tmp_path / "kv"))

        def failing_reader(device_index):
            raise RuntimeError("NVML_ERROR_UNKNOWN")

        local = LocalResourceBudget(
            kv_dir=str(tmp_path / "kv"), slots=slots, gpu_reader_fn=failing_reader,
            verbose=False,
        )
        assert local.gpu_available is False

    def test_gpu_reader_succeeding_sets_gpu_available_true(self, tmp_path):
        slots = make_local_slots(save_dir=str(tmp_path / "kv"))

        def ok_reader(device_index):
            return (8_000_000_000, 24_000_000_000)

        local = LocalResourceBudget(
            kv_dir=str(tmp_path / "kv"), slots=slots, gpu_reader_fn=ok_reader,
            verbose=False,
        )
        assert local.gpu_available is True

    def test_unreachable_server_propagates_error_immediately(self, tmp_path):
        def fake_get(url):
            raise ConnectionError("connection refused")

        slots = LlamaCppSlots(get_fn=fake_get)
        with pytest.raises(Exception):
            LocalResourceBudget(kv_dir=str(tmp_path / "kv"), slots=slots, verbose=False)


def make_step_clock(start: float = 0.0, step: float = 1.0):
    """A fake monotonic clock returning start, start+step, start+2*step, ...
    on successive calls -- deterministic and hand-verifiable, unlike real
    time.monotonic."""
    state = {"t": start}

    def clock() -> float:
        t = state["t"]
        state["t"] += step
        return t

    return clock


class TestLocalResourceBudgetCall:
    """__call__()'s composition rules: context-only before any GPU signal is
    usable, GPU included only once BOTH available and calibrated, and the
    time-budget wrapper's provisional-vs-measured reserve."""

    def test_call_before_any_save_uses_context_only_even_if_gpu_available(self, tmp_path):
        # n_ctx=4096, idle slot (0 used), safety_margin default 256 ->
        # context-only remaining = 4096 - 0 - 256 = 3840.
        slots = make_local_slots(n_ctx=4096, save_dir=str(tmp_path / "kv"))

        # GPU reports almost no free VRAM -- if GPUMemoryBudget were
        # (incorrectly) composed in already, remaining_tokens would come out
        # far below 3840, or the call would raise (bytes_per_token missing).
        # Neither happens: bytes_per_token is still None (no save yet), so
        # __call__ must fall back to context-only.
        def stingy_reader(device_index):
            return (1, 24_000_000_000)

        local = LocalResourceBudget(
            kv_dir=str(tmp_path / "kv"), slots=slots, gpu_reader_fn=stingy_reader,
            verbose=False,
        )
        assert local.gpu_available is True          # GPU IS available...
        assert local._bytes_per_token is None        # ...but not calibrated yet

        budget = local()
        assert budget.remaining_tokens == 3840
        assert budget.limit_tokens == 4096

    def test_call_after_calibration_includes_gpu_and_gpu_can_bind(self, tmp_path):
        kv_dir = str(tmp_path / "kv")
        slots = make_local_slots(n_ctx=4096, save_dir=kv_dir, n_saved=100, blob_bytes=1000)

        def gpu_reader(device_index):
            # 500 bytes free; bytes_per_token will calibrate to 1000/100=10.0
            # -> 500/10 = 50 tokens, far tighter than the context budget's
            # 4096-0-256=3840, so the GPU axis must be the binding constraint
            # once it's actually composed in.
            return (500, 24_000_000_000)

        local = LocalResourceBudget(
            kv_dir=kv_dir, slots=slots, gpu_reader_fn=gpu_reader, verbose=False,
        )
        kv_store = local.kv_store(StateStore(str(tmp_path / "store")))
        cp = Checkpoint(session_id="mission", messages=[{"role": "user", "content": "hi"}])
        kv_store.save_with_kv(cp)

        assert local._bytes_per_token == pytest.approx(10.0)

        budget = local()
        assert budget.remaining_tokens == 50   # GPU is now the binding constraint

    def test_time_budget_uses_provisional_reserve_before_calibration(self, tmp_path):
        slots = make_local_slots(save_dir=str(tmp_path / "kv"))
        clock = make_step_clock()

        local = LocalResourceBudget(
            kv_dir=str(tmp_path / "kv"), slots=slots, time_budget_s=100.0,
            clock=clock, verbose=False,
            gpu_reader_fn=lambda i: (_ for _ in ()).throw(GPUError("no gpu")),
        )
        budget = local()
        # tick 0.0 -> KVAwareTimeBudget._started_at; tick 1.0 -> elapsed read
        # inside __call__. elapsed = 1.0; reserve = 5.0 (provisional, no real
        # save yet). remaining_seconds = 100.0 - 1.0 - 5.0 = 94.0.
        assert budget.remaining_seconds == pytest.approx(94.0)

    def test_time_budget_uses_measured_reserve_after_calibration(self, tmp_path):
        kv_dir = str(tmp_path / "kv")
        slots = make_local_slots(save_dir=kv_dir, n_saved=100, blob_bytes=1000)
        clock = make_step_clock()

        local = LocalResourceBudget(
            kv_dir=kv_dir, slots=slots, time_budget_s=100.0, clock=clock,
            verbose=False, gpu_reader_fn=lambda i: (_ for _ in ()).throw(GPUError("no gpu")),
        )
        kv_store = local.kv_store(StateStore(str(tmp_path / "store")))
        cp = Checkpoint(session_id="mission", messages=[{"role": "user", "content": "hi"}])
        # tick 0.0 -> save started; tick 1.0 -> save elapsed read.
        # measured_save_s = 1.0 - 0.0 = 1.0.
        kv_store.save_with_kv(cp)
        assert local._measured_save_s == pytest.approx(1.0)

        budget = local()
        # tick 2.0 -> KVAwareTimeBudget._started_at; tick 3.0 -> elapsed read.
        # elapsed = 1.0; reserve = 1.0 (measured, not the 5.0 provisional
        # default). remaining_seconds = 100.0 - 1.0 - 1.0 = 98.0.
        assert budget.remaining_seconds == pytest.approx(98.0)


class TestLocalResourceBudgetKvStoreCalibration:
    """kv_store()'s wrapped save_with_kv: real bytes_per_token/measured_save_s
    calibration from a real (fake-transport) KV save, the auto min_free_bytes
    rule (and that an explicit min_free_bytes is left untouched), and that a
    failed save calibrates nothing at all."""

    def test_bytes_per_token_and_measured_save_s_calibrated_from_real_save(self, tmp_path):
        kv_dir = str(tmp_path / "kv")
        slots = make_local_slots(save_dir=kv_dir, n_saved=100, blob_bytes=4200)
        clock = make_step_clock(start=10.0, step=0.5)

        local = LocalResourceBudget(
            kv_dir=kv_dir, slots=slots, clock=clock, verbose=False,
            gpu_reader_fn=lambda i: (_ for _ in ()).throw(GPUError("no gpu")),
        )
        assert local._bytes_per_token is None
        assert local._measured_save_s is None

        kv_store = local.kv_store(StateStore(str(tmp_path / "store")))
        cp = Checkpoint(session_id="mission", messages=[{"role": "user", "content": "hi"}])
        kv_store.save_with_kv(cp)

        # hand-computed: bytes_per_token = 4200 / 100 = 42.0
        assert local._bytes_per_token == pytest.approx(42.0)
        # hand-computed: elapsed = (10.0 + 0.5) - 10.0 = 0.5
        assert local._measured_save_s == pytest.approx(0.5)

    def test_min_free_bytes_auto_set_above_floor_from_real_blob_size(self, tmp_path):
        kv_dir = str(tmp_path / "kv")
        # 5 * 200_000_000 = 1_000_000_000, comfortably above the 500_000_000
        # floor -- proves the REAL blob size drives the number, not just the
        # floor constant.
        slots = make_local_slots(save_dir=kv_dir, n_saved=1000, blob_bytes=200_000_000)

        local = LocalResourceBudget(
            kv_dir=kv_dir, slots=slots, verbose=False,
            gpu_reader_fn=lambda i: (_ for _ in ()).throw(GPUError("no gpu")),
        )
        assert local.min_free_bytes_override is None

        kv_store = local.kv_store(StateStore(str(tmp_path / "store")))
        cp = Checkpoint(session_id="mission", messages=[{"role": "user", "content": "hi"}])
        kv_store.save_with_kv(cp)

        assert kv_store.min_free_bytes == 1_000_000_000

    def test_min_free_bytes_auto_set_uses_floor_for_a_small_blob(self, tmp_path):
        kv_dir = str(tmp_path / "kv")
        # 5 * 1000 = 5000, far below the 500_000_000 floor -> floor wins.
        slots = make_local_slots(save_dir=kv_dir, n_saved=100, blob_bytes=1000)

        local = LocalResourceBudget(
            kv_dir=kv_dir, slots=slots, verbose=False,
            gpu_reader_fn=lambda i: (_ for _ in ()).throw(GPUError("no gpu")),
        )
        kv_store = local.kv_store(StateStore(str(tmp_path / "store")))
        cp = Checkpoint(session_id="mission", messages=[{"role": "user", "content": "hi"}])
        kv_store.save_with_kv(cp)

        assert kv_store.min_free_bytes == 500_000_000

    def test_explicit_min_free_bytes_is_left_untouched_by_calibration(self, tmp_path):
        kv_dir = str(tmp_path / "kv")
        slots = make_local_slots(save_dir=kv_dir, n_saved=1000, blob_bytes=200_000_000)

        local = LocalResourceBudget(
            kv_dir=kv_dir, slots=slots, min_free_bytes=123_456, verbose=False,
            gpu_reader_fn=lambda i: (_ for _ in ()).throw(GPUError("no gpu")),
        )
        assert local.min_free_bytes_override == 123_456

        kv_store = local.kv_store(StateStore(str(tmp_path / "store")))
        assert kv_store.min_free_bytes == 123_456   # unchanged at construction

        cp = Checkpoint(session_id="mission", messages=[{"role": "user", "content": "hi"}])
        kv_store.save_with_kv(cp)

        # still unchanged after a real, successful calibrating save -- the
        # user's explicit choice is never overwritten.
        assert kv_store.min_free_bytes == 123_456

    def test_failed_save_calibrates_nothing_and_propagates(self, tmp_path):
        kv_dir = str(tmp_path / "kv")
        slots = make_local_slots(save_dir=kv_dir)

        def always_fail(base_url, id_slot, filename):
            from agentpause.errors import KVError
            raise KVError("simulated save failure")

        local = LocalResourceBudget(
            kv_dir=kv_dir, slots=slots, verbose=False,
            gpu_reader_fn=lambda i: (_ for _ in ()).throw(GPUError("no gpu")),
        )
        kv_store = local.kv_store(StateStore(str(tmp_path / "store")))
        kv_store.slots.save = always_fail  # force the underlying save to fail

        from agentpause.errors import KVError
        cp = Checkpoint(session_id="mission", messages=[{"role": "user", "content": "hi"}])
        with pytest.raises(KVError, match="simulated save failure"):
            kv_store.save_with_kv(cp)

        assert local._bytes_per_token is None
        assert local._measured_save_s is None


class TestLocalResourceBudgetSuggestPrice:
    """suggest_price_per_1k_tokens: call-site hourly_cost, constructor
    fallback, and the ValueError when neither is ever provided."""

    def test_hourly_cost_from_call_site(self, tmp_path):
        slots = make_local_slots(save_dir=str(tmp_path / "kv"))
        local = LocalResourceBudget(
            kv_dir=str(tmp_path / "kv"), slots=slots, verbose=False,
            gpu_reader_fn=lambda i: (_ for _ in ()).throw(GPUError("no gpu")),
        )
        estimator = _FakeEstimatorWithLatency(output_tokens=100, latency_s=2.0)

        price = local.suggest_price_per_1k_tokens(estimator, input_tokens=2000, hourly_cost=1.0)
        expected = estimate_local_price_per_1k_tokens(tokens_per_second=50.0, hourly_cost=1.0)
        assert price == pytest.approx(expected)

    def test_hourly_cost_falls_back_to_constructor_default(self, tmp_path):
        slots = make_local_slots(save_dir=str(tmp_path / "kv"))
        local = LocalResourceBudget(
            kv_dir=str(tmp_path / "kv"), slots=slots, hourly_cost=2.0, verbose=False,
            gpu_reader_fn=lambda i: (_ for _ in ()).throw(GPUError("no gpu")),
        )
        estimator = _FakeEstimatorWithLatency(output_tokens=100, latency_s=2.0)

        price = local.suggest_price_per_1k_tokens(estimator, input_tokens=2000)
        expected = estimate_local_price_per_1k_tokens(tokens_per_second=50.0, hourly_cost=2.0)
        assert price == pytest.approx(expected)

    def test_call_site_hourly_cost_overrides_constructor_default(self, tmp_path):
        slots = make_local_slots(save_dir=str(tmp_path / "kv"))
        local = LocalResourceBudget(
            kv_dir=str(tmp_path / "kv"), slots=slots, hourly_cost=2.0, verbose=False,
            gpu_reader_fn=lambda i: (_ for _ in ()).throw(GPUError("no gpu")),
        )
        estimator = _FakeEstimatorWithLatency(output_tokens=100, latency_s=2.0)

        price = local.suggest_price_per_1k_tokens(estimator, input_tokens=2000, hourly_cost=9.0)
        expected = estimate_local_price_per_1k_tokens(tokens_per_second=50.0, hourly_cost=9.0)
        assert price == pytest.approx(expected)

    def test_neither_call_site_nor_constructor_hourly_cost_raises_value_error(self, tmp_path):
        slots = make_local_slots(save_dir=str(tmp_path / "kv"))
        local = LocalResourceBudget(
            kv_dir=str(tmp_path / "kv"), slots=slots, verbose=False,
            gpu_reader_fn=lambda i: (_ for _ in ()).throw(GPUError("no gpu")),
        )
        estimator = _FakeEstimatorWithLatency(output_tokens=100, latency_s=2.0)

        with pytest.raises(ValueError, match="hourly_cost"):
            local.suggest_price_per_1k_tokens(estimator, input_tokens=2000)

    def test_estimator_not_ready_yet_yields_none_not_an_error(self, tmp_path):
        slots = make_local_slots(save_dir=str(tmp_path / "kv"))
        local = LocalResourceBudget(
            kv_dir=str(tmp_path / "kv"), slots=slots, hourly_cost=1.0, verbose=False,
            gpu_reader_fn=lambda i: (_ for _ in ()).throw(GPUError("no gpu")),
        )
        estimator = _FakeEstimatorWithLatency(output_tokens=100, latency_s=None)

        assert local.suggest_price_per_1k_tokens(estimator, input_tokens=2000) is None
