"""Behavioral contract for Phase 3: surviving what prediction cannot prevent.

Prediction reduces 429s drastically, but estimates are statistical — a 429 can
still happen. The contract:

* an unexpected 429 is retried with exponential backoff (or the provider's
  ``retry-after``, when given), up to a bound;
* every hit is *feedback*: the safety factor k adapts upward (capped), so the
  scheduler grows more cautious on workloads it keeps underestimating;
* a failed call must leave the session state UNTOUCHED — no phantom steps,
  no half-recorded consumption — so a resume is always clean;
* checkpoint I/O failures surface as a typed ``CheckpointError``, and real
  provider 429 exceptions map to ``RateLimitHit`` in the LiteLLM adapter.
"""

import pytest

from agentpause import (
    AgentPauseError,
    CheckpointError,
    PredictiveScheduler,
    RateLimitHit,
    RetryPolicy,
    StateStore,
)
from agentpause.adapters.litellm import LiteLLMAdapter, TelemetryError


# -- test doubles -----------------------------------------------------------

class FlakyBackend:
    """Raises RateLimitHit for the first ``failures`` calls, then succeeds."""

    def __init__(self, failures: int, retry_after=None, tokens=300):
        self.failures = failures
        self.retry_after = retry_after
        self.tokens = tokens
        self.calls = 0

    def __call__(self, messages):
        self.calls += 1
        if self.calls <= self.failures:
            raise RateLimitHit(retry_after=self.retry_after)
        return "ok", self.tokens


class SleepRecorder:
    def __init__(self):
        self.delays = []

    def __call__(self, s):
        self.delays.append(s)


def make_sched(backend, tmp_path, **kwargs):
    return PredictiveScheduler(
        backend=backend,
        telemetry=lambda: 100_000,
        store=StateStore(str(tmp_path)),
        **kwargs,
    )


# -- retry with backoff --------------------------------------------------------

def test_retries_once_and_succeeds(tmp_path):
    sleep = SleepRecorder()
    sched = make_sched(FlakyBackend(failures=1), tmp_path,
                       retry=RetryPolicy(sleep_fn=sleep))
    with sched.session("t") as s:
        s.add_user("q")
        assert s.call() == "ok"
    assert len(sleep.delays) == 1
    assert s.step == 1


def test_backoff_grows_exponentially(tmp_path):
    sleep = SleepRecorder()
    sched = make_sched(FlakyBackend(failures=3), tmp_path,
                       retry=RetryPolicy(base_delay_s=1.0, factor=2.0,
                                         jitter=0.0, sleep_fn=sleep))
    with sched.session("t") as s:
        s.add_user("q")
        s.call()
    assert sleep.delays == [1.0, 2.0, 4.0]


def test_retry_after_from_provider_wins_over_backoff(tmp_path):
    sleep = SleepRecorder()
    sched = make_sched(FlakyBackend(failures=1, retry_after=7.5), tmp_path,
                       retry=RetryPolicy(base_delay_s=1.0, sleep_fn=sleep))
    with sched.session("t") as s:
        s.add_user("q")
        s.call()
    assert sleep.delays == [7.5]


def test_gives_up_after_max_retries_and_state_is_untouched(tmp_path):
    sleep = SleepRecorder()
    sched = make_sched(FlakyBackend(failures=99), tmp_path,
                       retry=RetryPolicy(max_retries=2, sleep_fn=sleep))
    with sched.session("t") as s:
        s.add_user("q")
        with pytest.raises(RateLimitHit):
            s.call()
        # the failed call must not leave phantom progress behind
        assert s.step == 0
        assert s.total_tokens_used == 0
        assert [m["role"] for m in s.messages] == ["user"]
    assert len(sleep.delays) == 2          # waited max_retries times, then gave up


def test_delay_is_capped(tmp_path):
    p = RetryPolicy(base_delay_s=10.0, factor=10.0, max_delay_s=25.0, jitter=0.0)
    assert p.delay(0) == 10.0
    assert p.delay(1) == 25.0              # 100 → capped
    assert p.delay(5) == 25.0


def test_jitter_randomizes_within_bounds():
    """±25% jitter (default): delays vary but stay inside the band."""
    p = RetryPolicy(base_delay_s=8.0, factor=1.0)
    delays = {p.delay(0) for _ in range(50)}
    assert len(delays) > 1                          # actually randomized
    assert all(6.0 <= d <= 10.0 for d in delays)    # 8 ± 25%


# -- retriable vs non-retriable backend failures ---------------------------------

class Transient(Exception):
    pass


def test_retriable_backend_error_is_retried(tmp_path):
    from agentpause import BackendError
    calls = {"n": 0}

    def flaky_5xx(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            raise BackendError("502 upstream", retriable=True)
        return "ok", 100

    sched = make_sched(flaky_5xx, tmp_path, retry=RetryPolicy(jitter=0.0, sleep_fn=SleepRecorder()))
    with sched.session("t") as s:
        s.add_user("q")
        assert s.call() == "ok"
    assert sched.rate_limit_hits == 0      # a 5xx is not a rate-limit hit


def test_non_retriable_backend_error_propagates_immediately(tmp_path):
    from agentpause import BackendError
    calls = {"n": 0}

    def bad_request(messages):
        calls["n"] += 1
        raise BackendError("400 invalid request", retriable=False)

    sched = make_sched(bad_request, tmp_path, retry=RetryPolicy(sleep_fn=SleepRecorder()))
    with sched.session("t") as s:
        s.add_user("q")
        with pytest.raises(BackendError):
            s.call()
    assert calls["n"] == 1                 # no retries: it would waste budget


def test_adapter_maps_5xx_to_retriable_backend_error():
    from agentpause import BackendError

    class Fake503(Exception):
        status_code = 503

    def exploding(**kwargs):
        raise Fake503("service unavailable")

    adapter = LiteLLMAdapter(model="m", completion_fn=exploding)
    with pytest.raises(BackendError) as exc:
        adapter.backend([{"role": "user", "content": "hi"}])
    assert exc.value.retriable is True


# -- the 429 as feedback: adaptive safety factor --------------------------------

def test_safety_k_grows_after_each_hit_and_is_capped(tmp_path):
    sched = make_sched(FlakyBackend(failures=2), tmp_path,
                       retry=RetryPolicy(sleep_fn=SleepRecorder()),
                       safety_k=2.0, k_bump=0.25, k_max=2.4)
    with sched.session("t") as s:
        s.add_user("q")
        s.call()                            # 2 hits absorbed along the way
    assert sched.rate_limit_hits == 2
    assert sched.safety_k == 2.4            # 2.0 + 0.25 + 0.25 → capped at 2.4


def test_errors_share_a_common_base():
    assert issubclass(RateLimitHit, AgentPauseError)
    assert issubclass(CheckpointError, AgentPauseError)
    assert issubclass(TelemetryError, AgentPauseError)   # adapter alias too


# -- checkpoint I/O failures -----------------------------------------------------

def test_unwritable_store_raises_checkpoint_error(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("I am a file, not a directory")
    with pytest.raises(CheckpointError):
        StateStore(str(blocker / "sub"))    # cannot create a dir inside a file


def test_corrupted_checkpoint_raises_checkpoint_error(tmp_path):
    store = StateStore(str(tmp_path))
    (tmp_path / "bad.json").write_text("{not valid json")
    with pytest.raises(CheckpointError):
        store.load("bad")


# -- the LiteLLM adapter maps real provider 429s ---------------------------------

class Fake429(Exception):
    def __init__(self, retry_after=None):
        super().__init__("Rate limit reached")
        self.status_code = 429
        self.headers = {"retry-after": str(retry_after)} if retry_after else {}


def test_adapter_maps_429_to_rate_limit_hit():
    def exploding(**kwargs):
        raise Fake429(retry_after=12)

    adapter = LiteLLMAdapter(model="m", completion_fn=exploding)
    with pytest.raises(RateLimitHit) as exc:
        adapter.backend([{"role": "user", "content": "hi"}])
    assert exc.value.retry_after == 12.0


def test_adapter_leaves_other_errors_alone():
    def exploding(**kwargs):
        raise ValueError("something unrelated")

    adapter = LiteLLMAdapter(model="m", completion_fn=exploding)
    with pytest.raises(ValueError):
        adapter.backend([{"role": "user", "content": "hi"}])
