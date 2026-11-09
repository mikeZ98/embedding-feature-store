"""Full-jitter backoff: formula, distribution, transient vs permanent handling."""

from __future__ import annotations

import statistics

import pytest
from hypothesis import given
from hypothesis import strategies as st

from embedding_feature_store.core.errors import (
    FeatureStoreError,
    NotFoundError,
    PermanentError,
    RateLimitedError,
    RetryExhaustedError,
    ServerError,
    StoreConnectionError,
    StoreTimeoutError,
    TransientError,
)
from embedding_feature_store.core.retry import Retrier, backoff_ceiling
from embedding_feature_store.schemas import RetryPolicy
from tests.fakes import RecordingSleep

POLICY = RetryPolicy(max_attempts=5, base_delay_s=0.1, max_delay_s=1.0, multiplier=2.0)

policies = st.builds(
    RetryPolicy,
    max_attempts=st.integers(1, 100),
    base_delay_s=st.floats(1e-6, 10.0),
    max_delay_s=st.just(10.0),
    multiplier=st.floats(1.0, 1e6),
)


class Flaky:
    """Fails with the given errors in order, then returns ``result``."""

    def __init__(self, *errors: Exception, result: str = "ok") -> None:
        self.errors = list(errors)
        self.calls = 0
        self.result = result

    async def __call__(self) -> str:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.result


class TestBackoffCeiling:
    def test_exponential_then_capped(self) -> None:
        ceilings = [backoff_ceiling(POLICY, n) for n in range(1, 7)]
        assert ceilings == pytest.approx([0.1, 0.2, 0.4, 0.8, 1.0, 1.0])

    def test_retry_number_is_one_based(self) -> None:
        with pytest.raises(ValueError, match="1-based"):
            backoff_ceiling(POLICY, 0)

    def test_overflow_saturates_at_cap(self) -> None:
        policy = RetryPolicy(base_delay_s=1.0, max_delay_s=5.0, multiplier=1e300)
        assert backoff_ceiling(policy, 3) == 5.0

    @given(policies, st.integers(1, 200))
    def test_bounded_and_monotone(self, policy: RetryPolicy, n: int) -> None:
        ceiling = backoff_ceiling(policy, n)
        assert 0 < ceiling <= policy.max_delay_s
        assert backoff_ceiling(policy, n + 1) >= ceiling


class TestJitter:
    @given(st.integers(), st.integers(1, 50))
    def test_delay_within_full_jitter_window(self, seed: int, n: int) -> None:
        delay = Retrier(POLICY, seed=seed).delay_for(n)
        assert 0.0 <= delay <= backoff_ceiling(POLICY, n)

    def test_seeded_sequences_are_reproducible(self) -> None:
        a, b = Retrier(POLICY, seed=7), Retrier(POLICY, seed=7)
        assert [a.delay_for(3) for _ in range(20)] == [b.delay_for(3) for _ in range(20)]
        assert Retrier(POLICY, seed=8).delay_for(3) != Retrier(POLICY, seed=7).delay_for(3)

    def test_distribution_is_uniform_over_window(self) -> None:
        retrier = Retrier(POLICY, seed=12345)
        ceiling = backoff_ceiling(POLICY, 4)
        draws = [retrier.delay_for(4) / ceiling for _ in range(20_000)]
        assert statistics.fmean(draws) == pytest.approx(0.5, abs=0.01)
        assert statistics.pvariance(draws) == pytest.approx(1 / 12, abs=0.005)
        deciles = [0] * 10
        for d in draws:
            deciles[min(int(d * 10), 9)] += 1
        assert all(abs(count / len(draws) - 0.1) < 0.01 for count in deciles)

    def test_retry_after_is_a_floor(self) -> None:
        retrier = Retrier(POLICY, seed=1)
        assert retrier.delay_for(1, RateLimitedError("slow", retry_after=3.0)) == 3.0
        unbounded = retrier.delay_for(1, RateLimitedError("slow"))
        assert 0.0 <= unbounded <= 0.1

    def test_policy_is_exposed(self) -> None:
        assert Retrier(POLICY).policy is POLICY


class TestRun:
    @pytest.mark.asyncio
    async def test_success_needs_no_retry(self) -> None:
        sleep = RecordingSleep()
        op = Flaky()
        assert await Retrier(POLICY, sleep=sleep, seed=0).run(op) == "ok"
        assert (op.calls, sleep.delays) == (1, [])

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error",
        [
            StoreTimeoutError("t"),
            StoreConnectionError("reset"),
            ServerError("bad gateway", status_code=502),
            RateLimitedError("slow"),
        ],
    )
    async def test_transient_errors_are_replayed(self, error: TransientError) -> None:
        sleep = RecordingSleep()
        seen: list[tuple[int, float, TransientError]] = []
        retrier = Retrier(
            POLICY, sleep=sleep, seed=3, on_retry=lambda n, d, e: seen.append((n, d, e))
        )
        op = Flaky(error, error)
        assert await retrier.run(op) == "ok"
        assert op.calls == 3
        assert len(sleep.delays) == 2
        assert [(n, e) for n, _, e in seen] == [(1, error), (2, error)]
        assert [d for _, d, _ in seen] == sleep.delays

    @pytest.mark.asyncio
    async def test_delays_follow_seeded_full_jitter(self) -> None:
        sleep = RecordingSleep()
        op = Flaky(*(StoreTimeoutError("t") for _ in range(4)))
        await Retrier(POLICY, sleep=sleep, seed=99).run(op)
        reference = Retrier(POLICY, seed=99)
        assert sleep.delays == [reference.delay_for(n) for n in range(1, 5)]

    @pytest.mark.asyncio
    async def test_retry_after_honoured_in_run(self) -> None:
        sleep = RecordingSleep()
        op = Flaky(RateLimitedError("slow", retry_after=7.5))
        await Retrier(POLICY, sleep=sleep, seed=0).run(op)
        assert sleep.delays == [7.5]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("error", [NotFoundError("gone"), PermanentError("no")])
    async def test_permanent_errors_are_not_replayed(self, error: PermanentError) -> None:
        sleep = RecordingSleep()
        op = Flaky(error)
        with pytest.raises(type(error)):
            await Retrier(POLICY, sleep=sleep).run(op)
        assert (op.calls, sleep.delays) == (1, [])

    @pytest.mark.asyncio
    async def test_foreign_exceptions_are_not_replayed(self) -> None:
        op = Flaky(KeyError("bug"))
        with pytest.raises(KeyError):
            await Retrier(POLICY, sleep=RecordingSleep()).run(op)
        assert op.calls == 1

    @pytest.mark.asyncio
    async def test_exhaustion(self) -> None:
        sleep = RecordingSleep()
        errors = [StoreTimeoutError(f"t{i}") for i in range(10)]
        op = Flaky(*errors)
        with pytest.raises(RetryExhaustedError) as exc:
            await Retrier(POLICY, sleep=sleep).run(op)
        assert op.calls == POLICY.max_attempts == exc.value.attempts
        assert exc.value.last_error is errors[POLICY.max_attempts - 1]
        assert exc.value.__cause__ is errors[POLICY.max_attempts - 1]
        assert len(sleep.delays) == POLICY.max_attempts - 1
        assert isinstance(exc.value, PermanentError)

    @pytest.mark.asyncio
    async def test_single_attempt_policy_never_sleeps(self) -> None:
        sleep = RecordingSleep()
        with pytest.raises(RetryExhaustedError):
            await Retrier(RetryPolicy(max_attempts=1), sleep=sleep).run(
                Flaky(StoreTimeoutError("t"))
            )
        assert sleep.delays == []

    @pytest.mark.asyncio
    async def test_default_sleep_is_real_asyncio_sleep(self) -> None:
        policy = RetryPolicy(base_delay_s=0.001, max_delay_s=0.001)
        assert await Retrier(policy).run(Flaky(StoreTimeoutError("t"))) == "ok"


def test_taxonomy_is_disjoint() -> None:
    assert issubclass(TransientError, FeatureStoreError)
    assert issubclass(PermanentError, FeatureStoreError)
    assert not issubclass(TransientError, PermanentError)
    assert not issubclass(PermanentError, TransientError)
