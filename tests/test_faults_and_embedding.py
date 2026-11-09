"""Seeded fault injection and the random-projection embedder."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import math
import random
import threading
from collections.abc import Sequence

import pytest
from hypothesis import given
from hypothesis import strategies as st

from embedding_feature_store.core.errors import (
    RateLimitedError,
    SchemaMismatchError,
    StoreTimeoutError,
    TransientError,
)
from embedding_feature_store.core.ports import EmbeddingAdapter, VectorStore
from embedding_feature_store.core.projection import PointProjector
from embedding_feature_store.core.retry import Retrier
from embedding_feature_store.core.verification import ExpectationLedger, MigrationVerifier
from embedding_feature_store.embedding import RandomProjectionEmbedder
from embedding_feature_store.schemas import (
    CreateAliasAction,
    DistanceMetric,
    PointRecord,
    RetryPolicy,
    VectorSpec,
)
from embedding_feature_store.storage.faults import ALL_METHODS, FaultInjectingStore, FaultPlan
from tests.fakes import InMemoryVectorStore, RecordingSleep

SPEC = VectorSpec(dimension=2, distance=DistanceMetric.DOT)

_store_conforms: VectorStore = FaultInjectingStore(InMemoryVectorStore(), FaultPlan(0.0, 0))
_embedder_conforms: EmbeddingAdapter = RandomProjectionEmbedder(4, 2, seed=0)


def pt(pid: int) -> PointRecord:
    return PointRecord(point_id=pid, vector=(float(pid), 1.0))


async def with_collection() -> InMemoryVectorStore:
    inner = InMemoryVectorStore()
    await inner.create_collection("c", SPEC)
    return inner


# ---------------------------------------------------------------------------
# FaultInjectingStore
# ---------------------------------------------------------------------------


class TestFaultPlan:
    @pytest.mark.parametrize(
        "kwargs",
        [{"rate": 1.5}, {"rate": -0.1}, {"rate": 0.5, "ambiguous_fraction": 2.0}],
    )
    def test_probabilities_validated(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError, match=r"within \[0, 1\]"):
            FaultPlan(seed=0, **kwargs)  # type: ignore[arg-type]

    def test_methods_validated(self) -> None:
        with pytest.raises(ValueError, match="unknown methods"):
            FaultPlan(0.1, 0, methods=frozenset({"upsert", "drop_table"}))


class TestFaultInjection:
    @pytest.mark.asyncio
    async def test_rate_zero_delegates_every_method(self) -> None:
        inner = InMemoryVectorStore()
        store = FaultInjectingStore(inner, FaultPlan(0.0, seed=1))
        assert store.inner is inner
        assert store.plan.rate == 0.0
        await store.create_collection("c", SPEC)
        assert await store.collection_exists("c")
        await store.upsert("c", [pt(1), pt(2)])
        await store.delete_points("c", [2])
        assert await store.count("c") == 1
        assert (await store.describe_collection("c")).points_count == 1
        assert [p.point_id for p in (await store.scroll("c", offset=None, limit=5)).points] == [1]
        await store.update_aliases([CreateAliasAction(alias="a", collection="c")])
        assert await store.list_aliases() == {"a": "c"}
        await store.delete_collection("c")
        assert set(store.calls) == ALL_METHODS
        assert not store.injected

    @pytest.mark.asyncio
    async def test_plain_faults_happen_before_delegating(self) -> None:
        inner = await with_collection()
        store = FaultInjectingStore(inner, FaultPlan(1.0, seed=3, retry_after=0.25))
        for i in range(40):
            with pytest.raises(TransientError) as exc:
                await store.upsert("c", [pt(i)])
            if isinstance(exc.value, RateLimitedError):
                assert exc.value.retry_after == 0.25
            else:
                assert isinstance(exc.value, StoreTimeoutError)
        assert await inner.count("c") == 0, "non-ambiguous faults must not apply the write"
        kinds = {kind for (_, kind) in store.injected}
        assert kinds == {"rate_limit", "timeout"}

    @pytest.mark.asyncio
    async def test_ambiguous_timeouts_apply_then_fail(self) -> None:
        inner = await with_collection()
        store = FaultInjectingStore(inner, FaultPlan(1.0, seed=5, ambiguous_fraction=1.0))
        applied = 0
        for i in range(30):
            with pytest.raises(TransientError) as exc:
                await store.upsert("c", [pt(i)])
            if isinstance(exc.value, StoreTimeoutError):
                assert "ambiguous_timeout" in str(exc.value)
                applied += 1
        assert await inner.count("c") == applied > 0

    @pytest.mark.asyncio
    async def test_method_filter(self) -> None:
        inner = await with_collection()
        store = FaultInjectingStore(inner, FaultPlan(1.0, seed=0, methods=frozenset({"upsert"})))
        for _ in range(20):
            assert await store.count("c") == 0
        with pytest.raises(TransientError):
            await store.upsert("c", [pt(1)])

    @pytest.mark.asyncio
    async def test_same_seed_same_faults(self) -> None:
        async def trace(seed: int) -> list[str]:
            store = FaultInjectingStore(await with_collection(), FaultPlan(0.4, seed=seed))
            out: list[str] = []
            for _ in range(200):
                try:
                    await store.count("c")
                    out.append("ok")
                except TransientError as exc:
                    out.append(type(exc).__name__)
            return out

        assert await trace(11) == await trace(11)
        assert await trace(11) != await trace(12)

    @pytest.mark.asyncio
    async def test_fault_distribution(self) -> None:
        plan = FaultPlan(0.3, seed=2024, ambiguous_fraction=0.5)
        store = FaultInjectingStore(await with_collection(), plan)
        n = 6000
        for _ in range(n):
            with contextlib.suppress(TransientError):
                await store.count("c")
        faults = sum(store.injected.values())
        assert faults / n == pytest.approx(0.3, abs=0.02)
        rate_limited = store.injected["count", "rate_limit"]
        assert rate_limited / faults == pytest.approx(0.5, abs=0.04)
        timeouts = faults - rate_limited
        assert store.injected["count", "ambiguous_timeout"] / timeouts == pytest.approx(
            0.5, abs=0.05
        )

    @pytest.mark.asyncio
    async def test_chaos_writes_converge_with_retries(self) -> None:
        """Idempotent writes + retries survive lost responses and throttling."""
        inner = await with_collection()
        store = FaultInjectingStore(inner, FaultPlan(0.35, seed=7, ambiguous_fraction=0.5))
        retrier = Retrier(RetryPolicy(max_attempts=50), sleep=RecordingSleep(), seed=7)
        ledger = ExpectationLedger()
        for batch in itertools.batched(range(200), 10):
            points = [pt(i) for i in batch]
            await retrier.run(lambda points=points: store.upsert("c", points))  # type: ignore[misc]
            for p in points:
                ledger.record_upsert(p)
        for pid in range(0, 200, 3):
            await retrier.run(lambda pid=pid: store.delete_points("c", [pid]))  # type: ignore[misc]
            ledger.record_delete(pid)
        assert sum(store.injected.values()) > 50
        report = await MigrationVerifier(inner, page_size=17).require("c", SPEC, ledger)
        assert report.passed


# ---------------------------------------------------------------------------
# RandomProjectionEmbedder
# ---------------------------------------------------------------------------


class TestEmbedder:
    @pytest.mark.parametrize(("k", "d"), [(0, 3), (3, 0)])
    def test_dimensions_validated(self, k: int, d: int) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            RandomProjectionEmbedder(k, d, seed=0)

    def test_metadata(self) -> None:
        emb = RandomProjectionEmbedder(5, 3, seed=42)
        assert (emb.input_dimension, emb.dimension) == (5, 3)
        assert emb.model_id == "rp-gauss:5x3:seed=42"

    @pytest.mark.asyncio
    async def test_output_shape_and_determinism(self) -> None:
        batch = [[1.0, 2.0, 3.0, 4.0], [0.0, -1.0, 0.5, 2.0]]
        a = await RandomProjectionEmbedder(4, 6, seed=1).embed(batch)
        b = await RandomProjectionEmbedder(4, 6, seed=1).embed(batch)
        c = await RandomProjectionEmbedder(4, 6, seed=2).embed(batch)
        assert a == b
        assert a != c
        assert [len(v) for v in a] == [6, 6]

    @pytest.mark.asyncio
    async def test_empty_batch(self) -> None:
        assert await RandomProjectionEmbedder(2, 2, seed=0).embed([]) == ()

    def test_input_dimension_checked_at_call_time(self) -> None:
        emb = RandomProjectionEmbedder(3, 2, seed=0)
        with pytest.raises(SchemaMismatchError, match="input 1 has 2 features, expected 3"):
            _ = emb.embed([[1.0, 2.0, 3.0], [1.0, 2.0]])

    @pytest.mark.asyncio
    async def test_runs_off_the_event_loop_thread(self) -> None:
        threads: list[int] = []

        class Recording(RandomProjectionEmbedder):
            def project_batch(
                self, batch: Sequence[Sequence[float]]
            ) -> tuple[tuple[float, ...], ...]:
                threads.append(threading.get_ident())
                return super().project_batch(batch)

        await Recording(2, 2, seed=0).embed([[1.0, 1.0]])
        assert threads
        assert threads[0] != threading.get_ident()

    @pytest.mark.asyncio
    async def test_input_snapshotted_before_offload(self) -> None:
        emb = RandomProjectionEmbedder(2, 3, seed=0)
        buffer = [[1.0, 2.0]]
        pending = emb.embed(buffer)
        buffer[0][0] = 999.0
        buffer.append([5.0, 5.0])
        result = await pending
        assert result == emb.project_batch([[1.0, 2.0]])

    @pytest.mark.asyncio
    async def test_plugs_into_projector(self) -> None:
        emb = RandomProjectionEmbedder(4, 8, seed=3)
        projector = PointProjector(VectorSpec(dimension=8, distance=DistanceMetric.COSINE), emb)
        with pytest.raises(SchemaMismatchError):
            PointProjector(VectorSpec(dimension=4, distance=DistanceMetric.COSINE), emb)
        from embedding_feature_store.schemas import SourceRecord  # noqa: PLC0415

        rows = [
            SourceRecord(record_id=i, tenant="t", features=(1.0, i, 0.0, 2.0), lsn=i)
            for i in (1, 2)
        ]
        points = await projector.project(rows)
        assert all(len(p.vector) == 8 for p in points)
        assert dict(points[0].payload)["model_id"] == emb.model_id

    def test_johnson_lindenstrauss_distance_preservation(self) -> None:
        rng = random.Random(0)  # noqa: S311 - test data
        k, d, n = 64, 512, 30
        data = [[rng.gauss(0, 1) for _ in range(k)] for _ in range(n)]
        projected = RandomProjectionEmbedder(k, d, seed=99).project_batch(data)
        ratios = [
            math.dist(projected[i], projected[j]) / math.dist(data[i], data[j])
            for i, j in itertools.combinations(range(n), 2)
        ]
        assert min(ratios) > 0.8
        assert max(ratios) < 1.2

    @given(
        st.lists(st.floats(-100, 100), min_size=3, max_size=3),
        st.lists(st.floats(-100, 100), min_size=3, max_size=3),
    )
    def test_projection_is_linear(self, x: list[float], y: list[float]) -> None:
        emb = RandomProjectionEmbedder(3, 4, seed=5)
        fx, fy, fxy = emb.project_batch([x, y, [a + b for a, b in zip(x, y, strict=True)]])
        for a, b, ab in zip(fx, fy, fxy, strict=True):
            assert ab == pytest.approx(a + b, abs=1e-9)

    @pytest.mark.asyncio
    async def test_concurrent_embeds_do_not_interfere(self) -> None:
        emb = RandomProjectionEmbedder(3, 3, seed=8)
        batches = [[[float(i), 1.0, -1.0]] for i in range(20)]
        results = await asyncio.gather(*(emb.embed(b) for b in batches))
        assert list(results) == [emb.project_batch(b) for b in batches]
