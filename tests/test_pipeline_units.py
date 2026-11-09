"""Units: in-memory store, PointWriter, BackfillEngine, CdcReconciler, bootstrap_blue."""

from __future__ import annotations

import pytest

from embedding_feature_store.core.errors import (
    ConflictError,
    NotFoundError,
    SchemaMismatchError,
    ValidationFailedError,
)
from embedding_feature_store.core.ports import VectorStore
from embedding_feature_store.core.projection import PointProjector
from embedding_feature_store.core.retry import Retrier
from embedding_feature_store.core.verification import ExpectationLedger, MigrationVerifier
from embedding_feature_store.pipeline.backfill import BackfillEngine
from embedding_feature_store.pipeline.bootstrap import bootstrap_blue
from embedding_feature_store.pipeline.reconcile import CdcReconciler, coalesce
from embedding_feature_store.pipeline.writer import PointWriter
from embedding_feature_store.schemas import (
    CdcEvent,
    CdcOp,
    CreateAliasAction,
    DeleteAliasAction,
    DistanceMetric,
    PointRecord,
    RetryPolicy,
    SourceRecord,
    VectorSpec,
)
from embedding_feature_store.storage.faults import FaultInjectingStore, FaultPlan
from embedding_feature_store.storage.memory import InMemoryVectorStore
from embedding_feature_store.storage.postgres_sim import PostgresSimulator
from embedding_feature_store.telemetry.metrics import MigrationMetrics
from tests.fakes import FakeEmbedder, RecordingSleep
from tests.pipeline_support import make_plan

SPEC = VectorSpec(dimension=3, distance=DistanceMetric.DOT)
_conforms: VectorStore = InMemoryVectorStore()


def pt(pid: int, *vector: float) -> PointRecord:
    return PointRecord(point_id=pid, vector=vector)


def rec(rid: int, lsn: int = 1, value: float = 1.0) -> SourceRecord:
    return SourceRecord(record_id=rid, tenant="t", features=(value, 2.0, 3.0), lsn=lsn)


async def target_store() -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    await store.create_collection("dst", SPEC)
    return store


def writer_for(
    store: VectorStore, *, embedder: FakeEmbedder | None = None
) -> tuple[PointWriter, ExpectationLedger, MigrationMetrics]:
    ledger = ExpectationLedger()
    metrics = MigrationMetrics(make_plan())
    writer = PointWriter(
        store,
        "dst",
        projector=PointProjector(SPEC, embedder or FakeEmbedder(dimension=3)),
        ledger=ledger,
        retrier=Retrier(RetryPolicy(max_attempts=20), sleep=RecordingSleep(), seed=0),
        metrics=metrics,
    )
    return writer, ledger, metrics


# ---------------------------------------------------------------------------
# InMemoryVectorStore
# ---------------------------------------------------------------------------


class TestInMemoryStore:
    @pytest.mark.asyncio
    async def test_collections_and_aliases_resolve(self) -> None:
        store = await target_store()
        await store.update_aliases([CreateAliasAction(alias="live", collection="dst")])
        await store.upsert("live", [pt(2, 1, 1, 1), pt(1, 2, 2, 2)])
        assert await store.collection_exists("live")
        assert await store.count("dst") == 2
        info = await store.describe_collection("live")
        assert (info.name, info.spec, info.points_count) == ("live", SPEC, 2)
        page = await store.scroll("live", offset=None, limit=1)
        assert ([p.point_id for p in page.points], page.next_offset) == ([1], 2)
        assert (await store.scroll("dst", offset=2, limit=5)).next_offset is None
        await store.delete_points("live", [1, 99])
        assert await store.count("live") == 1

    @pytest.mark.asyncio
    async def test_conflicts_and_missing(self) -> None:
        store = await target_store()
        await store.update_aliases([CreateAliasAction(alias="live", collection="dst")])
        for name in ("dst", "live"):
            with pytest.raises(ConflictError):
                await store.create_collection(name, SPEC)
        with pytest.raises(NotFoundError):
            await store.count("nope")
        assert not await store.collection_exists("nope")

    @pytest.mark.asyncio
    async def test_dimension_enforced_atomically(self) -> None:
        store = await target_store()
        with pytest.raises(ValidationFailedError, match="dimension 2"):
            await store.upsert("dst", [pt(1, 1, 1, 1), pt(2, 1, 1)])
        assert await store.count("dst") == 0

    @pytest.mark.asyncio
    async def test_alias_updates_are_all_or_nothing(self) -> None:
        store = await target_store()
        await store.update_aliases([CreateAliasAction(alias="live", collection="dst")])
        with pytest.raises(NotFoundError):
            await store.update_aliases(
                [DeleteAliasAction(alias="live"), CreateAliasAction(alias="live", collection="x")]
            )
        with pytest.raises(NotFoundError):
            await store.update_aliases([DeleteAliasAction(alias="ghost")])
        assert await store.list_aliases() == {"live": "dst"}

    @pytest.mark.asyncio
    async def test_delete_collection_drops_its_aliases_and_is_idempotent(self) -> None:
        store = await target_store()
        await store.create_collection("other", SPEC)
        await store.update_aliases(
            [
                CreateAliasAction(alias="a", collection="dst"),
                CreateAliasAction(alias="b", collection="other"),
            ]
        )
        await store.delete_collection("dst")
        await store.delete_collection("dst")
        assert await store.list_aliases() == {"b": "other"}


# ---------------------------------------------------------------------------
# PointWriter
# ---------------------------------------------------------------------------


class TestPointWriter:
    @pytest.mark.asyncio
    async def test_upsert_writes_records_ledger_and_metrics(self) -> None:
        store = await target_store()
        writer, ledger, metrics = writer_for(store)
        assert writer.collection == "dst"
        assert writer.ledger is ledger
        assert await writer.upsert([rec(1), rec(2)]) == 2
        assert await store.count("dst") == 2
        assert ledger.count == 2
        snap = metrics.snapshot()
        assert (snap.vectors_verified, snap.dimension_violations) == (2, 0)
        report = await MigrationVerifier(store).verify("dst", SPEC, ledger)
        assert report.passed

    @pytest.mark.asyncio
    async def test_empty_batches_are_noops(self) -> None:
        store = await target_store()
        writer, ledger, _ = writer_for(store)
        assert await writer.upsert([]) == 0
        assert await writer.delete([]) == 0
        assert ledger.count == 0

    @pytest.mark.asyncio
    async def test_dimension_violation_recorded_and_raised(self) -> None:
        store = await target_store()
        writer, ledger, metrics = writer_for(
            store, embedder=FakeEmbedder(dimension=3, transform=lambda f: tuple(f)[:2])
        )
        with pytest.raises(SchemaMismatchError):
            await writer.upsert([rec(1)])
        assert metrics.snapshot().dimension_violations == 1
        assert ledger.count == 0
        assert await store.count("dst") == 0

    @pytest.mark.asyncio
    async def test_delete_updates_ledger(self) -> None:
        store = await target_store()
        writer, ledger, _ = writer_for(store)
        await writer.upsert([rec(1), rec(2)])
        assert await writer.delete([1, 42]) == 2
        assert (ledger.count, await store.count("dst")) == (1, 1)

    @pytest.mark.asyncio
    async def test_retries_through_transient_faults(self) -> None:
        inner = await target_store()
        chaos = FaultInjectingStore(inner, FaultPlan(0.5, seed=3, ambiguous_fraction=0.5))
        writer, ledger, _ = writer_for(chaos)
        for i in range(30):
            await writer.upsert([rec(i, value=float(i))])
        for i in range(0, 30, 4):
            await writer.delete([i])
        assert sum(chaos.injected.values()) > 0
        assert (await MigrationVerifier(inner).verify("dst", SPEC, ledger)).passed


# ---------------------------------------------------------------------------
# BackfillEngine
# ---------------------------------------------------------------------------


def seeded_source(rows: int) -> PostgresSimulator:
    source = PostgresSimulator(queue_capacity=1)
    for i in range(1, rows + 1):
        source.upsert(i, "t", (float(i), 1.0, 0.5))
    return source


class TestBackfill:
    def test_batch_size_validated(self) -> None:
        writer, _, metrics = writer_for(InMemoryVectorStore())
        with pytest.raises(ValueError, match="batch_size"):
            BackfillEngine(PostgresSimulator(), writer, metrics, batch_size=0)

    @pytest.mark.asyncio
    async def test_writes_snapshot_in_batches_with_checkpoints(self) -> None:
        store = await target_store()
        source = seeded_source(10)
        source.upsert(11, "t", (1.0, 1.0, 1.0))  # after the snapshot: excluded
        writer, ledger, metrics = writer_for(store)
        cursors: list[int] = []
        engine = BackfillEngine(source, writer, metrics, batch_size=4, checkpoint=cursors.append)
        result = await engine.run(10)
        assert (result.snapshot_lsn, result.rows, result.batches, result.last_id) == (10, 10, 3, 10)
        assert cursors == [4, 8, 10]
        assert (await store.count("dst"), ledger.count) == (10, 10)
        assert (metrics.snapshot().backfilled, metrics.snapshot().batches) == (10, 3)

    @pytest.mark.asyncio
    async def test_resume_after_id_and_no_checkpoint(self) -> None:
        store = await target_store()
        writer, _, metrics = writer_for(store)
        result = await BackfillEngine(seeded_source(10), writer, metrics, batch_size=4).run(
            10, after_id=6
        )
        assert (result.rows, result.last_id) == (4, 10)

    @pytest.mark.asyncio
    async def test_empty_snapshot(self) -> None:
        writer, _, metrics = writer_for(await target_store())
        result = await BackfillEngine(PostgresSimulator(), writer, metrics, batch_size=4).run(0)
        assert (result.rows, result.batches, result.last_id) == (0, 0, None)

    @pytest.mark.asyncio
    async def test_checkpoint_abort_stops_between_batches_and_closes_cursor(self) -> None:
        store = await target_store()
        source = seeded_source(40)
        writer, _, metrics = writer_for(store)

        class StopError(Exception):
            pass

        def stop_after_second(last_id: int) -> None:
            if last_id >= 8:
                raise StopError

        engine = BackfillEngine(source, writer, metrics, batch_size=4, checkpoint=stop_after_second)
        with pytest.raises(StopError):
            await engine.run(40)
        assert await store.count("dst") == 8, "only whole batches are written"
        queries = len(source.queries)
        assert queries < 10, "the bounded cursor was not read to the end"


# ---------------------------------------------------------------------------
# CdcReconciler
# ---------------------------------------------------------------------------


def ev_upsert(lsn: int, rid: int, value: float = 1.0) -> CdcEvent:
    return CdcEvent(lsn=lsn, op=CdcOp.UPSERT, record_id=rid, record=rec(rid, lsn, value))


def ev_delete(lsn: int, rid: int) -> CdcEvent:
    return CdcEvent(lsn=lsn, op=CdcOp.DELETE, record_id=rid)


class TestCoalesce:
    def test_last_writer_wins_per_id(self) -> None:
        upserts, deletes = coalesce(
            [
                ev_upsert(1, 1),
                ev_delete(2, 1),
                ev_delete(3, 2),
                ev_upsert(4, 2, 9.0),
                ev_upsert(5, 3),
            ]
        )
        assert [(r.record_id, r.features[0]) for r in upserts] == [(2, 9.0), (3, 1.0)]
        assert deletes == [1]

    def test_empty(self) -> None:
        assert coalesce([]) == ([], [])


class TestReconciler:
    def test_page_size_validated(self) -> None:
        writer, _, metrics = writer_for(InMemoryVectorStore())
        with pytest.raises(ValueError, match="page_size"):
            CdcReconciler(PostgresSimulator(), writer, metrics, page_size=0)

    @pytest.mark.asyncio
    async def test_catch_up_applies_range_and_converges(self) -> None:
        store = await target_store()
        source = seeded_source(5)
        writer, ledger, metrics = writer_for(store)
        await BackfillEngine(source, writer, metrics, batch_size=10).run(5)
        source.upsert(2, "t", (9.0, 9.0, 9.0))
        source.delete(3)
        source.upsert(6, "t", (6.0, 6.0, 6.0))
        source.upsert(6, "t", (7.0, 7.0, 7.0))
        source.delete(6)
        source.upsert(7, "t", (1.0, 1.0, 1.0))  # beyond until_lsn
        lsns: list[int] = []
        reconciler = CdcReconciler(source, writer, metrics, page_size=2, checkpoint=lsns.append)
        applied = await reconciler.catch_up(5, 10)
        assert applied == 10
        assert lsns == [7, 9, 10]
        ids = sorted(p.point_id for p in (await store.scroll("dst", offset=None, limit=99)).points)
        assert ids == [1, 2, 4, 5]
        snap = metrics.snapshot()
        assert (snap.applied_lsn, snap.source_head_lsn, snap.cdc_events, snap.cdc_lag) == (
            10,
            10,
            5,
            0,
        )
        assert (await MigrationVerifier(store).verify("dst", SPEC, ledger)).passed

    @pytest.mark.asyncio
    async def test_replay_is_idempotent(self) -> None:
        store = await target_store()
        source = seeded_source(3)
        source.delete(2)
        writer, ledger, metrics = writer_for(store)
        reconciler = CdcReconciler(source, writer, metrics)
        await reconciler.catch_up(0, 4)
        before = ledger.digest
        await reconciler.catch_up(0, 4)
        assert ledger.digest == before
        assert await store.count("dst") == 2

    @pytest.mark.asyncio
    async def test_until_beyond_head_stops_when_log_is_exhausted(self) -> None:
        writer, _, metrics = writer_for(await target_store())
        source = seeded_source(3)
        assert await CdcReconciler(source, writer, metrics).catch_up(1, 99) == 3

    @pytest.mark.asyncio
    async def test_nothing_to_do(self) -> None:
        writer, _, metrics = writer_for(await target_store())
        reconciler = CdcReconciler(seeded_source(3), writer, metrics)
        assert await reconciler.catch_up(3, 3) == 3
        assert await reconciler.catch_up_to_head(3) == 3


# ---------------------------------------------------------------------------
# bootstrap_blue
# ---------------------------------------------------------------------------


class TestBootstrap:
    @pytest.mark.asyncio
    async def test_seeds_collection_and_binds_alias(self) -> None:
        store = InMemoryVectorStore()
        source = seeded_source(25)
        projector = PointProjector(SPEC, FakeEmbedder(dimension=3))
        retrier = Retrier(RetryPolicy(), sleep=RecordingSleep())
        lsn = await bootstrap_blue(
            store, source, projector, retrier, collection="blue", alias="live", batch_size=7
        )
        assert lsn == 25
        assert await store.count("live") == 25
        assert await store.list_aliases() == {"live": "blue"}

    @pytest.mark.asyncio
    async def test_refuses_to_repoint_bound_alias(self) -> None:
        store = await target_store()
        await store.update_aliases([CreateAliasAction(alias="live", collection="dst")])
        projector = PointProjector(SPEC, FakeEmbedder(dimension=3))
        with pytest.raises(ConflictError, match="already bound"):
            await bootstrap_blue(
                store,
                seeded_source(1),
                projector,
                Retrier(RetryPolicy(), sleep=RecordingSleep()),
                collection="blue",
                alias="live",
            )
        assert not await store.collection_exists("blue")

    @pytest.mark.asyncio
    async def test_retries_transient_failures(self) -> None:
        inner = InMemoryVectorStore()
        store = FaultInjectingStore(inner, FaultPlan(0.4, seed=11))
        projector = PointProjector(SPEC, FakeEmbedder(dimension=3))
        retrier = Retrier(RetryPolicy(max_attempts=30), sleep=RecordingSleep(), seed=1)
        await bootstrap_blue(
            store,
            seeded_source(30),
            projector,
            retrier,
            collection="blue",
            alias="live",
            batch_size=4,
        )
        assert await inner.count("blue") == 30
        assert sum(store.injected.values()) > 0
