"""PostgreSQL simulator: keyset cursor, MVCC snapshots, WAL sequencing."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest
from hypothesis import given
from hypothesis import strategies as st

from embedding_feature_store.core.errors import ValidationFailedError
from embedding_feature_store.core.ports import SourceReader
from embedding_feature_store.schemas import CdcOp, SourceRecord
from embedding_feature_store.storage.postgres_sim import KeysetQuery, PostgresSimulator

_conforms: SourceReader = PostgresSimulator()


async def read_all(
    db: PostgresSimulator, *, snapshot_lsn: int, after_id: int | None = None, batch_size: int = 3
) -> list[list[int]]:
    stream = db.stream_rows(snapshot_lsn=snapshot_lsn, after_id=after_id, batch_size=batch_size)
    return [[r.record_id for r in batch] async for batch in stream]


def seeded(n: int = 10) -> PostgresSimulator:
    db = PostgresSimulator()
    db.populate(n, 4, seed=1)
    return db


class TestWal:
    @pytest.mark.asyncio
    async def test_lsn_strictly_increasing_and_stamped(self) -> None:
        db = PostgresSimulator()
        events = [db.upsert(i, "t", (float(i),)) for i in (5, 3, 5)]
        assert [e.lsn for e in events] == [1, 2, 3]
        assert all(e.record is not None and e.record.lsn == e.lsn for e in events)
        assert await db.head_lsn() == 3
        assert await db.estimate_rows() == 2

    @pytest.mark.asyncio
    async def test_delete_logs_event_and_missing_delete_is_noop(self) -> None:
        db = PostgresSimulator()
        db.upsert(1, "t", (1.0,))
        event = db.delete(1)
        assert event is not None
        assert (event.lsn, event.op, event.record) == (2, CdcOp.DELETE, None)
        assert db.delete(1) is None
        assert db.delete(42) is None
        assert await db.head_lsn() == 2
        assert await db.estimate_rows() == 0

    @pytest.mark.asyncio
    async def test_changes_since_is_exclusive_ordered_and_limited(self) -> None:
        db = seeded(10)
        page = await db.changes_since(3, limit=4)
        assert [e.lsn for e in page] == [4, 5, 6, 7]
        assert await db.changes_since(10, limit=5) == ()
        assert [e.lsn for e in await db.changes_since(-1, limit=2)] == [1, 2]

    @pytest.mark.asyncio
    async def test_changes_since_is_replay_safe(self) -> None:
        db = seeded(5)
        assert await db.changes_since(1, limit=10) == await db.changes_since(1, limit=10)

    @pytest.mark.asyncio
    async def test_tailing_the_log_in_pages_sees_every_event_once(self) -> None:
        db = seeded(7)
        db.delete(3)
        db.upsert(2, "t0", (9.0,))
        seen: list[int] = []
        cursor = 0
        while batch := await db.changes_since(cursor, limit=3):
            seen += [e.lsn for e in batch]
            cursor = batch[-1].lsn
        assert seen == list(range(1, 10))

    @pytest.mark.asyncio
    async def test_limit_validated(self) -> None:
        with pytest.raises(ValidationFailedError):
            await PostgresSimulator().changes_since(0, limit=0)


class TestKeysetCursor:
    @pytest.mark.asyncio
    async def test_pages_in_id_order(self) -> None:
        db = seeded(10)
        assert await read_all(db, snapshot_lsn=10) == [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]

    @pytest.mark.asyncio
    async def test_issues_keyset_queries_and_stops_on_short_page(self) -> None:
        db = seeded(10)
        await read_all(db, snapshot_lsn=10)
        assert [(q.after_id, q.limit) for q in db.queries] == [(None, 3), (3, 3), (6, 3), (9, 3)]
        assert db.queries[0].statement == "SELECT * FROM features ORDER BY id LIMIT $1"
        assert db.queries[0].params == (3,)
        assert db.queries[1].statement == (
            "SELECT * FROM features WHERE id > $1 ORDER BY id LIMIT $2"
        )
        assert db.queries[1].params == (3, 3)

    @pytest.mark.asyncio
    async def test_exact_multiple_needs_one_empty_probe(self) -> None:
        db = seeded(9)
        assert await read_all(db, snapshot_lsn=9) == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        assert db.queries[-1] == KeysetQuery(snapshot_lsn=9, after_id=9, limit=3)

    @pytest.mark.asyncio
    async def test_resume_after_id(self) -> None:
        db = seeded(10)
        assert await read_all(db, snapshot_lsn=10, after_id=5, batch_size=4) == [[6, 7, 8, 9], [10]]

    @pytest.mark.asyncio
    async def test_sparse_ids(self) -> None:
        db = PostgresSimulator()
        for rid in (100, 7, 42, 3):
            db.upsert(rid, "t", (1.0,))
        assert await read_all(db, snapshot_lsn=4, after_id=7, batch_size=1) == [[42], [100]]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("lsn", [-1, 11])
    async def test_snapshot_bounds(self, lsn: int) -> None:
        with pytest.raises(ValidationFailedError, match="outside"):
            seeded(10).stream_rows(snapshot_lsn=lsn, after_id=None, batch_size=3)

    def test_batch_size_validated(self) -> None:
        with pytest.raises(ValidationFailedError, match="batch_size"):
            seeded(1).stream_rows(snapshot_lsn=1, after_id=None, batch_size=0)

    def test_queue_capacity_validated(self) -> None:
        with pytest.raises(ValueError, match="queue_capacity"):
            PostgresSimulator(queue_capacity=0)


class TestSnapshotIsolation:
    @pytest.mark.asyncio
    async def test_past_snapshot_hides_later_writes(self) -> None:
        db = seeded(5)
        original = db.snapshot(5)[2]
        db.upsert(2, "t0", (0.0, 0.0, 0.0, 0.0))
        db.delete(4)
        db.upsert(6, "t0", (1.0, 1.0, 1.0, 1.0))
        rows = db.snapshot(5)
        assert sorted(rows) == [1, 2, 3, 4, 5]
        assert rows[2] == original
        assert sorted(db.snapshot(await db.head_lsn())) == [1, 2, 3, 5, 6]
        assert db.snapshot(0) == {}

    @pytest.mark.asyncio
    async def test_writes_during_streaming_do_not_leak_in(self) -> None:
        db = seeded(6)
        stream = db.stream_rows(snapshot_lsn=6, after_id=None, batch_size=2)
        first = await anext(stream)
        db.delete(5)
        db.upsert(3, "t0", (7.0, 7.0, 7.0, 7.0))
        db.upsert(99, "t0", (7.0, 7.0, 7.0, 7.0))
        rest = [batch async for batch in stream]
        records: list[SourceRecord] = [*first, *(r for batch in rest for r in batch)]
        assert [r.record_id for r in records] == [1, 2, 3, 4, 5, 6]
        assert all(r.lsn <= 6 for r in records)


class TestBackpressure:
    @pytest.mark.asyncio
    async def test_cursor_is_throttled_by_bounded_queue(self) -> None:
        events: list[tuple[int, int, bool]] = []
        db = PostgresSimulator(queue_capacity=1, observer=lambda *e: events.append(e))
        db.populate(20, 2, seed=0)
        stream = db.stream_rows(snapshot_lsn=20, after_id=None, batch_size=2)
        await anext(stream)
        for _ in range(20):
            await asyncio.sleep(0)
        # 1 page consumed, 1 queued, 1 produced and blocked: 3 queries, not 10.
        assert len(db.queries) == 3
        assert (1, 1, True) in events
        await stream.aclose()


def test_populate_is_deterministic_and_cycles_tenants() -> None:
    a, b = PostgresSimulator(), PostgresSimulator()
    a.populate(4, 3, seed=9, tenants=("x", "y"))
    b.populate(4, 3, seed=9, tenants=("x", "y"))
    assert a.snapshot(4) == b.snapshot(4)
    assert [a.snapshot(4)[i].tenant for i in range(1, 5)] == ["x", "y", "x", "y"]


ops = st.lists(st.tuples(st.booleans(), st.integers(1, 8), st.floats(-5, 5)), max_size=40)


@given(ops, st.integers(1, 5))
def test_wal_replay_and_cursor_agree_with_table(
    history: list[tuple[bool, int, float]], batch_size: int
) -> None:
    db = PostgresSimulator()
    for is_upsert, rid, value in history:
        if is_upsert:
            db.upsert(rid, "t", (value,))
        else:
            db.delete(rid)

    async def scenario() -> tuple[Sequence[int], list[int]]:
        head = await db.head_lsn()
        events = await db.changes_since(0, limit=10_000)
        streamed = [
            r.record_id
            async for batch in db.stream_rows(
                snapshot_lsn=head, after_id=None, batch_size=batch_size
            )
            for r in batch
        ]
        return [e.lsn for e in events], streamed

    lsns, streamed = asyncio.run(scenario())
    assert lsns == list(range(1, len(lsns) + 1)), "WAL LSNs are gap-free and increasing"
    assert streamed == sorted(db.snapshot(lsns[-1] if lsns else 0))
