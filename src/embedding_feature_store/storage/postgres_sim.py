"""Deterministic in-process stand-in for a PostgreSQL ``features`` table.

Models the two read paths a migration uses against a real database:

* **Snapshot cursor** — keyset pagination
  (``SELECT ... WHERE id > $last ORDER BY id LIMIT $batch``) over the table as
  of a given LSN (MVCC snapshot), streamed through :func:`bounded_stream` so a
  slow consumer throttles the cursor.
* **Logical replication** — every committed write is stamped with a strictly
  increasing LSN and appended to a WAL of :class:`CdcEvent`, readable with
  :meth:`PostgresSimulator.changes_since`.

The snapshot for any LSN is reconstructed by replaying the WAL, so reads are
exactly repeatable regardless of writes that happen while a stream is open.
"""

from __future__ import annotations

import asyncio
import bisect
import random
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass

from embedding_feature_store.core.errors import ValidationFailedError
from embedding_feature_store.core.ports import QueueObserver
from embedding_feature_store.schemas import CdcEvent, CdcOp, SourceRecord
from embedding_feature_store.storage.bounded_stream import bounded_stream

__all__ = ["KeysetQuery", "PostgresSimulator"]


@dataclass(frozen=True, slots=True)
class KeysetQuery:
    """One page request the simulated cursor issued (for inspection in tests)."""

    snapshot_lsn: int
    after_id: int | None
    limit: int

    @property
    def statement(self) -> str:
        """The parameterised SQL a driver would send (values bound separately)."""
        if self.after_id is None:
            return "SELECT * FROM features ORDER BY id LIMIT $1"
        return "SELECT * FROM features WHERE id > $1 ORDER BY id LIMIT $2"

    @property
    def params(self) -> tuple[int, ...]:
        return (self.limit,) if self.after_id is None else (self.after_id, self.limit)


class PostgresSimulator:
    """Implements :class:`~embedding_feature_store.core.ports.SourceReader`."""

    def __init__(
        self,
        *,
        queue_capacity: int = 4,
        observer: QueueObserver | None = None,
    ) -> None:
        if queue_capacity < 1:
            msg = "queue_capacity must be >= 1"
            raise ValueError(msg)
        self._queue_capacity = queue_capacity
        self._observer = observer
        self._rows: dict[int, SourceRecord] = {}
        self._wal: list[CdcEvent] = []
        self._lsn = 0
        self.queries: list[KeysetQuery] = []

    # -- writes (the "application" side) -----------------------------------

    def upsert(self, record_id: int, tenant: str, features: tuple[float, ...]) -> CdcEvent:
        """INSERT ... ON CONFLICT DO UPDATE; commits and logs one WAL event."""
        lsn = self._lsn + 1
        record = SourceRecord(record_id=record_id, tenant=tenant, features=features, lsn=lsn)
        event = CdcEvent(lsn=lsn, op=CdcOp.UPSERT, record_id=record_id, record=record)
        self._commit(event)
        return event

    def delete(self, record_id: int) -> CdcEvent | None:
        """DELETE; a no-op (no WAL entry) when the row does not exist."""
        if record_id not in self._rows:
            return None
        event = CdcEvent(lsn=self._lsn + 1, op=CdcOp.DELETE, record_id=record_id)
        self._commit(event)
        return event

    def populate(
        self, count: int, dimension: int, *, seed: int, tenants: Sequence[str] = ("t0",)
    ) -> None:
        """Insert ``count`` rows (ids 1..count) of seeded Gaussian features."""
        rng = random.Random(seed)  # noqa: S311 - synthetic data, not cryptography
        for record_id in range(1, count + 1):
            features = tuple(rng.gauss(0.0, 1.0) for _ in range(dimension))
            self.upsert(record_id, tenants[(record_id - 1) % len(tenants)], features)

    def _commit(self, event: CdcEvent) -> None:
        self._lsn = event.lsn
        self._wal.append(event)
        self._apply(self._rows, event)

    @staticmethod
    def _apply(table: dict[int, SourceRecord], event: CdcEvent) -> None:
        if event.record is None:
            table.pop(event.record_id, None)
        else:
            table[event.record_id] = event.record

    # -- SourceReader ------------------------------------------------------

    async def head_lsn(self) -> int:
        return self._lsn

    async def estimate_rows(self) -> int:
        return len(self._rows)

    def snapshot(self, lsn: int) -> dict[int, SourceRecord]:
        """The table as of ``lsn`` (MVCC view), rebuilt from the WAL."""
        if not 0 <= lsn <= self._lsn:
            msg = f"snapshot lsn {lsn} outside [0, {self._lsn}]"
            raise ValidationFailedError(msg)
        if lsn == self._lsn:
            return dict(self._rows)
        table: dict[int, SourceRecord] = {}
        for event in self._wal[: self._wal_index_after(lsn)]:
            self._apply(table, event)
        return table

    def stream_rows(
        self, *, snapshot_lsn: int, after_id: int | None, batch_size: int
    ) -> AsyncGenerator[Sequence[SourceRecord]]:
        if batch_size < 1:
            msg = "batch_size must be >= 1"
            raise ValidationFailedError(msg)
        table = self.snapshot(snapshot_lsn)
        return bounded_stream(
            self._keyset_cursor(table, snapshot_lsn, after_id, batch_size),
            capacity=self._queue_capacity,
            observer=self._observer,
        )

    async def _keyset_cursor(
        self,
        table: dict[int, SourceRecord],
        snapshot_lsn: int,
        after_id: int | None,
        batch_size: int,
    ) -> AsyncGenerator[tuple[SourceRecord, ...]]:
        ids = sorted(table)
        last = after_id
        while True:
            self.queries.append(KeysetQuery(snapshot_lsn, last, batch_size))
            start = 0 if last is None else bisect.bisect_right(ids, last)
            page = ids[start : start + batch_size]
            await asyncio.sleep(0)  # a round trip to the database
            if page:
                yield tuple(table[i] for i in page)
            if len(page) < batch_size:  # short page: the cursor is exhausted
                return
            last = page[-1]

    async def changes_since(self, lsn: int, *, limit: int) -> Sequence[CdcEvent]:
        if limit < 1:
            msg = "limit must be >= 1"
            raise ValidationFailedError(msg)
        start = self._wal_index_after(lsn)
        return tuple(self._wal[start : start + limit])

    def _wal_index_after(self, lsn: int) -> int:
        return bisect.bisect_right(self._wal, lsn, key=lambda event: event.lsn)
