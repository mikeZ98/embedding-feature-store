"""Change-data-capture catch-up.

:meth:`CdcReconciler.catch_up` replays the source's logical-replication log in
pages over ``(from_lsn, until_lsn]``. Within a page, events are coalesced per
record id (last writer wins), so an id touched many times costs one write, and
upserts and deletes of distinct ids can be applied in either order. Pages are
applied strictly in LSN order, so the target converges on the source state at
``until_lsn``. Replaying a range is harmless because every write is
idempotent.
"""

from __future__ import annotations

from collections.abc import Callable

from embedding_feature_store.core.ports import SourceReader
from embedding_feature_store.pipeline.writer import PointWriter
from embedding_feature_store.schemas import CdcEvent, CdcOp, SourceRecord
from embedding_feature_store.telemetry.metrics import MigrationMetrics

__all__ = ["CdcReconciler", "coalesce"]


def coalesce(events: list[CdcEvent]) -> tuple[list[SourceRecord], list[int]]:
    """Reduce ``events`` (in LSN order) to final upserts and deletes per record id."""
    latest: dict[int, CdcEvent] = {}
    for event in events:
        latest[event.record_id] = event
    upserts = [e.record for e in latest.values() if e.op is CdcOp.UPSERT and e.record is not None]
    deletes = [e.record_id for e in latest.values() if e.op is CdcOp.DELETE]
    return upserts, deletes


class CdcReconciler:
    def __init__(
        self,
        source: SourceReader,
        writer: PointWriter,
        metrics: MigrationMetrics,
        *,
        page_size: int = 512,
        checkpoint: Callable[[int], None] | None = None,
    ) -> None:
        if page_size < 1:
            msg = "page_size must be >= 1"
            raise ValueError(msg)
        self._source = source
        self._writer = writer
        self._metrics = metrics
        self._page_size = page_size
        self._checkpoint = checkpoint

    async def catch_up(self, from_lsn: int, until_lsn: int) -> int:
        """Apply every event with ``from_lsn < lsn <= until_lsn``; returns the applied LSN."""
        self._metrics.set_source_head(until_lsn)
        cursor = from_lsn
        while cursor < until_lsn:
            page = [
                e
                for e in await self._source.changes_since(cursor, limit=self._page_size)
                if e.lsn <= until_lsn
            ]
            if not page:
                break
            upserts, deletes = coalesce(page)
            await self._writer.upsert(upserts)
            await self._writer.delete(deletes)
            cursor = page[-1].lsn
            self._metrics.record_cdc(cursor, events=len(page))
            if self._checkpoint is not None:
                self._checkpoint(cursor)
        return cursor

    async def catch_up_to_head(self, from_lsn: int) -> int:
        """Catch up to the source head as observed now."""
        return await self.catch_up(from_lsn, await self._source.head_lsn())
