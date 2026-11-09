"""The single write path into the target collection.

``PointWriter`` turns source rows into points (embed -> project), asserts their
dimension, writes them with retries, and only then records them in the
:class:`ExpectationLedger` — so the ledger describes exactly what the store
acknowledged. Every upsert and delete is idempotent, which is what makes the
retries (and replays after ambiguous timeouts) safe.
"""

from __future__ import annotations

from collections.abc import Sequence

from embedding_feature_store.core.errors import SchemaMismatchError
from embedding_feature_store.core.ports import VectorStore
from embedding_feature_store.core.projection import PointProjector
from embedding_feature_store.core.retry import Retrier
from embedding_feature_store.core.verification import ExpectationLedger
from embedding_feature_store.schemas import SourceRecord
from embedding_feature_store.telemetry.metrics import MigrationMetrics

__all__ = ["PointWriter"]


class PointWriter:
    def __init__(
        self,
        store: VectorStore,
        collection: str,
        *,
        projector: PointProjector,
        ledger: ExpectationLedger,
        retrier: Retrier,
        metrics: MigrationMetrics,
    ) -> None:
        self._store = store
        self._collection = collection
        self._projector = projector
        self._ledger = ledger
        self._retrier = retrier
        self._metrics = metrics

    @property
    def collection(self) -> str:
        return self._collection

    @property
    def ledger(self) -> ExpectationLedger:
        return self._ledger

    async def upsert(self, records: Sequence[SourceRecord]) -> int:
        """Embed, check and write ``records``; returns the number of points written."""
        if not records:
            return 0
        try:
            points = await self._projector.project(records)
        except SchemaMismatchError:
            self._metrics.record_dimension_check(verified=0, violations=1)
            raise
        self._metrics.record_dimension_check(verified=len(points))
        await self._retrier.run(lambda: self._store.upsert(self._collection, points))
        for point in points:
            self._ledger.record_upsert(point)
        return len(points)

    async def delete(self, record_ids: Sequence[int]) -> int:
        """Delete points by id (missing ids are fine); returns the number requested."""
        if not record_ids:
            return 0
        ids = list(record_ids)
        await self._retrier.run(lambda: self._store.delete_points(self._collection, ids))
        for record_id in ids:
            self._ledger.record_delete(record_id)
        return len(ids)
