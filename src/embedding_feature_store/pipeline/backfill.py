"""Historical backfill from a consistent source snapshot.

The source's keyset cursor is consumed through its bounded queue (so a slow
embedder or store throttles the database read) and written batch by batch via
:class:`PointWriter`. After every batch the ``checkpoint`` hook receives the
last written id; it is the place to persist a resume cursor and to honour a
cooperative shutdown (by raising), which therefore only ever happens between
fully written batches.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass

from embedding_feature_store.core.ports import SourceReader
from embedding_feature_store.pipeline.writer import PointWriter
from embedding_feature_store.telemetry.metrics import MigrationMetrics

__all__ = ["BackfillEngine", "BackfillResult", "Checkpoint"]

type Checkpoint = Callable[[int], None]
"""Called with the last written record id after each batch; may raise to stop."""


@dataclass(frozen=True, slots=True)
class BackfillResult:
    snapshot_lsn: int
    rows: int
    batches: int
    last_id: int | None


class BackfillEngine:
    def __init__(
        self,
        source: SourceReader,
        writer: PointWriter,
        metrics: MigrationMetrics,
        *,
        batch_size: int,
        checkpoint: Checkpoint | None = None,
    ) -> None:
        if batch_size < 1:
            msg = "batch_size must be >= 1"
            raise ValueError(msg)
        self._source = source
        self._writer = writer
        self._metrics = metrics
        self._batch_size = batch_size
        self._checkpoint = checkpoint

    async def run(self, snapshot_lsn: int, *, after_id: int | None = None) -> BackfillResult:
        """Write every row of the snapshot with id > ``after_id``."""
        rows = batches = 0
        last_id = after_id
        stream = self._source.stream_rows(
            snapshot_lsn=snapshot_lsn, after_id=after_id, batch_size=self._batch_size
        )
        async with contextlib.aclosing(stream) as batches_in:
            async for batch in batches_in:
                written = await self._writer.upsert(batch)
                rows += written
                batches += 1
                last_id = batch[-1].record_id
                self._metrics.record_batch(written)
                if self._checkpoint is not None:
                    self._checkpoint(last_id)
        return BackfillResult(
            snapshot_lsn=snapshot_lsn, rows=rows, batches=batches, last_id=last_id
        )
