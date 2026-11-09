"""Static conformance of the test doubles to the hexagonal ports."""

from __future__ import annotations

import pytest

from embedding_feature_store.core.ports import EmbeddingAdapter, Sleep, SourceReader, VectorStore
from embedding_feature_store.schemas import SourceRecord
from tests.fakes import FakeEmbedder, FakeSource, InMemoryVectorStore, RecordingSleep

# mypy verifies these assignments structurally; a drifting fake fails type-checking.
_store: VectorStore = InMemoryVectorStore()
_source: SourceReader = FakeSource()
_embedder: EmbeddingAdapter = FakeEmbedder(dimension=2)
_sleep: Sleep = RecordingSleep()


@pytest.mark.asyncio
async def test_fake_source_streams_snapshot_in_id_batches() -> None:
    rows = [
        SourceRecord(record_id=i, tenant="t", features=(1.0,), lsn=lsn)
        for i, lsn in [(3, 1), (1, 1), (2, 9), (4, 1)]
    ]
    source = FakeSource(rows=rows)
    batches = [
        [r.record_id for r in batch]
        async for batch in source.stream_rows(snapshot_lsn=5, after_id=1, batch_size=1)
    ]
    assert batches == [[3], [4]]
    assert await source.estimate_rows() == 4
    assert await source.head_lsn() == 0
