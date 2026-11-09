"""Provision the initial ("Blue") collection a migration starts from.

Used for local runs, demos and benchmarks: creates ``collection`` with the
projector's spec, loads the source snapshot at its current head and binds
``alias`` to it, so that live traffic has something to read before the
migration to "Green" begins.
"""

from __future__ import annotations

import contextlib
import functools

from embedding_feature_store.core.errors import ConflictError
from embedding_feature_store.core.ports import SourceReader, VectorStore
from embedding_feature_store.core.projection import PointProjector
from embedding_feature_store.core.retry import Retrier
from embedding_feature_store.schemas import CreateAliasAction

__all__ = ["bootstrap_blue"]


async def bootstrap_blue(
    store: VectorStore,
    source: SourceReader,
    projector: PointProjector,
    retrier: Retrier,
    *,
    collection: str,
    alias: str,
    batch_size: int = 256,
) -> int:
    """Create, seed and alias the Blue collection; returns the snapshot LSN loaded.

    Refuses (:class:`ConflictError`) if ``alias`` is already bound, so it can
    never silently repoint production traffic.
    """
    aliases = await retrier.run(store.list_aliases)
    if alias in aliases:
        msg = f"alias {alias!r} is already bound to {aliases[alias]!r}"
        raise ConflictError(msg)
    await retrier.run(lambda: store.create_collection(collection, projector.spec))
    snapshot_lsn = await source.head_lsn()
    stream = source.stream_rows(snapshot_lsn=snapshot_lsn, after_id=None, batch_size=batch_size)
    async with contextlib.aclosing(stream) as batches:
        async for batch in batches:
            points = await projector.project(batch)
            await retrier.run(functools.partial(store.upsert, collection, points))
    await retrier.run(
        lambda: store.update_aliases([CreateAliasAction(alias=alias, collection=collection)])
    )
    return snapshot_lsn
