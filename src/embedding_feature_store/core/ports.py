"""Hexagonal ports: the only surfaces through which the core touches the world.

Adapters raise errors from :mod:`embedding_feature_store.core.errors` — never
transport-specific exceptions — so that retry and rollback decisions can be
made on the transient/permanent classification alone.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from typing import Protocol

from embedding_feature_store.schemas import (
    AliasAction,
    CdcEvent,
    CollectionInfo,
    PointRecord,
    ScrollPage,
    SourceRecord,
    VectorSpec,
)

__all__ = [
    "Clock",
    "EmbeddingAdapter",
    "InFlightObserver",
    "QueueObserver",
    "Sleep",
    "SourceReader",
    "VectorStore",
]

type Clock = Callable[[], float]
"""Monotonic seconds. Never wall-clock time."""

type Sleep = Callable[[float], Awaitable[None]]
"""Suspend for the given number of seconds (``asyncio.sleep`` in production)."""

type QueueObserver = Callable[[int, int, bool], None]
"""Bounded-queue telemetry: ``(depth, capacity, producer_blocked)`` on every change."""

type InFlightObserver = Callable[[int], None]
"""Receives the number of in-flight operations whenever it changes."""


class VectorStore(Protocol):
    """A vector database holding collections of :class:`PointRecord` and aliases."""

    async def collection_exists(self, name: str) -> bool: ...

    async def create_collection(self, name: str, spec: VectorSpec) -> None: ...

    async def delete_collection(self, name: str) -> None: ...

    async def describe_collection(self, name: str) -> CollectionInfo: ...

    async def upsert(self, collection: str, points: Sequence[PointRecord]) -> None:
        """Idempotently write ``points``; replaying the same batch is harmless."""
        ...

    async def delete_points(self, collection: str, point_ids: Sequence[int]) -> None:
        """Idempotently delete ``point_ids``; missing ids are ignored."""
        ...

    async def count(self, collection: str) -> int:
        """Exact number of points in ``collection``."""
        ...

    async def scroll(self, collection: str, *, offset: int | None, limit: int) -> ScrollPage:
        """Points with id >= ``offset`` in ascending id order, at most ``limit``."""
        ...

    async def list_aliases(self) -> Mapping[str, str]:
        """Current ``alias -> collection`` bindings."""
        ...

    async def update_aliases(self, actions: Sequence[AliasAction]) -> None:
        """Apply ``actions`` atomically: all of them take effect or none do."""
        ...


class SourceReader(Protocol):
    """The system of record being migrated, readable as a snapshot plus a change log."""

    async def head_lsn(self) -> int:
        """The latest committed log sequence number."""
        ...

    async def estimate_rows(self) -> int:
        """Approximate row count, for progress reporting only."""
        ...

    def stream_rows(
        self, *, snapshot_lsn: int, after_id: int | None, batch_size: int
    ) -> AsyncGenerator[Sequence[SourceRecord]]:
        """Rows as of ``snapshot_lsn`` with id > ``after_id``, in ascending id batches.

        Consumers that may stop early must close the generator (e.g. with
        ``contextlib.aclosing``) so the underlying cursor is released promptly.
        """
        ...

    async def changes_since(self, lsn: int, *, limit: int) -> Sequence[CdcEvent]:
        """Up to ``limit`` events with lsn > ``lsn``, in ascending lsn order."""
        ...


class EmbeddingAdapter(Protocol):
    """Turns source feature vectors into embeddings of a fixed dimension."""

    @property
    def dimension(self) -> int: ...

    @property
    def model_id(self) -> str: ...

    async def embed(self, features: Sequence[Sequence[float]]) -> Sequence[Sequence[float]]:
        """One embedding per input, in input order."""
        ...
