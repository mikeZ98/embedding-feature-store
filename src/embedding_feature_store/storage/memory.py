"""In-process :class:`VectorStore` with Qdrant's observable semantics.

Used for local runs, benchmarks and end-to-end tests where no Qdrant server is
available. Mirrors the behaviours the pipeline relies on:

* collection names in data-plane calls may be **aliases** and are resolved;
* ``update_aliases`` validates every action first and applies all or none;
* deleting a collection also removes the aliases bound to it;
* upserting a vector of the wrong dimension is rejected (HTTP 400 equivalent);
* creating an existing collection is a conflict (HTTP 409 equivalent).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from embedding_feature_store.core.errors import (
    ConflictError,
    NotFoundError,
    ValidationFailedError,
)
from embedding_feature_store.schemas import (
    AliasAction,
    CollectionInfo,
    CreateAliasAction,
    PointRecord,
    ScrollPage,
    VectorSpec,
)

__all__ = ["InMemoryVectorStore"]


@dataclass
class _Collection:
    spec: VectorSpec
    points: dict[int, PointRecord] = field(default_factory=dict)


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._collections: dict[str, _Collection] = {}
        self._aliases: dict[str, str] = {}

    def _resolve(self, name: str) -> _Collection:
        collection = self._collections.get(self._aliases.get(name, name))
        if collection is None:
            msg = f"collection {name!r} does not exist"
            raise NotFoundError(msg)
        return collection

    async def collection_exists(self, name: str) -> bool:
        return self._aliases.get(name, name) in self._collections

    async def create_collection(self, name: str, spec: VectorSpec) -> None:
        if name in self._collections or name in self._aliases:
            msg = f"collection {name!r} already exists"
            raise ConflictError(msg)
        self._collections[name] = _Collection(spec)

    async def delete_collection(self, name: str) -> None:
        if self._collections.pop(name, None) is not None:
            self._aliases = {a: c for a, c in self._aliases.items() if c != name}

    async def describe_collection(self, name: str) -> CollectionInfo:
        collection = self._resolve(name)
        return CollectionInfo(name=name, spec=collection.spec, points_count=len(collection.points))

    async def upsert(self, collection: str, points: Sequence[PointRecord]) -> None:
        target = self._resolve(collection)
        for point in points:
            if len(point.vector) != target.spec.dimension:
                msg = (
                    f"point {point.point_id}: dimension {len(point.vector)} "
                    f"!= {target.spec.dimension} in {collection!r}"
                )
                raise ValidationFailedError(msg)
        for point in points:
            target.points[point.point_id] = point

    async def delete_points(self, collection: str, point_ids: Sequence[int]) -> None:
        target = self._resolve(collection)
        for point_id in point_ids:
            target.points.pop(point_id, None)

    async def count(self, collection: str) -> int:
        return len(self._resolve(collection).points)

    async def scroll(self, collection: str, *, offset: int | None, limit: int) -> ScrollPage:
        points = self._resolve(collection).points
        ids = sorted(i for i in points if offset is None or i >= offset)
        page, rest = ids[:limit], ids[limit:]
        return ScrollPage(
            points=tuple(points[i] for i in page), next_offset=rest[0] if rest else None
        )

    async def list_aliases(self) -> Mapping[str, str]:
        return dict(self._aliases)

    async def update_aliases(self, actions: Sequence[AliasAction]) -> None:
        staged = dict(self._aliases)
        for action in actions:
            if isinstance(action, CreateAliasAction):
                if action.collection not in self._collections:
                    msg = f"cannot alias missing collection {action.collection!r}"
                    raise NotFoundError(msg)
                staged[action.alias] = action.collection
            elif staged.pop(action.alias, None) is None:
                msg = f"alias {action.alias!r} does not exist"
                raise NotFoundError(msg)
        self._aliases = staged
