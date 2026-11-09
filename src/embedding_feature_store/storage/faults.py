"""Seeded chaos for any :class:`VectorStore`.

:class:`FaultInjectingStore` wraps a store and, with probability ``rate`` per
call, raises a transient fault instead of (or after) delegating:

* ``429`` — :class:`RateLimitedError`, raised *before* delegating (the server
  refused the request).
* ``timeout`` — :class:`StoreTimeoutError`. With probability
  ``ambiguous_fraction`` the call is delegated first and the timeout raised
  afterwards, modelling a response lost after the server committed. Callers
  must therefore be idempotent, exactly as against a real network.

All randomness comes from one ``random.Random(seed)``, so a given seed and call
sequence always produces the same faults.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

from embedding_feature_store.core.errors import (
    RateLimitedError,
    StoreTimeoutError,
    TransientError,
)
from embedding_feature_store.core.ports import VectorStore
from embedding_feature_store.schemas import (
    AliasAction,
    CollectionInfo,
    PointRecord,
    ScrollPage,
    VectorSpec,
)

__all__ = ["ALL_METHODS", "FaultInjectingStore", "FaultKind", "FaultPlan"]

type FaultKind = Literal["timeout", "rate_limit", "ambiguous_timeout"]

ALL_METHODS: Final = frozenset(
    {
        "collection_exists",
        "create_collection",
        "delete_collection",
        "describe_collection",
        "upsert",
        "delete_points",
        "count",
        "scroll",
        "list_aliases",
        "update_aliases",
    }
)


@dataclass(frozen=True, slots=True)
class FaultPlan:
    """What to inject. ``rate`` is per call; see the module docstring for kinds."""

    rate: float
    seed: int
    ambiguous_fraction: float = 0.0
    retry_after: float | None = None
    methods: frozenset[str] = ALL_METHODS

    def __post_init__(self) -> None:
        if not 0.0 <= self.rate <= 1.0 or not 0.0 <= self.ambiguous_fraction <= 1.0:
            msg = "rate and ambiguous_fraction must be within [0, 1]"
            raise ValueError(msg)
        unknown = self.methods - ALL_METHODS
        if unknown:
            msg = f"unknown methods: {sorted(unknown)}"
            raise ValueError(msg)


class FaultInjectingStore:
    def __init__(self, inner: VectorStore, plan: FaultPlan) -> None:
        self._inner = inner
        self._plan = plan
        self._rng = random.Random(plan.seed)  # noqa: S311 - chaos testing, not cryptography
        self.calls: Counter[str] = Counter()
        self.injected: Counter[tuple[str, FaultKind]] = Counter()

    @property
    def plan(self) -> FaultPlan:
        return self._plan

    @property
    def inner(self) -> VectorStore:
        return self._inner

    def _draw(self, method: str) -> FaultKind | None:
        self.calls[method] += 1
        if method not in self._plan.methods or self._rng.random() >= self._plan.rate:
            return None
        if self._rng.random() < 0.5:  # noqa: PLR2004 - even split between fault kinds
            kind: FaultKind = "rate_limit"
        elif self._rng.random() < self._plan.ambiguous_fraction:
            kind = "ambiguous_timeout"
        else:
            kind = "timeout"
        self.injected[method, kind] += 1
        return kind

    def _fault(self, method: str, kind: FaultKind) -> TransientError:
        if kind == "rate_limit":
            return RateLimitedError(f"injected 429 on {method}", retry_after=self._plan.retry_after)
        return StoreTimeoutError(f"injected {kind} on {method}")

    async def _call[R](self, method: str, operation: Callable[[], Awaitable[R]]) -> R:
        kind = self._draw(method)
        if kind is None:
            return await operation()
        if kind == "ambiguous_timeout":
            await operation()
        raise self._fault(method, kind)

    # -- VectorStore -----------------------------------------------------------

    async def collection_exists(self, name: str) -> bool:
        return await self._call("collection_exists", lambda: self._inner.collection_exists(name))

    async def create_collection(self, name: str, spec: VectorSpec) -> None:
        await self._call("create_collection", lambda: self._inner.create_collection(name, spec))

    async def delete_collection(self, name: str) -> None:
        await self._call("delete_collection", lambda: self._inner.delete_collection(name))

    async def describe_collection(self, name: str) -> CollectionInfo:
        return await self._call(
            "describe_collection", lambda: self._inner.describe_collection(name)
        )

    async def upsert(self, collection: str, points: Sequence[PointRecord]) -> None:
        await self._call("upsert", lambda: self._inner.upsert(collection, points))

    async def delete_points(self, collection: str, point_ids: Sequence[int]) -> None:
        await self._call("delete_points", lambda: self._inner.delete_points(collection, point_ids))

    async def count(self, collection: str) -> int:
        return await self._call("count", lambda: self._inner.count(collection))

    async def scroll(self, collection: str, *, offset: int | None, limit: int) -> ScrollPage:
        return await self._call(
            "scroll", lambda: self._inner.scroll(collection, offset=offset, limit=limit)
        )

    async def list_aliases(self) -> Mapping[str, str]:
        return await self._call("list_aliases", self._inner.list_aliases)

    async def update_aliases(self, actions: Sequence[AliasAction]) -> None:
        await self._call("update_aliases", lambda: self._inner.update_aliases(actions))
