"""In-memory test doubles for the core ports, with fault injection."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from embedding_feature_store.core.errors import NotFoundError
from embedding_feature_store.core.write_gate import WriteGate
from embedding_feature_store.schemas import (
    AliasAction,
    CdcEvent,
    CollectionInfo,
    CreateAliasAction,
    PointRecord,
    ScrollPage,
    SourceRecord,
    VectorSpec,
)

type AliasFaultMode = Literal["raise_before", "raise_after", "partial", "noop"]


@dataclass
class AliasFault:
    """How the next ``update_aliases`` call misbehaves.

    raise_before: raise without applying.  raise_after: apply, then raise.
    partial: apply only the first action, then raise (a non-atomic store).
    noop: accept silently without applying.
    """

    mode: AliasFaultMode
    error: Exception | None = None


@dataclass
class AliasCall:
    actions: tuple[AliasAction, ...]
    gate_open: bool | None
    in_flight: int | None


@dataclass
class InMemoryVectorStore:
    collections: dict[str, tuple[VectorSpec, dict[int, PointRecord]]] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)
    alias_faults: list[AliasFault] = field(default_factory=list)
    list_alias_errors: list[Exception] = field(default_factory=list)
    alias_calls: list[AliasCall] = field(default_factory=list)
    observed_gate: WriteGate | None = None
    on_update_aliases: Callable[[], None] | None = None

    def _points(self, name: str) -> dict[int, PointRecord]:
        if name not in self.collections:
            raise NotFoundError(name)
        return self.collections[name][1]

    async def collection_exists(self, name: str) -> bool:
        return name in self.collections

    async def create_collection(self, name: str, spec: VectorSpec) -> None:
        self.collections[name] = (spec, {})

    async def delete_collection(self, name: str) -> None:
        self.collections.pop(name, None)

    async def describe_collection(self, name: str) -> CollectionInfo:
        spec, points = self.collections[name]
        return CollectionInfo(name=name, spec=spec, points_count=len(points))

    async def upsert(self, collection: str, points: Sequence[PointRecord]) -> None:
        store = self._points(collection)
        for point in points:
            store[point.point_id] = point

    async def delete_points(self, collection: str, point_ids: Sequence[int]) -> None:
        store = self._points(collection)
        for pid in point_ids:
            store.pop(pid, None)

    async def count(self, collection: str) -> int:
        return len(self._points(collection))

    async def scroll(self, collection: str, *, offset: int | None, limit: int) -> ScrollPage:
        ids = sorted(pid for pid in self._points(collection) if offset is None or pid >= offset)
        page, rest = ids[:limit], ids[limit:]
        return ScrollPage(
            points=tuple(self._points(collection)[pid] for pid in page),
            next_offset=rest[0] if rest else None,
        )

    async def list_aliases(self) -> Mapping[str, str]:
        if self.list_alias_errors:
            raise self.list_alias_errors.pop(0)
        return dict(self.aliases)

    def _apply(self, actions: Sequence[AliasAction]) -> None:
        for action in actions:
            if isinstance(action, CreateAliasAction):
                self.aliases[action.alias] = action.collection
            else:
                self.aliases.pop(action.alias, None)

    async def update_aliases(self, actions: Sequence[AliasAction]) -> None:
        gate = self.observed_gate
        self.alias_calls.append(
            AliasCall(
                actions=tuple(actions),
                gate_open=None if gate is None else gate.is_open,
                in_flight=None if gate is None else gate.in_flight,
            )
        )
        if self.on_update_aliases is not None:
            self.on_update_aliases()
        fault = self.alias_faults.pop(0) if self.alias_faults else None
        if fault is None:
            self._apply(actions)
            return
        if fault.mode == "raise_after":
            self._apply(actions)
        elif fault.mode == "partial":
            self._apply(actions[:1])
        elif fault.mode == "noop":
            return
        assert fault.error is not None
        raise fault.error


@dataclass
class FakeEmbedder:
    dimension: int
    model_id: str = "fake-model@1"
    transform: Callable[[Sequence[float]], Sequence[float]] = tuple
    calls: int = 0
    drop_last: bool = False

    async def embed(self, features: Sequence[Sequence[float]]) -> Sequence[Sequence[float]]:
        self.calls += 1
        out = [self.transform(f) for f in features]
        return out[:-1] if self.drop_last else out


@dataclass
class FakeSource:
    rows: list[SourceRecord] = field(default_factory=list)
    events: list[CdcEvent] = field(default_factory=list)

    async def head_lsn(self) -> int:
        return max((e.lsn for e in self.events), default=0)

    async def estimate_rows(self) -> int:
        return len(self.rows)

    async def stream_rows(
        self, *, snapshot_lsn: int, after_id: int | None, batch_size: int
    ) -> AsyncIterator[Sequence[SourceRecord]]:
        eligible = [
            r
            for r in sorted(self.rows, key=lambda r: r.record_id)
            if r.lsn <= snapshot_lsn and (after_id is None or r.record_id > after_id)
        ]
        for start in range(0, len(eligible), batch_size):
            yield eligible[start : start + batch_size]

    async def changes_since(self, lsn: int, *, limit: int) -> Sequence[CdcEvent]:
        return [e for e in self.events if e.lsn > lsn][:limit]


class RecordingSleep:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)
