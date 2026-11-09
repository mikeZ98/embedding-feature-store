"""Background production traffic for exercising a migration under load.

:class:`LiveTrafficSimulator` runs two loops until stopped:

* **reads** — ``count`` and a small ``scroll`` against the serving alias, the
  way an online service would query the active collection;
* **writes** — upserts and deletes against the source of record, each wrapped
  in ``WriteGate.write()``. During a freeze window writes are *held* (counted
  in ``writes_held``) and committed once the gate reopens; none are rejected.

The run is healthy when ``read_errors == 0`` and ``dropped == 0`` (every write
attempted was committed). Randomness is seeded for reproducibility.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
from dataclasses import dataclass
from types import TracebackType
from typing import Protocol, Self

from embedding_feature_store.core.errors import FeatureStoreError
from embedding_feature_store.core.ports import Sleep, VectorStore
from embedding_feature_store.core.write_gate import WriteGate
from embedding_feature_store.schemas import CdcEvent
from embedding_feature_store.telemetry.metrics import MigrationMetrics

__all__ = ["LiveTrafficSimulator", "SourceWriter", "TrafficProfile", "TrafficStats"]

_DELETE_SHARE = 0.2


class SourceWriter(Protocol):
    """The write side of the system of record (e.g. ``PostgresSimulator``)."""

    def upsert(self, record_id: int, tenant: str, features: tuple[float, ...]) -> CdcEvent: ...

    def delete(self, record_id: int) -> CdcEvent | None: ...


@dataclass(frozen=True, slots=True)
class TrafficProfile:
    """Shape of the simulated load."""

    alias: str
    feature_dim: int
    id_range: int
    seed: int
    read_interval_s: float = 0.002
    write_interval_s: float = 0.002
    tenant: str = "live"

    def __post_init__(self) -> None:
        if self.id_range < 1 or self.feature_dim < 1:
            msg = "id_range and feature_dim must be >= 1"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class TrafficStats:
    reads: int
    read_errors: int
    writes_attempted: int
    writes_committed: int
    writes_held: int
    write_errors: int

    @property
    def dropped(self) -> int:
        return self.writes_attempted - self.writes_committed - self.write_errors


class LiveTrafficSimulator:
    def __init__(
        self,
        store: VectorStore,
        source: SourceWriter,
        gate: WriteGate,
        metrics: MigrationMetrics,
        profile: TrafficProfile,
        *,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._store = store
        self._source = source
        self._gate = gate
        self._metrics = metrics
        self._profile = profile
        self._rng = random.Random(profile.seed)  # noqa: S311 - traffic simulation, not crypto
        self._sleep = sleep
        self._tasks: list[asyncio.Task[None]] = []
        self._reads = self._read_errors = 0
        self._attempted = self._committed = self._held = self._write_errors = 0
        self.last_read_error: str | None = None

    @property
    def stats(self) -> TrafficStats:
        return TrafficStats(
            reads=self._reads,
            read_errors=self._read_errors,
            writes_attempted=self._attempted,
            writes_committed=self._committed,
            writes_held=self._held,
            write_errors=self._write_errors,
        )

    async def read_once(self) -> None:
        self._reads += 1
        try:
            await self._store.count(self._profile.alias)
            await self._store.scroll(self._profile.alias, offset=None, limit=8)
        except FeatureStoreError as exc:
            self._read_errors += 1
            self.last_read_error = f"{type(exc).__name__}: {exc}"
            self._metrics.record_live_read(ok=False)
        else:
            self._metrics.record_live_read(ok=True)

    async def write_once(self) -> None:
        self._attempted += 1
        held = not self._gate.is_open
        record_id = self._rng.randint(1, self._profile.id_range)
        delete = self._rng.random() < _DELETE_SHARE
        features = tuple(self._rng.gauss(0.0, 1.0) for _ in range(self._profile.feature_dim))
        try:
            async with self._gate.write():
                await asyncio.sleep(0)  # the write is in flight across an await
                if delete:
                    self._source.delete(record_id)
                else:
                    self._source.upsert(record_id, self._profile.tenant, features)
        except FeatureStoreError:
            self._write_errors += 1
            self._metrics.record_live_write(ok=False, held=held)
            return
        self._committed += 1
        if held:
            self._held += 1
        self._metrics.record_live_write(ok=True, held=held)

    async def _loop_reads(self) -> None:
        while True:
            await self.read_once()
            await self._sleep(self._profile.read_interval_s)

    async def _loop_writes(self) -> None:
        while True:
            await self.write_once()
            await self._sleep(self._profile.write_interval_s)

    @property
    def running(self) -> bool:
        return bool(self._tasks)

    def start(self) -> None:
        if self._tasks:
            msg = "live traffic already running"
            raise RuntimeError(msg)
        self._tasks = [
            asyncio.create_task(self._loop_reads()),
            asyncio.create_task(self._loop_writes()),
        ]

    async def stop(self) -> TrafficStats:
        """Stop both loops; a write already inside the gate completes first."""
        tasks, self._tasks = self._tasks, []
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        return self.stats

    async def __aenter__(self) -> Self:
        self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.stop()
