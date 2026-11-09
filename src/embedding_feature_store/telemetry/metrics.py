"""Migration telemetry registry.

Concurrency model: **single writer, lock-free.** Every mutator is a plain,
non-awaiting method intended to be called on the event-loop thread, so on one
loop no two updates can interleave and no lock is needed. :meth:`snapshot`
copies the state into a frozen :class:`TelemetrySnapshot` made only of
immutable values, which may then be handed to any thread (e.g. a renderer).
Code running in worker threads must publish through
``loop.call_soon_threadsafe(metrics.record_batch, n)`` rather than mutate
directly.

Hooks are shaped to plug straight into the rest of the system:
:meth:`MigrationMetrics.observe_queue` is a ``QueueObserver`` and
:meth:`MigrationMetrics.on_retry` is a ``Retrier`` ``OnRetry`` callback.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

from embedding_feature_store.core.errors import TransientError
from embedding_feature_store.core.ports import Clock
from embedding_feature_store.schemas import (
    DistanceMetric,
    MigrationPhase,
    MigrationPlan,
    MigrationState,
    VerificationReport,
)

__all__ = ["MigrationMetrics", "SlidingWindowRate", "TelemetrySnapshot", "VerificationSummary"]

_CHECKSUM_PREFIX = 12


class SlidingWindowRate:
    """Events per second over the trailing ``horizon_s`` seconds.

    ``rate()`` is ``sum(counts in (now - horizon, now]) / horizon``: it ramps up
    over the first horizon, tracks steady load exactly once the window is full,
    and decays to zero once no event has been recorded for a full horizon.
    Memory is bounded by the number of ``record`` calls within one horizon.
    """

    def __init__(self, horizon_s: float, *, clock: Clock = time.monotonic) -> None:
        if not horizon_s > 0:
            msg = "horizon_s must be > 0"
            raise ValueError(msg)
        self._horizon = horizon_s
        self._clock = clock
        self._events: deque[tuple[float, int]] = deque()
        self._in_window = 0
        self._total = 0

    @property
    def horizon_s(self) -> float:
        return self._horizon

    @property
    def total(self) -> int:
        return self._total

    def _expire(self, now: float) -> None:
        cutoff = now - self._horizon
        while self._events and self._events[0][0] <= cutoff:
            self._in_window -= self._events.popleft()[1]

    def record(self, count: int = 1) -> None:
        if count < 0:
            msg = "count must be >= 0"
            raise ValueError(msg)
        now = self._clock()
        self._expire(now)
        self._events.append((now, count))
        self._in_window += count
        self._total += count

    def rate(self) -> float:
        self._expire(self._clock())
        return self._in_window / self._horizon


@dataclass(frozen=True, slots=True)
class VerificationSummary:
    passed: bool
    source_count: int
    target_count: int
    source_checksum_prefix: str
    target_checksum_prefix: str

    @classmethod
    def of(cls, report: VerificationReport) -> VerificationSummary:
        return cls(
            passed=report.passed,
            source_count=report.source_count,
            target_count=report.target_count,
            source_checksum_prefix=report.source_checksum[:_CHECKSUM_PREFIX],
            target_checksum_prefix=report.target_checksum[:_CHECKSUM_PREFIX],
        )


@dataclass(frozen=True, slots=True)
class TelemetrySnapshot:
    """An immutable, instantaneous view of a migration's telemetry."""

    # identity / lifecycle
    migration_id: str
    source_collection: str
    target_collection: str
    alias: str
    dimension: int
    distance: DistanceMetric
    phase: MigrationPhase
    error: str | None
    elapsed_s: float
    # backfill
    estimated_total: int | None
    backfilled: int
    batches: int
    retries: int
    last_retry: str | None
    throughput_pps: float
    queue_depth: int
    queue_capacity: int
    backpressure_stalls: int
    # CDC
    applied_lsn: int
    source_head_lsn: int
    cdc_events: int
    # dimension checks
    vectors_verified: int
    dimension_violations: int
    # live traffic
    live_reads: int
    live_read_errors: int
    live_writes: int
    live_write_errors: int
    writes_held: int
    # latency
    drain_ms: float | None
    cutover_ms: float | None
    # verification
    verification: VerificationSummary | None

    @property
    def progress(self) -> float | None:
        """Backfill fraction in [0, 1], or None while the total is unknown."""
        if not self.estimated_total:
            return None
        return min(1.0, self.backfilled / self.estimated_total)

    @property
    def cdc_lag(self) -> int:
        return max(0, self.source_head_lsn - self.applied_lsn)


class MigrationMetrics:
    """Mutable registry; see the module docstring for the concurrency model."""

    def __init__(
        self,
        plan: MigrationPlan,
        *,
        clock: Clock = time.monotonic,
        rate_horizon_s: float = 5.0,
    ) -> None:
        self._plan = plan
        self._clock = clock
        self._started = clock()
        self._throughput = SlidingWindowRate(rate_horizon_s, clock=clock)
        self._phase = MigrationPhase.PENDING
        self._error: str | None = None
        self._estimated_total: int | None = None
        self._batches = 0
        self._retries = 0
        self._last_retry: str | None = None
        self._queue_depth = 0
        self._queue_capacity = 0
        self._stalls = 0
        self._applied_lsn = 0
        self._head_lsn = 0
        self._cdc_events = 0
        self._verified = 0
        self._violations = 0
        self._reads = 0
        self._read_errors = 0
        self._writes = 0
        self._write_errors = 0
        self._held = 0
        self._drain_ms: float | None = None
        self._cutover_ms: float | None = None
        self._verification: VerificationSummary | None = None

    # -- lifecycle -------------------------------------------------------------

    def set_phase(self, phase: MigrationPhase, *, error: str | None = None) -> None:
        self._phase = phase
        self._error = error

    def observe_state(self, state: MigrationState) -> None:
        self.set_phase(state.phase, error=state.error)
        self._applied_lsn = max(self._applied_lsn, state.applied_lsn)

    # -- backfill --------------------------------------------------------------

    def set_estimated_total(self, total: int) -> None:
        self._estimated_total = max(0, total)

    def record_batch(self, points: int) -> None:
        self._batches += 1
        self._throughput.record(points)

    def on_retry(self, attempt: int, delay: float, error: TransientError) -> None:
        """``OnRetry`` hook for :class:`~embedding_feature_store.core.retry.Retrier`."""
        self._retries += 1
        self._last_retry = f"#{attempt} in {delay:.2f}s: {type(error).__name__}: {error}"

    def observe_queue(self, depth: int, capacity: int, blocked: bool) -> None:
        """``QueueObserver`` hook for bounded streams."""
        self._queue_depth = depth
        self._queue_capacity = capacity
        if blocked:
            self._stalls += 1

    # -- CDC -------------------------------------------------------------------

    def set_source_head(self, lsn: int) -> None:
        self._head_lsn = max(self._head_lsn, lsn)

    def record_cdc(self, applied_lsn: int, events: int) -> None:
        self._applied_lsn = max(self._applied_lsn, applied_lsn)
        self._cdc_events += events

    # -- dimension checks ------------------------------------------------------

    def record_dimension_check(self, *, verified: int, violations: int = 0) -> None:
        self._verified += verified
        self._violations += violations

    # -- live traffic ----------------------------------------------------------

    def record_live_read(self, *, ok: bool) -> None:
        self._reads += 1
        if not ok:
            self._read_errors += 1

    def record_live_write(self, *, ok: bool, held: bool = False) -> None:
        self._writes += 1
        if not ok:
            self._write_errors += 1
        if held:
            self._held += 1

    # -- latency & verification ------------------------------------------------

    def record_drain(self, seconds: float) -> None:
        self._drain_ms = seconds * 1000.0

    def record_cutover(self, seconds: float) -> None:
        self._cutover_ms = seconds * 1000.0

    def record_verification(self, report: VerificationReport) -> None:
        self._verification = VerificationSummary.of(report)

    # -- read side -------------------------------------------------------------

    def snapshot(self) -> TelemetrySnapshot:
        plan = self._plan
        return TelemetrySnapshot(
            migration_id=plan.migration_id,
            source_collection=plan.source_collection,
            target_collection=plan.target_collection,
            alias=plan.alias,
            dimension=plan.target_spec.dimension,
            distance=plan.target_spec.distance,
            phase=self._phase,
            error=self._error,
            elapsed_s=self._clock() - self._started,
            estimated_total=self._estimated_total,
            backfilled=self._throughput.total,
            batches=self._batches,
            retries=self._retries,
            last_retry=self._last_retry,
            throughput_pps=self._throughput.rate(),
            queue_depth=self._queue_depth,
            queue_capacity=self._queue_capacity,
            backpressure_stalls=self._stalls,
            applied_lsn=self._applied_lsn,
            source_head_lsn=self._head_lsn,
            cdc_events=self._cdc_events,
            vectors_verified=self._verified,
            dimension_violations=self._violations,
            live_reads=self._reads,
            live_read_errors=self._read_errors,
            live_writes=self._writes,
            live_write_errors=self._write_errors,
            writes_held=self._held,
            drain_ms=self._drain_ms,
            cutover_ms=self._cutover_ms,
            verification=self._verification,
        )
