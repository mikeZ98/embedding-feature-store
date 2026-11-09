"""Metrics registry, sliding-window rates, HUD rendering and the live HUD lifecycle."""

from __future__ import annotations

import asyncio
import dataclasses
import io
import json
from collections.abc import AsyncGenerator

import pytest
from hypothesis import given
from hypothesis import strategies as st
from rich.console import Console
from rich.panel import Panel
from rich.style import Style

from embedding_feature_store.core.errors import StoreTimeoutError
from embedding_feature_store.core.ports import QueueObserver
from embedding_feature_store.core.retry import OnRetry, Retrier
from embedding_feature_store.core.state_machine import advance, fail
from embedding_feature_store.schemas import (
    DistanceMetric,
    MigrationPhase,
    MigrationPlan,
    MigrationState,
    RetryPolicy,
    VectorSpec,
    VerificationReport,
)
from embedding_feature_store.storage.bounded_stream import bounded_stream
from embedding_feature_store.telemetry.hud import TelemetryHud, phase_style, render_hud
from embedding_feature_store.telemetry.metrics import (
    MigrationMetrics,
    SlidingWindowRate,
    TelemetrySnapshot,
)
from tests.fakes import RecordingSleep

P = MigrationPhase
PLAN = MigrationPlan(
    migration_id="mig-42",
    source_collection="feat_v1",
    target_collection="feat_v2",
    alias="features",
    target_spec=VectorSpec(dimension=768, distance=DistanceMetric.COSINE),
)


class FakeClock:
    def __init__(self, start: float = 100.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def report(*, passed: bool) -> VerificationReport:
    return VerificationReport(
        passed=passed,
        source_count=10,
        target_count=10 if passed else 9,
        source_checksum="ab" * 32,
        target_checksum="ab" * 32 if passed else "cd" * 32,
    )


def render_text(renderable: Panel, *, width: int = 120) -> str:
    console = Console(file=io.StringIO(), width=width, record=True, color_system=None)
    console.print(renderable)
    return console.export_text()


def styles_of(renderable: Panel, text: str) -> list[Style]:
    console = Console(file=io.StringIO(), width=120, color_system="truecolor", force_terminal=True)
    return [
        seg.style
        for seg in console.render(renderable)
        if seg.style is not None and text in seg.text
    ]


# ---------------------------------------------------------------------------
# SlidingWindowRate
# ---------------------------------------------------------------------------


class TestSlidingWindowRate:
    @pytest.mark.parametrize("horizon", [0.0, -1.0])
    def test_horizon_validated(self, horizon: float) -> None:
        with pytest.raises(ValueError, match="horizon_s"):
            SlidingWindowRate(horizon)

    def test_negative_count_rejected(self) -> None:
        with pytest.raises(ValueError, match="count"):
            SlidingWindowRate(1.0).record(-1)

    @pytest.mark.parametrize("horizon", [1.0, 5.0])
    def test_stationary_rate_is_exact_once_window_full(self, horizon: float) -> None:
        clock = FakeClock()
        rate = SlidingWindowRate(horizon, clock=clock)
        for _ in range(int(horizon * 8) * 3):  # 3 horizons of 125 pts every 0.125 s
            clock.advance(0.125)  # binary-exact step: no float drift at the window edge
            rate.record(125)
        assert rate.horizon_s == horizon
        assert rate.rate() == pytest.approx(1000.0)

    def test_ramps_up_during_first_horizon(self) -> None:
        clock = FakeClock()
        rate = SlidingWindowRate(1.0, clock=clock)
        for _ in range(5):
            clock.advance(0.1)
            rate.record(100)
        assert rate.rate() == pytest.approx(500.0)

    def test_burst_then_idle_decay(self) -> None:
        clock = FakeClock()
        rate = SlidingWindowRate(2.0, clock=clock)
        rate.record(1000)
        assert rate.rate() == 500.0
        clock.advance(1.999)
        assert rate.rate() == 500.0
        clock.advance(0.001)
        assert rate.rate() == 0.0, "an event exactly one horizon old has left the window"
        assert rate.total == 1000

    def test_partial_expiry(self) -> None:
        clock = FakeClock()
        rate = SlidingWindowRate(1.0, clock=clock)
        rate.record(10)
        clock.advance(0.5)
        rate.record(30)
        clock.advance(0.6)
        assert rate.rate() == 30.0

    @given(st.lists(st.tuples(st.floats(0, 3), st.integers(0, 1000)), max_size=50))
    def test_matches_brute_force(self, steps: list[tuple[float, int]]) -> None:
        clock = FakeClock(0.0)
        rate = SlidingWindowRate(1.5, clock=clock)
        history: list[tuple[float, int]] = []
        for dt, n in steps:
            clock.advance(dt)
            rate.record(n)
            history.append((clock.now, n))
            expected = sum(c for t, c in history if t > clock.now - 1.5) / 1.5
            assert rate.rate() == pytest.approx(expected)
        assert rate.total == sum(n for _, n in steps)


# ---------------------------------------------------------------------------
# MigrationMetrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_initial_snapshot(self) -> None:
        s = MigrationMetrics(PLAN, clock=FakeClock()).snapshot()
        assert (s.phase, s.error, s.backfilled, s.progress, s.verification) == (
            P.PENDING,
            None,
            0,
            None,
            None,
        )
        assert (s.dimension, s.distance, s.alias) == (768, DistanceMetric.COSINE, "features")
        assert (s.drain_ms, s.cutover_ms) == (None, None)

    def test_hooks_conform_to_ports(self) -> None:
        metrics = MigrationMetrics(PLAN)
        observer: QueueObserver = metrics.observe_queue
        hook: OnRetry = metrics.on_retry
        observer(2, 4, False)
        hook(1, 0.5, StoreTimeoutError("t"))
        s = metrics.snapshot()
        assert (s.queue_depth, s.queue_capacity, s.retries) == (2, 4, 1)
        assert s.last_retry == "#1 in 0.50s: StoreTimeoutError: t"

    def test_backfill_progress_and_throughput(self) -> None:
        clock = FakeClock()
        metrics = MigrationMetrics(PLAN, clock=clock, rate_horizon_s=1.0)
        metrics.set_estimated_total(1000)
        for _ in range(10):
            clock.advance(0.1)
            metrics.record_batch(50)
        s = metrics.snapshot()
        assert (s.backfilled, s.batches) == (500, 10)
        assert s.progress == 0.5
        assert s.throughput_pps == pytest.approx(500.0)
        assert s.elapsed_s == pytest.approx(1.0)

    def test_progress_clamped_and_unknown_total(self) -> None:
        metrics = MigrationMetrics(PLAN)
        metrics.record_batch(30)
        metrics.set_estimated_total(20)
        assert metrics.snapshot().progress == 1.0
        metrics.set_estimated_total(-5)
        assert metrics.snapshot().estimated_total == 0
        assert metrics.snapshot().progress is None

    def test_backpressure_stalls_counted(self) -> None:
        metrics = MigrationMetrics(PLAN)
        for blocked in (False, True, False, True):
            metrics.observe_queue(3, 3, blocked)
        assert metrics.snapshot().backpressure_stalls == 2

    def test_cdc_lag_and_monotonic_lsns(self) -> None:
        metrics = MigrationMetrics(PLAN)
        metrics.set_source_head(100)
        metrics.record_cdc(60, events=5)
        metrics.record_cdc(40, events=2)  # out-of-order report never moves backwards
        metrics.set_source_head(90)
        s = metrics.snapshot()
        assert (s.applied_lsn, s.source_head_lsn, s.cdc_events, s.cdc_lag) == (60, 100, 7, 40)
        metrics.record_cdc(120, events=1)
        assert metrics.snapshot().cdc_lag == 0

    def test_dimension_live_traffic_latency_verification(self) -> None:
        metrics = MigrationMetrics(PLAN)
        metrics.record_dimension_check(verified=100)
        metrics.record_dimension_check(verified=10, violations=2)
        metrics.record_live_read(ok=True)
        metrics.record_live_read(ok=False)
        metrics.record_live_write(ok=True, held=True)
        metrics.record_live_write(ok=False)
        metrics.record_drain(0.0125)
        metrics.record_cutover(0.25)
        metrics.record_verification(report(passed=False))
        s = metrics.snapshot()
        assert (s.vectors_verified, s.dimension_violations) == (110, 2)
        assert (s.live_reads, s.live_read_errors, s.live_writes, s.live_write_errors) == (
            2,
            1,
            2,
            1,
        )
        assert s.writes_held == 1
        assert (s.drain_ms, s.cutover_ms) == (12.5, 250.0)
        assert s.verification is not None
        assert not s.verification.passed
        assert (s.verification.source_checksum_prefix, s.verification.target_count) == ("ab" * 6, 9)

    def test_observe_state_tracks_phase_error_and_lsn(self) -> None:
        metrics = MigrationMetrics(PLAN)
        state = evolve_lsn(advance(MigrationState(plan=PLAN)), 33)
        metrics.observe_state(state)
        assert (metrics.snapshot().phase, metrics.snapshot().applied_lsn) == (P.PROVISIONING, 33)
        metrics.observe_state(fail(MigrationState(plan=PLAN), "disk full"))
        s = metrics.snapshot()
        assert (s.phase, s.error, s.applied_lsn) == (P.FAILED, "disk full", 33)

    @pytest.mark.asyncio
    async def test_wired_into_retrier_and_bounded_stream(self) -> None:
        metrics = MigrationMetrics(PLAN)
        failures = [StoreTimeoutError("t"), StoreTimeoutError("t")]

        async def flaky() -> int:
            if failures:
                raise failures.pop()
            return 1

        retrier = Retrier(RetryPolicy(), sleep=RecordingSleep(), seed=0, on_retry=metrics.on_retry)
        await retrier.run(flaky)

        async def source() -> AsyncGenerator[int]:
            for i in range(10):
                yield i

        stream = bounded_stream(source(), capacity=2, observer=metrics.observe_queue)
        first = await anext(stream)
        for _ in range(10):
            await asyncio.sleep(0)
        rest = [i async for i in stream]
        s = metrics.snapshot()
        assert [first, *rest] == list(range(10))
        assert s.retries == 2
        assert s.queue_capacity == 2
        assert s.backpressure_stalls >= 1


def evolve_lsn(state: MigrationState, lsn: int) -> MigrationState:
    from embedding_feature_store.schemas import evolve  # noqa: PLC0415

    return evolve(state, applied_lsn=lsn)


class TestSnapshotImmutability:
    def test_frozen(self) -> None:
        snapshot = MigrationMetrics(PLAN).snapshot()
        with pytest.raises(dataclasses.FrozenInstanceError):
            snapshot.backfilled = 5  # type: ignore[misc]

    @pytest.mark.asyncio
    async def test_snapshot_unaffected_by_concurrent_updates(self) -> None:
        clock = FakeClock()
        metrics = MigrationMetrics(PLAN, clock=clock)
        metrics.set_estimated_total(10_000)
        metrics.record_batch(10)
        metrics.record_verification(report(passed=True))
        frozen = metrics.snapshot()
        before = dataclasses.astuple(frozen)

        async def writer(i: int) -> None:
            for _ in range(50):
                metrics.record_batch(i)
                metrics.observe_queue(i % 4, 4, i % 2 == 0)
                metrics.record_cdc(i, events=1)
                metrics.record_live_write(ok=i % 3 != 0, held=True)
                metrics.record_verification(report(passed=False))
                clock.advance(0.001)
                await asyncio.sleep(0)

        async def reader() -> list[TelemetrySnapshot]:
            seen = []
            for _ in range(50):
                seen.append(metrics.snapshot())
                await asyncio.sleep(0)
            return seen

        results = await asyncio.gather(*(writer(i) for i in range(1, 9)), reader())
        assert dataclasses.astuple(frozen) == before
        assert frozen.verification is not None
        assert frozen.verification.passed
        snapshots: list[TelemetrySnapshot] = results[-1]  # type: ignore[assignment]
        backfilled = [s.backfilled for s in snapshots]
        assert backfilled == sorted(backfilled), "each snapshot is a consistent point in time"
        assert metrics.snapshot().backfilled == 10 + 50 * sum(range(1, 9))

    @pytest.mark.asyncio
    async def test_snapshot_renders_in_worker_thread_while_loop_mutates(self) -> None:
        metrics = MigrationMetrics(PLAN)
        snapshot = metrics.snapshot()
        rendering = asyncio.create_task(asyncio.to_thread(render_text, render_hud(snapshot)))
        for _ in range(200):
            metrics.record_batch(1)
            await asyncio.sleep(0)
        text = await rendering
        assert "0 points backfilled" in text


# ---------------------------------------------------------------------------
# render_hud
# ---------------------------------------------------------------------------


def snapshot_for(
    phase: MigrationPhase, *, error: str | None = None, **changes: object
) -> TelemetrySnapshot:
    metrics = MigrationMetrics(PLAN, clock=FakeClock())
    metrics.set_phase(phase, error=error)
    return dataclasses.replace(metrics.snapshot(), **changes)  # type: ignore[arg-type]


class TestRenderHud:
    @pytest.mark.parametrize(
        ("phase", "style"),
        [
            (P.PENDING, "bold black on grey70"),
            (P.BACKFILLING, "bold white on blue"),
            (P.RECONCILING, "bold white on blue"),
            (P.VERIFYING, "bold white on blue"),
            (P.DRAINING, "bold white on magenta"),
            (P.CUTTING_OVER, "bold white on magenta"),
            (P.COMPLETED, "bold white on green"),
            (P.FAILED, "bold white on red"),
            (P.ROLLED_BACK, "bold black on yellow"),
            (P.ABORTED, "bold black on yellow"),
        ],
    )
    def test_phase_badge(self, phase: MigrationPhase, style: str) -> None:
        assert phase_style(phase) == style
        label = f" {phase.value.upper().replace('_', ' ')} "
        panel = render_hud(snapshot_for(phase, error="x" if phase is P.FAILED else None))
        assert Style.parse(style) in styles_of(panel, label)
        assert label in render_text(panel)

    @pytest.mark.parametrize(
        ("phase", "border"),
        [(P.COMPLETED, "green"), (P.FAILED, "red"), (P.ABORTED, "yellow"), (P.BACKFILLING, "blue")],
    )
    def test_panel_border(self, phase: MigrationPhase, border: str) -> None:
        assert render_hud(snapshot_for(phase)).border_style == border

    def test_header_identity(self) -> None:
        text = render_text(render_hud(snapshot_for(P.BACKFILLING)))
        assert "mig-42" in text
        assert "feat_v1 → feat_v2" in text
        assert "alias features" in text
        assert "dim 768 · cosine" in text

    def test_progress_known_total(self) -> None:
        text = render_text(
            render_hud(snapshot_for(P.BACKFILLING, estimated_total=8000, backfilled=2000))
        )
        assert "25.0%  2,000 / 8,000 points" in text

    def test_progress_unknown_total(self) -> None:
        text = render_text(render_hud(snapshot_for(P.BACKFILLING, backfilled=1234)))
        assert "1,234 points backfilled (total unknown)" in text

    def test_metric_grid_contents(self) -> None:
        snap = snapshot_for(
            P.CUTTING_OVER,
            throughput_pps=1234.5,
            batches=12,
            retries=3,
            queue_depth=4,
            queue_capacity=4,
            backpressure_stalls=7,
            vectors_verified=900,
            applied_lsn=5000,
            source_head_lsn=5040,
            cdc_events=321,
            live_reads=10,
            live_writes=20,
            live_write_errors=1,
            writes_held=6,
            drain_ms=12.34,
            cutover_ms=56.7,
            elapsed_s=61.25,
            last_retry="#1 in 0.10s: StoreTimeoutError: t",
        )
        text = render_text(render_hud(snap))
        for fragment in [
            "1,234.5 pts/s",
            "12 · 3",
            "4/4",
            "7 stalls",
            "900 ok · 0 bad",
            "lsn 5,000 · 321 ev",
            "lag 40",
            "10 / 20",
            "1 err · 6 held",
            "12.3 ms · 56.7 ms",
            "61.2 s",
            "StoreTimeoutError",
        ]:
            assert fragment in text, fragment

    def test_latency_placeholders(self) -> None:
        assert "— · —" in render_text(render_hud(snapshot_for(P.BACKFILLING)))

    def test_alarm_highlighting(self) -> None:
        calm = render_hud(snapshot_for(P.BACKFILLING, vectors_verified=5, queue_capacity=4))
        alarmed = render_hud(
            snapshot_for(
                P.BACKFILLING,
                vectors_verified=5,
                dimension_violations=2,
                queue_depth=4,
                queue_capacity=4,
                live_read_errors=1,
                source_head_lsn=9,
            )
        )
        red, yellow, bold = Style.parse("bold red"), Style.parse("bold yellow"), Style.parse("bold")
        assert red in styles_of(alarmed, "2 bad")
        assert red not in styles_of(calm, "0 bad")
        assert yellow in styles_of(alarmed, "4/4")
        assert yellow in styles_of(alarmed, "lag 9")
        assert red in styles_of(alarmed, "1 err")
        assert bold in styles_of(calm, "0/4")

    def test_verification_badges(self) -> None:
        none = render_text(render_hud(snapshot_for(P.VERIFYING)))
        assert "NOT VERIFIED" in none

        metrics = MigrationMetrics(PLAN)
        metrics.record_verification(report(passed=True))
        ok = render_hud(metrics.snapshot())
        assert Style.parse("bold white on green") in styles_of(ok, "VERIFIED ✓")
        assert "source 10 [abababababab]" in render_text(ok)

        metrics.record_verification(report(passed=False))
        bad = render_hud(metrics.snapshot())
        assert Style.parse("bold white on red") in styles_of(bad, "MISMATCH ✗")
        assert "target 9 [cdcdcdcdcdcd]" in render_text(bad)

    def test_error_panel_only_when_error(self) -> None:
        failed = render_hud(snapshot_for(P.FAILED, error="alias swap rejected: 503"))
        text = render_text(failed)
        assert "error" in text
        assert "alias swap rejected: 503" in text
        assert Style.parse("bold red") in styles_of(failed, "alias swap rejected")
        assert "alias swap rejected" not in render_text(render_hud(snapshot_for(P.COMPLETED)))

    @pytest.mark.parametrize(("width", "columns"), [(140, 3), (70, 2)])
    def test_grid_adapts_to_width(self, width: int, columns: int) -> None:
        text = render_text(render_hud(snapshot_for(P.BACKFILLING)), width=width)
        first_row = next(line for line in text.splitlines() if "THROUGHPUT" in line)
        assert ("QUEUE · BACKPRESSURE" in first_row) is (columns == 3)
        assert "BATCHES · RETRIES" in first_row
        assert all(len(line) <= width for line in text.splitlines())


# ---------------------------------------------------------------------------
# TelemetryHud
# ---------------------------------------------------------------------------


class PipelineCrashedError(Exception):
    pass


def terminal_console(buffer: io.StringIO | None = None) -> Console:
    return Console(
        file=buffer if buffer is not None else io.StringIO(),
        force_terminal=True,
        width=120,
        color_system=None,
    )


def running(hud: TelemetryHud) -> bool:
    """Read through a call so mypy does not narrow the property across awaits."""
    return hud.is_running


class TestTelemetryHud:
    def test_refresh_rate_validated(self) -> None:
        with pytest.raises(ValueError, match="refresh_hz"):
            TelemetryHud(MigrationMetrics(PLAN), refresh_hz=0)

    def test_defaults_to_stderr_console(self) -> None:
        hud = TelemetryHud(MigrationMetrics(PLAN))
        assert hud.console.stderr
        assert not hud.is_running
        assert not hud.refresh(force=True), "no frames before start"

    @pytest.mark.asyncio
    async def test_lifecycle_and_final_frame(self) -> None:
        metrics = MigrationMetrics(PLAN)
        buffer = io.StringIO()
        hud = TelemetryHud(metrics, console=terminal_console(buffer), refresh_hz=1000)
        assert not running(hud)
        async with hud:
            assert running(hud)
            assert hud.frames == 1
            metrics.set_phase(P.COMPLETED)
        assert not running(hud)
        assert hud.frames >= 2
        assert " COMPLETED " in buffer.getvalue()

    @pytest.mark.asyncio
    async def test_throttles_refreshes(self) -> None:
        clock = FakeClock()
        async with TelemetryHud(
            MigrationMetrics(PLAN), console=terminal_console(), refresh_hz=2, clock=clock
        ) as hud:
            assert not hud.refresh()
            clock.advance(0.3)
            assert not hud.refresh()
            clock.advance(0.2)
            assert hud.refresh()
            assert not hud.refresh()
            assert hud.refresh(force=True)
            assert hud.frames == 3

    @pytest.mark.asyncio
    async def test_background_pump_renders(self) -> None:
        async with TelemetryHud(
            MigrationMetrics(PLAN), console=terminal_console(), refresh_hz=200
        ) as hud:
            await asyncio.sleep(0.05)
            assert hud.frames > 1

    @pytest.mark.asyncio
    async def test_stops_cleanly_on_error(self) -> None:
        hud = TelemetryHud(MigrationMetrics(PLAN), console=terminal_console())

        async def crash() -> None:
            async with hud:
                raise PipelineCrashedError

        with pytest.raises(PipelineCrashedError):
            await crash()
        assert not hud.is_running

    @pytest.mark.asyncio
    async def test_reentry_rejected_and_restart_allowed(self) -> None:
        hud = TelemetryHud(MigrationMetrics(PLAN), console=terminal_console())
        async with hud:
            with pytest.raises(RuntimeError, match="already running"):
                await hud.__aenter__()
        async with hud:
            assert hud.is_running

    @pytest.mark.asyncio
    async def test_exit_without_enter_is_harmless(self) -> None:
        hud = TelemetryHud(MigrationMetrics(PLAN), console=terminal_console())
        await hud.__aexit__(None, None, None)
        assert hud.frames == 0

    @pytest.mark.asyncio
    async def test_stdout_json_untouched(self, capsys: pytest.CaptureFixture[str]) -> None:
        metrics = MigrationMetrics(PLAN)
        lines = [{"event": "batch", "n": i} for i in range(5)]
        async with TelemetryHud(metrics, refresh_hz=1000):
            for n, line in enumerate(lines):
                metrics.record_batch(n)
                print(json.dumps(line))
                await asyncio.sleep(0.002)
            metrics.set_phase(P.COMPLETED)
        captured = capsys.readouterr()
        assert [json.loads(raw) for raw in captured.out.splitlines()] == lines
        assert "COMPLETED" in captured.err
