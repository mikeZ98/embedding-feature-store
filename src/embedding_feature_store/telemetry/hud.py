"""Rich terminal HUD for a running migration.

:func:`render_hud` is a pure function of a :class:`TelemetrySnapshot`; it
returns a :class:`rich.panel.Panel` whose metric grid adapts to the available
width (3x3 when wide, 2 columns when narrow).

:class:`TelemetryHud` drives a :class:`rich.live.Live` display from the event
loop at a throttled rate. It writes to **stderr** and never redirects
``sys.stdout``, so machine-readable output (JSON lines) printed to stdout while
the HUD is up stays byte-for-byte intact. ``Live`` runs with
``auto_refresh=False``: rendering happens on the event-loop thread, never on a
background thread that could observe the registry mid-update.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from types import TracebackType
from typing import Final, Self

from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from embedding_feature_store.core.ports import Clock
from embedding_feature_store.schemas import MigrationPhase
from embedding_feature_store.telemetry.metrics import MigrationMetrics, TelemetrySnapshot

__all__ = ["TelemetryHud", "phase_style", "render_hud"]

P = MigrationPhase

_PHASE_STYLES: Final[dict[MigrationPhase, str]] = {
    P.PENDING: "bold black on grey70",
    P.COMPLETED: "bold white on green",
    P.FAILED: "bold white on red",
    P.ROLLED_BACK: "bold black on yellow",
    P.ABORTED: "bold black on yellow",
    P.DRAINING: "bold white on magenta",
    P.CUTTING_OVER: "bold white on magenta",
}
_RUNNING_STYLE: Final = "bold white on blue"
_BORDER: Final[dict[MigrationPhase, str]] = {
    P.COMPLETED: "green",
    P.FAILED: "red",
    P.ROLLED_BACK: "yellow",
    P.ABORTED: "yellow",
}
_WIDE: Final = 90
_NONE: Final = "—"


def phase_style(phase: MigrationPhase) -> str:
    """Badge style: green done, red failed, magenta freeze window, blue running."""
    return _PHASE_STYLES.get(phase, _RUNNING_STYLE)


def _ms(value: float | None) -> str:
    return _NONE if value is None else f"{value:,.1f} ms"


def _cell(title: str, *lines: Text | str) -> Text:
    text = Text()
    text.append(title.upper(), style="dim")
    for line in lines:
        text.append("\n")
        text.append(line if isinstance(line, Text) else Text(line, style="bold"))
    return text


def _flag(value: str, *, alarm: bool, style: str = "bold red") -> Text:
    return Text(value, style=style if alarm else "bold")


class _MetricGrid:
    """Grid that picks its column count from the width it is given."""

    def __init__(self, cells: list[Text]) -> None:
        self.cells = cells

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        columns = 3 if options.max_width >= _WIDE else 2
        grid = Table.grid(expand=True, padding=(0, 2))
        for _ in range(columns):
            grid.add_column(ratio=1)
        for start in range(0, len(self.cells), columns):
            row = self.cells[start : start + columns]
            grid.add_row(*row, *([""] * (columns - len(row))))
        yield grid


def _header(s: TelemetrySnapshot) -> Text:
    header = Text()
    header.append(f" {s.phase.value.upper().replace('_', ' ')} ", style=phase_style(s.phase))
    header.append(f"  {s.migration_id}  ", style="bold")
    header.append(f"{s.source_collection} → {s.target_collection}", style="cyan")
    header.append(f"  alias {s.alias}", style="dim")
    header.append(f"  dim {s.dimension} · {s.distance.value}", style="dim")
    return header


def _progress(s: TelemetrySnapshot) -> RenderableType:
    progress = s.progress
    if progress is None or s.estimated_total is None:
        bar = ProgressBar(total=None)  # pulses: total not yet known
        label = Text(f"{s.backfilled:,} points backfilled (total unknown)", style="dim")
    else:
        bar = ProgressBar(total=s.estimated_total, completed=min(s.backfilled, s.estimated_total))
        label = Text(f"{progress:6.1%}  {s.backfilled:,} / {s.estimated_total:,} points")
    rows = Table.grid(expand=True)
    rows.add_column()
    rows.add_row(bar)
    rows.add_row(label)
    return rows


def _grid(s: TelemetrySnapshot) -> _MetricGrid:
    queue_full = s.queue_capacity > 0 and s.queue_depth >= s.queue_capacity
    live_errors = s.live_read_errors + s.live_write_errors
    return _MetricGrid(
        [
            _cell("throughput", f"{s.throughput_pps:,.1f} pts/s"),
            _cell("batches · retries", f"{s.batches:,} · {s.retries:,}"),
            _cell(
                "queue · backpressure",
                _flag(f"{s.queue_depth}/{s.queue_capacity}", alarm=queue_full, style="bold yellow"),
                f"{s.backpressure_stalls:,} stalls",
            ),
            _cell(
                "dimension checks",
                _flag(
                    f"{s.vectors_verified:,} ok · {s.dimension_violations:,} bad",
                    alarm=s.dimension_violations > 0,
                ),
            ),
            _cell(
                "cdc applied · lag",
                f"lsn {s.applied_lsn:,} · {s.cdc_events:,} ev",
                _flag(f"lag {s.cdc_lag:,}", alarm=s.cdc_lag > 0, style="bold yellow"),
            ),
            _cell(
                "live r/w · err · held",
                f"{s.live_reads:,} / {s.live_writes:,}",
                _flag(f"{live_errors:,} err · {s.writes_held:,} held", alarm=live_errors > 0),
            ),
            _cell("drain · cutover", f"{_ms(s.drain_ms)} · {_ms(s.cutover_ms)}"),
            _cell("elapsed", f"{s.elapsed_s:,.1f} s"),
            _cell("last retry", Text(s.last_retry or _NONE, style="dim")),
        ]
    )


def _verification(s: TelemetrySnapshot) -> Text:
    v = s.verification
    if v is None:
        return Text(" NOT VERIFIED ", style="dim reverse")
    badge = Text(" VERIFIED ✓ " if v.passed else " MISMATCH ✗ ")
    badge.stylize("bold white on green" if v.passed else "bold white on red")
    badge.append(
        f"  source {v.source_count:,} [{v.source_checksum_prefix}]"
        f"  target {v.target_count:,} [{v.target_checksum_prefix}]",
        style="dim",
    )
    return badge


def render_hud(snapshot: TelemetrySnapshot) -> Panel:
    parts: list[RenderableType] = [
        _header(snapshot),
        _progress(snapshot),
        _grid(snapshot),
        _verification(snapshot),
    ]
    if snapshot.error:
        parts.append(
            Panel(Text(snapshot.error, style="bold red"), title="error", border_style="red")
        )
    return Panel(
        Group(*parts),
        title="embedding-feature-store",
        border_style=_BORDER.get(snapshot.phase, "blue"),
    )


class TelemetryHud:
    """Async context manager rendering ``metrics`` live on stderr.

    ``refresh_hz`` caps the render rate; :meth:`refresh` calls arriving faster
    are dropped unless ``force=True``. A final frame is always drawn on exit.
    """

    def __init__(
        self,
        metrics: MigrationMetrics,
        *,
        console: Console | None = None,
        refresh_hz: float = 4.0,
        clock: Clock = time.monotonic,
    ) -> None:
        if not refresh_hz > 0:
            msg = "refresh_hz must be > 0"
            raise ValueError(msg)
        self._metrics = metrics
        self._console = console if console is not None else Console(stderr=True)
        self._interval = 1.0 / refresh_hz
        self._clock = clock
        self._live: Live | None = None
        self._task: asyncio.Task[None] | None = None
        self._last: float | None = None
        self.frames = 0

    @property
    def console(self) -> Console:
        return self._console

    @property
    def is_running(self) -> bool:
        return self._live is not None

    def refresh(self, *, force: bool = False) -> bool:
        """Render a frame now unless throttled; returns whether one was drawn."""
        if self._live is None:
            return False
        now = self._clock()
        if not force and self._last is not None and now - self._last < self._interval:
            return False
        self._last = now
        self._live.update(render_hud(self._metrics.snapshot()), refresh=True)
        self.frames += 1
        return True

    async def _pump(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            self.refresh()

    async def __aenter__(self) -> Self:
        if self._live is not None:
            msg = "TelemetryHud is already running"
            raise RuntimeError(msg)
        self._live = Live(
            render_hud(self._metrics.snapshot()),
            console=self._console,
            auto_refresh=False,
            redirect_stdout=False,
            redirect_stderr=False,
            transient=False,
        )
        self._live.start(refresh=True)
        self.frames = 1
        self._last = self._clock()
        self._task = asyncio.create_task(self._pump())
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        task, live = self._task, self._live
        self._task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self.refresh(force=True)
        self._live = None
        if live is not None:
            live.stop()
