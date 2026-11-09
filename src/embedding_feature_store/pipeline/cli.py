"""``efs-migrate``: run a blue/green embedding migration end to end.

Output contract:

* **stdout** — exactly one line of JSON (the run summary), nothing else;
* **stderr** — logs and, when enabled, the live Rich HUD;
* ``--report PATH`` — the summary plus the phase timeline, pretty-printed.

Exit codes: ``0`` completed; ``1`` failed, rolled back or aborted by an error;
``2`` usage or configuration error; ``128 + signum`` aborted by a signal
(130 SIGINT, 143 SIGTERM).

The source of record is the in-process PostgreSQL simulator, seeded
deterministically; the vector store is Qdrant (``--qdrant-url``) or the
in-process store. A Blue collection is bootstrapped and bound to the alias,
then migrated to Green under optional live read/write traffic.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import sys
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

from rich.logging import RichHandler

from embedding_feature_store.config import ConfigError, Settings
from embedding_feature_store.core.errors import FeatureStoreError
from embedding_feature_store.core.ports import VectorStore
from embedding_feature_store.core.projection import PointProjector
from embedding_feature_store.core.retry import Retrier
from embedding_feature_store.core.write_gate import WriteGate
from embedding_feature_store.embedding import RandomProjectionEmbedder
from embedding_feature_store.pipeline.bootstrap import bootstrap_blue
from embedding_feature_store.pipeline.live_traffic import (
    LiveTrafficSimulator,
    TrafficProfile,
    TrafficStats,
)
from embedding_feature_store.pipeline.orchestrator import MigrationOrchestrator, MigrationResult
from embedding_feature_store.pipeline.shutdown import ShutdownController
from embedding_feature_store.schemas import DistanceMetric
from embedding_feature_store.storage.memory import InMemoryVectorStore
from embedding_feature_store.storage.postgres_sim import PostgresSimulator
from embedding_feature_store.storage.qdrant_http import QdrantHttpStore
from embedding_feature_store.telemetry.hud import TelemetryHud
from embedding_feature_store.telemetry.metrics import MigrationMetrics, TelemetrySnapshot

__all__ = ["build_parser", "main"]

EXIT_USAGE = 2
_PACKAGE_LOGGER = "embedding_feature_store"
_TENANTS = ("t0", "t1", "t2")

type Summary = dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="efs-migrate",
        description="Blue/green embedding migration with live verification and atomic cutover.",
        epilog="Every option can also be set as EFS_<OPTION> in the environment.",
    )
    add = parser.add_argument
    add("--qdrant-url", dest="qdrant_url", help="Qdrant base URL (default: in-process store)")
    add("--qdrant-timeout", dest="qdrant_timeout_s", type=float, metavar="SECONDS")
    add("--migration-id", dest="migration_id")
    add("--alias", dest="alias", help="serving alias (default: features_active)")
    add("--blue", dest="blue_collection", metavar="COLLECTION")
    add("--green", dest="green_collection", metavar="COLLECTION")
    add("--rows", dest="source_rows", type=int, help="rows in the simulated source table")
    add("--source-dim", dest="source_dim", type=int)
    add("--blue-dim", dest="blue_dim", type=int)
    add("--green-dim", dest="green_dim", type=int)
    add("--distance", dest="distance", choices=[m.value for m in DistanceMetric])
    add("--batch-size", dest="batch_size", type=int)
    add("--queue-capacity", dest="queue_capacity", type=int)
    add("--cdc-page-size", dest="cdc_page_size", type=int)
    add("--retries", dest="retry_attempts", type=int, metavar="ATTEMPTS")
    add("--drain-timeout", dest="drain_timeout_s", type=float, metavar="SECONDS")
    add("--seed", dest="seed", type=int)
    add("--live-traffic", dest="live_traffic", action=argparse.BooleanOptionalAction)
    add("--hud", dest="hud", action=argparse.BooleanOptionalAction)
    add("--report", dest="report_path", metavar="PATH", help="write a JSON report file")
    add("--log-level", dest="log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def _build_store(settings: Settings) -> tuple[VectorStore, Callable[[], Awaitable[None]]]:
    if settings.qdrant_url is None:

        async def nothing() -> None:
            return None

        return InMemoryVectorStore(), nothing
    key = settings.qdrant_api_key
    store = QdrantHttpStore.connect(
        settings.qdrant_url,
        api_key=None if key is None else key.get_secret_value(),
        timeout_s=settings.qdrant_timeout_s,
    )
    return store, store.aclose


@contextlib.contextmanager
def _logging_to(handler: logging.Handler, level: str) -> Any:  # noqa: ANN401
    logger = logging.getLogger(_PACKAGE_LOGGER)
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)


def _summary(
    settings: Settings,
    result: MigrationResult,
    *,
    snapshot: TelemetrySnapshot,
    traffic: TrafficStats | None,
    shutdown: ShutdownController,
    bound_to: str | None,
) -> Summary:
    report = result.report
    return {
        "migration_id": settings.migration_id,
        "status": result.state.phase.value,
        "exit_code": result.exit_code,
        "error": result.state.error,
        "alias": settings.alias,
        "alias_bound_to": bound_to,
        "source_collection": settings.blue_collection,
        "target_collection": settings.green_collection,
        "target_dimension": settings.green_dim,
        "rows_backfilled": snapshot.backfilled,
        "cdc_events": snapshot.cdc_events,
        "handoff_lsn": result.handoff_lsn,
        "elapsed_s": round(snapshot.elapsed_s, 3),
        "drain_ms": snapshot.drain_ms,
        "cutover_ms": snapshot.cutover_ms,
        "verification": None
        if report is None
        else {
            "passed": report.passed,
            "source_count": report.source_count,
            "target_count": report.target_count,
            "source_checksum": report.source_checksum,
            "target_checksum": report.target_checksum,
        },
        "live_traffic": None
        if traffic is None
        else {
            "reads": traffic.reads,
            "read_errors": traffic.read_errors,
            "writes": traffic.writes_attempted,
            "writes_held": traffic.writes_held,
            "write_errors": traffic.write_errors,
            "dropped_writes": traffic.dropped,
        },
        "signals": list(shutdown.received),
    }


async def _migrate(settings: Settings, store: VectorStore) -> tuple[Summary, MigrationResult]:
    plan = settings.plan()
    source = PostgresSimulator(queue_capacity=settings.queue_capacity)
    source.populate(settings.source_rows, settings.source_dim, seed=settings.seed, tenants=_TENANTS)
    metrics = MigrationMetrics(plan)
    retrier = Retrier(plan.retry, seed=settings.seed, on_retry=metrics.on_retry)

    blue_embedder = RandomProjectionEmbedder(
        settings.source_dim, settings.blue_dim, seed=settings.seed
    )
    green_embedder = RandomProjectionEmbedder(
        settings.source_dim, settings.green_dim, seed=settings.seed + 1
    )
    await bootstrap_blue(
        store,
        source,
        PointProjector(settings.blue_spec, blue_embedder),
        retrier,
        collection=settings.blue_collection,
        alias=settings.alias,
        batch_size=settings.batch_size,
    )

    gate = WriteGate(observer=None)
    shutdown = ShutdownController()
    orchestrator = MigrationOrchestrator(
        plan,
        store=store,
        source=source,
        embedder=green_embedder,
        gate=gate,
        metrics=metrics,
        shutdown=shutdown,
        retrier=retrier,
        drain_timeout_s=settings.drain_timeout_s,
        cdc_page_size=settings.cdc_page_size,
    )
    traffic = (
        LiveTrafficSimulator(
            store,
            source,
            gate,
            metrics,
            TrafficProfile(
                alias=settings.alias,
                feature_dim=settings.source_dim,
                id_range=max(1, settings.source_rows),
                seed=settings.seed,
            ),
        )
        if settings.live_traffic
        else None
    )
    show_hud = settings.hud if settings.hud is not None else sys.stderr.isatty()
    hud = TelemetryHud(metrics) if show_hud else None
    handler: logging.Handler = (
        logging.StreamHandler(sys.stderr)
        if hud is None
        else RichHandler(console=hud.console, show_path=False)
    )

    async with contextlib.AsyncExitStack() as stack:
        stack.enter_context(_logging_to(handler, settings.log_level))
        stack.enter_context(shutdown.installed())
        if hud is not None:
            await stack.enter_async_context(hud)
        if traffic is not None:
            await stack.enter_async_context(traffic)
        result = await orchestrator.run()

    bound_to = (await retrier.run(store.list_aliases)).get(settings.alias)
    summary = _summary(
        settings,
        result,
        snapshot=metrics.snapshot(),
        traffic=None if traffic is None else traffic.stats,
        shutdown=shutdown,
        bound_to=bound_to,
    )
    return summary, result


async def _execute(settings: Settings) -> tuple[int, Summary, MigrationResult | None]:
    store, close = _build_store(settings)
    try:
        summary, result = await _migrate(settings, store)
    except FeatureStoreError as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(f"efs-migrate: {error}", file=sys.stderr)
        return (
            1,
            {
                "migration_id": settings.migration_id,
                "status": "error",
                "exit_code": 1,
                "error": error,
            },
            None,
        )
    finally:
        await close()
    return result.exit_code, summary, result


def _write_report(path: str, summary: Summary, result: MigrationResult | None) -> None:
    timeline = (
        []
        if result is None
        else [
            {
                "sequence": t.sequence,
                "from": t.from_phase.value,
                "to": t.to_phase.value,
                "reason": t.reason,
            }
            for t in result.state.history
        ]
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({**summary, "timeline": timeline}, indent=2, sort_keys=True) + "\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        namespace = parser.parse_args(argv)
    except SystemExit as exc:  # argparse: --help (0) or usage error (2)
        return exc.code if isinstance(exc.code, int) else EXIT_USAGE
    try:
        settings = Settings.load(os.environ, **vars(namespace))
    except ConfigError as exc:
        print(f"efs-migrate: invalid configuration:\n{exc}", file=sys.stderr)
        return EXIT_USAGE

    code, summary, result = asyncio.run(_execute(settings))
    if settings.report_path is not None:
        _write_report(settings.report_path, summary, result)
    sys.stdout.write(json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n")
    sys.stdout.flush()
    return code
