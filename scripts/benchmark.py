"""Benchmark harness behind the README tables.

Runs the real efs-migrate entry point (``cli.main``) once per scenario, each in
a fresh child process so peak RSS is attributable to that run alone, with live
read/write traffic on by default. Two timings the CLI summary does not report
are taken by thin subclasses swapped into the CLI module:

* migration_s: wall time of ``MigrationOrchestrator.run`` (excludes building the
  source table and loading the Blue collection, which ``elapsed_s`` includes);
* freeze_ms: from the write gate closing to the first reopen, which covers the
  drain, the final CDC catch-up, two-tier verification and the alias swap.

Usage:
    uv run python scripts/benchmark.py [--rows 1000 10000 50000] [--repeat 3]
                                       [--qdrant-url http://127.0.0.1:6333]
                                       [--no-live-traffic] [--json PATH]

Each scenario's medians are printed as a Markdown table row; ``--json`` keeps
every raw run. Against Qdrant, each run uses its own alias and collections.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# child: one instrumented migration
# ---------------------------------------------------------------------------


def run_child(rows: int, qdrant_url: str | None, run_id: str, *, live_traffic: bool) -> int:
    from embedding_feature_store.core.write_gate import WriteGate  # noqa: PLC0415
    from embedding_feature_store.pipeline import cli  # noqa: PLC0415
    from embedding_feature_store.pipeline.orchestrator import (  # noqa: PLC0415
        MigrationOrchestrator,
        MigrationResult,
    )

    marks: dict[str, float] = {}

    class TimedGate(WriteGate):
        async def close_and_drain(self, timeout_s: float) -> None:
            marks.setdefault("frozen", time.perf_counter())
            await super().close_and_drain(timeout_s)

        def reopen(self) -> None:
            if "frozen" in marks:
                marks.setdefault("reopened", time.perf_counter())
            super().reopen()

    class TimedOrchestrator(MigrationOrchestrator):
        async def run(self) -> MigrationResult:
            started = time.perf_counter()
            try:
                return await super().run()
            finally:
                marks["migration_s"] = time.perf_counter() - started

    cli.WriteGate = TimedGate  # type: ignore[misc]
    cli.MigrationOrchestrator = TimedOrchestrator  # type: ignore[misc]

    argv = ["--rows", str(rows), "--no-hud", "--log-level", "WARNING", "--seed", "0"]
    argv.append("--live-traffic" if live_traffic else "--no-live-traffic")
    if qdrant_url is not None:
        argv += [
            "--qdrant-url", qdrant_url,
            "--migration-id", f"bench-{run_id}",
            "--alias", f"bench_{run_id}_active",
            "--blue", f"bench_{run_id}_blue",
            "--green", f"bench_{run_id}_green",
        ]  # fmt: skip

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = cli.main(argv)
    summary = json.loads(out.getvalue())
    frozen = marks.get("frozen")
    reopened = marks.get("reopened")
    print(
        json.dumps(
            {
                "exit_code": code,
                "summary": summary,
                "migration_s": marks.get("migration_s"),
                "freeze_ms": None
                if frozen is None or reopened is None
                else (reopened - frozen) * 1000.0,
            }
        )
    )
    return code


# ---------------------------------------------------------------------------
# parent: scenarios, peak RSS, medians
# ---------------------------------------------------------------------------


def spawn(rows: int, qdrant_url: str | None, run_id: str, *, live_traffic: bool) -> dict[str, Any]:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", "--rows", str(rows)]
    cmd += ["--run-id", run_id]
    if qdrant_url is not None:
        cmd += ["--qdrant-url", qdrant_url]
    if not live_traffic:
        cmd.append("--no-live-traffic")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, cwd=ROOT)  # noqa: S603
    assert proc.stdout is not None  # noqa: S101 - set by stdout=PIPE
    stdout = proc.stdout.read()
    _, status, usage = os.wait4(proc.pid, 0)
    proc.returncode = os.waitstatus_to_exitcode(status)
    if proc.returncode != 0:
        msg = f"child exited {proc.returncode} for rows={rows}: {stdout}"
        raise RuntimeError(msg)
    # ru_maxrss is bytes on macOS and KiB on Linux.
    rss_bytes = usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024
    record: dict[str, Any] = json.loads(stdout.strip().splitlines()[-1])
    record["peak_rss_mib"] = rss_bytes / 2**20
    return record


def median(values: list[float]) -> float:
    return statistics.median(values)


def row(backend: str, rows: int, runs: list[dict[str, Any]]) -> str:
    summaries = [r["summary"] for r in runs]
    migration_s = median([r["migration_s"] for r in runs])
    traffic = [s["live_traffic"] for s in summaries if s["live_traffic"] is not None]
    writes = sum(t["writes"] for t in traffic)
    held = sum(t["writes_held"] for t in traffic)
    dropped = sum(t["dropped_writes"] for t in traffic)
    read_errors = sum(t["read_errors"] for t in traffic)
    verified = all(s["verification"] and s["verification"]["passed"] for s in summaries)
    cells = [
        backend,
        f"{rows:,}",
        f"{median([s['rows_backfilled'] / r['migration_s'] for s, r in zip(summaries, runs, strict=True)]):,.0f}",  # noqa: E501
        f"{migration_s:.3f}",
        f"{median([r['freeze_ms'] for r in runs]):.1f}",
        f"{median([s['drain_ms'] for s in summaries]):.3f}",
        f"{median([s['cutover_ms'] for s in summaries]):.3f}",
        f"{median([r['peak_rss_mib'] for r in runs]):.0f}",
        f"{writes} / {held} / {dropped}" if traffic else "off",
        str(read_errors) if traffic else "off",
        f"{len(runs)}/{len(runs)}" if verified else "FAILED",
    ]
    return "| " + " | ".join(cells) + " |"


HEADER = (
    "| Backend | Rows | Throughput (pts/s) | Migration (s) | Freeze window (ms) "
    "| Drain (ms) | Alias swap (ms) | Peak RSS (MiB) | Live writes / held / dropped "
    "| Read errors | Verified |\n"
    "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, nargs="+", default=[1_000, 10_000, 50_000])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--qdrant-url")
    parser.add_argument("--no-live-traffic", dest="live_traffic", action="store_false")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-id", default="0", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:
        return run_child(args.rows[0], args.qdrant_url, args.run_id, live_traffic=args.live_traffic)

    backend = "Qdrant 1.13.6" if args.qdrant_url else "in-memory"
    print(
        f"<!-- {platform.platform()} | Python {platform.python_version()} | "
        f"{os.cpu_count()} CPUs | backend={backend} | repeat={args.repeat} -->"
    )
    print(HEADER)
    raw: dict[str, list[dict[str, Any]]] = {}
    stamp = time.strftime("%H%M%S")
    for rows in args.rows:
        runs = [
            spawn(rows, args.qdrant_url, f"{stamp}_{rows}_{i}", live_traffic=args.live_traffic)
            for i in range(args.repeat)
        ]
        raw[str(rows)] = runs
        print(row(backend, rows, runs), flush=True)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(raw, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
