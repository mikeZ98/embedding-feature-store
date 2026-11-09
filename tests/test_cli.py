"""efs-migrate: flags, exit codes, stdout JSON contract, report files, backends."""

from __future__ import annotations

import importlib
import json
import logging
import os
import runpy
import signal
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from embedding_feature_store.core.errors import PermanentError
from embedding_feature_store.pipeline import cli
from embedding_feature_store.pipeline.shutdown import ShutdownController
from embedding_feature_store.storage.memory import InMemoryVectorStore
from embedding_feature_store.storage.qdrant_http import QdrantHttpStore
from tests.qdrant_emulator import QdrantEmulator

SMALL = ["--rows", "60", "--batch-size", "16", "--source-dim", "4", "--green-dim", "6"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for key in list(os.environ):
        if key.startswith("EFS_"):
            monkeypatch.delenv(key)
    yield
    assert not logging.getLogger("embedding_feature_store").handlers, "handlers must be removed"


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, Any], str]:
    code = cli.main(list(argv))
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert len(lines) == 1, f"stdout must be exactly one JSON line, got {captured.out!r}"
    return code, json.loads(lines[0]), captured.err


# ---------------------------------------------------------------------------
# Usage and configuration errors
# ---------------------------------------------------------------------------


def test_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--help"]) == 0
    assert "efs-migrate" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [["--bogus"], ["--rows", "many"], ["--distance", "manhattan"]])
def test_usage_errors_exit_two(capsys: pytest.CaptureFixture[str], argv: list[str]) -> None:
    assert cli.main(argv) == cli.EXIT_USAGE == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "usage:" in captured.err


def test_invalid_config_exits_two(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EFS_BATCH_SIZE", "0")
    assert cli.main([]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "invalid configuration" in captured.err


def test_flags_validated_as_config(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--green", "features_blue"]) == 2
    assert "distinct" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Successful runs
# ---------------------------------------------------------------------------


def test_completed_run_prints_one_json_line(capsys: pytest.CaptureFixture[str]) -> None:
    code, summary, err = run(capsys, *SMALL, "--no-hud", "--no-live-traffic")
    assert code == 0
    assert summary["status"] == "completed"
    assert summary["exit_code"] == 0
    assert summary["alias"] == "features_active"
    assert summary["alias_bound_to"] == "features_green"
    assert summary["rows_backfilled"] == 60
    assert summary["target_dimension"] == 6
    assert summary["verification"]["passed"] is True
    assert summary["verification"]["source_count"] == 60
    assert summary["live_traffic"] is None
    assert summary["signals"] == []
    assert summary["error"] is None
    assert "phase completed" in err, "logs go to stderr"


def test_live_traffic_summary(capsys: pytest.CaptureFixture[str]) -> None:
    code, summary, _ = run(capsys, *SMALL, "--no-hud", "--live-traffic", "--seed", "4")
    assert code == 0
    traffic = summary["live_traffic"]
    assert traffic["read_errors"] == 0
    assert traffic["dropped_writes"] == 0
    assert traffic["reads"] >= 1


def test_report_file(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    report = tmp_path / "reports" / "nested" / "run.json"
    code, summary, _ = run(capsys, *SMALL, "--no-hud", "--report", str(report))
    assert code == 0
    data = json.loads(report.read_text())
    assert {k: v for k, v in data.items() if k != "timeline"} == summary
    assert [step["to"] for step in data["timeline"]] == [
        "provisioning",
        "backfilling",
        "reconciling",
        "draining",
        "verifying",
        "cutting_over",
        "decommissioning",
        "completed",
    ]
    assert data["timeline"][0] == {
        "sequence": 0,
        "from": "pending",
        "to": "provisioning",
        "reason": "provision target",
    }


def test_hud_renders_to_stderr_only(capsys: pytest.CaptureFixture[str]) -> None:
    code, summary, err = run(capsys, *SMALL, "--hud", "--no-live-traffic")
    assert code == 0
    assert summary["status"] == "completed"
    assert "embedding-feature-store" in err
    assert "COMPLETED" in err


def test_log_level_filters(capsys: pytest.CaptureFixture[str]) -> None:
    _, _, err = run(capsys, *SMALL, "--no-hud", "--no-live-traffic", "--log-level", "WARNING")
    assert "phase" not in err


def test_environment_configures_run(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EFS_MIGRATION_ID", "from-env")
    monkeypatch.setenv("EFS_LIVE_TRAFFIC", "false")
    monkeypatch.setenv("EFS_HUD", "false")
    code, summary, _ = run(capsys, *SMALL)
    assert code == 0
    assert summary["migration_id"] == "from-env"
    assert summary["live_traffic"] is None


# ---------------------------------------------------------------------------
# Qdrant backend (offline emulator)
# ---------------------------------------------------------------------------


def use_emulator(monkeypatch: pytest.MonkeyPatch, emulator: QdrantEmulator) -> list[dict[str, Any]]:
    connects: list[dict[str, Any]] = []

    def connect(url: str, **kwargs: Any) -> QdrantHttpStore:
        connects.append({"url": url, **kwargs})
        headers = {} if kwargs["api_key"] is None else {"api-key": kwargs["api_key"]}
        client = httpx.AsyncClient(
            base_url=url, headers=headers, transport=httpx.MockTransport(emulator)
        )
        return QdrantHttpStore(client)

    monkeypatch.setattr(QdrantHttpStore, "connect", staticmethod(connect))
    return connects


def test_qdrant_backend_end_to_end(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    emulator = QdrantEmulator()
    connects = use_emulator(monkeypatch, emulator)
    monkeypatch.setenv("EFS_QDRANT_API_KEY", "s3cret")
    code, summary, _ = run(
        capsys,
        *SMALL,
        "--no-hud",
        "--qdrant-url",
        "http://qdrant.test:6333",
        "--qdrant-timeout",
        "3",
    )
    assert code == 0, summary
    assert connects == [{"url": "http://qdrant.test:6333", "api_key": "s3cret", "timeout_s": 3.0}]
    assert emulator.aliases == {"features_active": "features_green"}
    assert set(emulator.collections) == {"features_green"}
    assert all(r.headers["api-key"] == "s3cret" for r in emulator.requests)
    assert "s3cret" not in json.dumps(summary)


def test_qdrant_without_api_key(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    emulator = QdrantEmulator()
    use_emulator(monkeypatch, emulator)
    code, _, _ = run(capsys, *SMALL, "--no-hud", "--no-live-traffic", "--qdrant-url", "http://q:1")
    assert code == 0
    assert all("api-key" not in r.headers for r in emulator.requests)


def test_bootstrap_error_exits_one_with_json(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    emulator = QdrantEmulator()
    emulator.aliases["features_active"] = "someone_elses_collection"
    use_emulator(monkeypatch, emulator)
    code, summary, err = run(capsys, *SMALL, "--no-hud", "--qdrant-url", "http://q:1")
    assert code == 1
    assert summary["status"] == "error"
    assert "ConflictError" in summary["error"]
    assert "already bound" in err
    assert emulator.aliases == {"features_active": "someone_elses_collection"}


# ---------------------------------------------------------------------------
# Failure and signal exit codes
# ---------------------------------------------------------------------------


def test_failed_migration_exits_one(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    class StubbornBlue(InMemoryVectorStore):
        async def delete_collection(self, name: str) -> None:
            if name == "features_blue":
                raise PermanentError("blue is pinned")
            await super().delete_collection(name)

    monkeypatch.setattr(cli, "InMemoryVectorStore", StubbornBlue)
    code, summary, _ = run(capsys, *SMALL, "--no-hud", "--no-live-traffic")
    assert code == 1
    assert summary["status"] == "aborted"
    assert "left for manual cleanup" in summary["error"]
    assert summary["alias_bound_to"] == "features_green"


def test_sigterm_exits_143_and_keeps_alias_on_blue(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    class SelfSignalling(ShutdownController):
        def install(self, loop: Any = None) -> None:
            super().install(loop)
            os.kill(os.getpid(), signal.SIGTERM)  # delivered on the next loop iteration

    monkeypatch.setattr(cli, "ShutdownController", SelfSignalling)
    code, summary, _ = run(capsys, *SMALL, "--no-hud")
    assert code == 143
    assert summary["status"] == "aborted"
    assert summary["signals"] == [int(signal.SIGTERM)]
    assert summary["alias_bound_to"] == "features_blue"
    assert summary["exit_code"] == 143


# ---------------------------------------------------------------------------
# python -m embedding_feature_store
# ---------------------------------------------------------------------------


def test_module_entry_point(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["efs-migrate", *SMALL, "--no-hud", "--no-live-traffic"])
    # runpy warns (an error here) if the module is already imported; make the
    # test independent of execution order.
    monkeypatch.delitem(sys.modules, "embedding_feature_store.__main__", raising=False)
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("embedding_feature_store", run_name="__main__")
    assert exc.value.code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"


def test_module_import_has_no_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "embedding_feature_store.__main__", raising=False)
    module = importlib.import_module("embedding_feature_store.__main__")
    assert module.main is cli.main
