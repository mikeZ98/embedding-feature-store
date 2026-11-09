"""Live traffic simulator and cooperative signal handling."""

from __future__ import annotations

import asyncio
import os
import signal

import pytest

from embedding_feature_store.core.errors import AbortRequestedError, StoreConnectionError
from embedding_feature_store.core.write_gate import WriteGate
from embedding_feature_store.pipeline.live_traffic import (
    LiveTrafficSimulator,
    SourceWriter,
    TrafficProfile,
    TrafficStats,
)
from embedding_feature_store.pipeline.shutdown import SIGNAL_EXIT_BASE, ShutdownController
from embedding_feature_store.schemas import CdcEvent, CreateAliasAction, DistanceMetric, VectorSpec
from embedding_feature_store.storage.memory import InMemoryVectorStore
from embedding_feature_store.storage.postgres_sim import PostgresSimulator
from embedding_feature_store.telemetry.metrics import MigrationMetrics
from tests.pipeline_support import make_plan

_conforms: SourceWriter = PostgresSimulator()
PROFILE = TrafficProfile(alias="live", feature_dim=3, id_range=50, seed=1)


async def served_store() -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    await store.create_collection("blue", VectorSpec(dimension=2, distance=DistanceMetric.DOT))
    await store.update_aliases([CreateAliasAction(alias="live", collection="blue")])
    return store


def simulator(
    store: InMemoryVectorStore,
    source: SourceWriter | None = None,
    gate: WriteGate | None = None,
    profile: TrafficProfile = PROFILE,
) -> tuple[LiveTrafficSimulator, MigrationMetrics, PostgresSimulator | SourceWriter]:
    metrics = MigrationMetrics(make_plan())
    src = source if source is not None else PostgresSimulator()
    sim = LiveTrafficSimulator(store, src, gate or WriteGate(), metrics, profile)
    return sim, metrics, src


class FailingSource:
    def upsert(self, record_id: int, tenant: str, features: tuple[float, ...]) -> CdcEvent:
        raise StoreConnectionError("db down")

    def delete(self, record_id: int) -> CdcEvent | None:
        raise StoreConnectionError("db down")


def running(sim: LiveTrafficSimulator) -> bool:
    """Read through a call so mypy does not narrow the property across awaits."""
    return sim.running


def critical(controller: ShutdownController) -> bool:
    return controller.in_critical_section


# ---------------------------------------------------------------------------
# LiveTrafficSimulator
# ---------------------------------------------------------------------------


class TestTraffic:
    @pytest.mark.parametrize("kwargs", [{"id_range": 0}, {"feature_dim": 0}])
    def test_profile_validated(self, kwargs: dict[str, int]) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            TrafficProfile(**{"alias": "a", "feature_dim": 1, "id_range": 1, "seed": 0, **kwargs})  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_reads_ok_and_errors(self) -> None:
        store = await served_store()
        sim, metrics, _ = simulator(store)
        await sim.read_once()
        await store.delete_collection("blue")  # alias disappears with its collection
        await sim.read_once()
        stats = sim.stats
        assert (stats.reads, stats.read_errors) == (2, 1)
        assert sim.last_read_error is not None
        assert "NotFoundError" in sim.last_read_error
        snap = metrics.snapshot()
        assert (snap.live_reads, snap.live_read_errors) == (2, 1)

    @pytest.mark.asyncio
    async def test_writes_commit_to_source(self) -> None:
        sim, metrics, source = simulator(await served_store())
        for _ in range(40):
            await sim.write_once()
        assert isinstance(source, PostgresSimulator)
        stats = sim.stats
        assert stats.writes_attempted == stats.writes_committed == 40
        assert (stats.dropped, stats.write_errors, stats.writes_held) == (0, 0, 0)
        assert 0 < await source.head_lsn() <= 40, "deletes of absent ids log nothing"
        assert metrics.snapshot().live_writes == 40

    @pytest.mark.asyncio
    async def test_writes_held_by_closed_gate_then_committed(self) -> None:
        gate = WriteGate()
        sim, metrics, source = simulator(await served_store(), gate=gate)
        await gate.close_and_drain(timeout_s=1)
        pending = [asyncio.create_task(sim.write_once()) for _ in range(5)]
        for _ in range(10):
            await asyncio.sleep(0)
        assert sim.stats.writes_committed == 0
        assert isinstance(source, PostgresSimulator)
        assert await source.head_lsn() == 0
        gate.reopen()
        await asyncio.gather(*pending)
        stats = sim.stats
        assert (stats.writes_committed, stats.writes_held, stats.dropped) == (5, 5, 0)
        assert metrics.snapshot().writes_held == 5

    @pytest.mark.asyncio
    async def test_write_errors_are_counted_not_dropped(self) -> None:
        sim, metrics, _ = simulator(await served_store(), source=FailingSource())
        for _ in range(10):
            await sim.write_once()
        stats = sim.stats
        assert (stats.write_errors, stats.writes_committed, stats.dropped) == (10, 0, 0)
        assert metrics.snapshot().live_write_errors == 10

    @pytest.mark.asyncio
    async def test_background_loops_and_lifecycle(self) -> None:
        profile = TrafficProfile(
            alias="live",
            feature_dim=3,
            id_range=20,
            seed=2,
            read_interval_s=0.0,
            write_interval_s=0.0,
        )
        sim, _, _ = simulator(await served_store(), profile=profile)
        assert not running(sim)
        async with sim:
            assert running(sim)
            with pytest.raises(RuntimeError, match="already running"):
                sim.start()
            for _ in range(50):
                await asyncio.sleep(0)
        assert not running(sim)
        stats = sim.stats
        assert stats.reads > 5
        assert stats.writes_attempted > 5
        assert (stats.read_errors, stats.dropped) == (0, 0)
        assert await sim.stop() == stats, "stop is idempotent"

    def test_dropped_arithmetic(self) -> None:
        stats = TrafficStats(
            reads=0,
            read_errors=0,
            writes_attempted=10,
            writes_committed=7,
            writes_held=0,
            write_errors=2,
        )
        assert stats.dropped == 1


# ---------------------------------------------------------------------------
# ShutdownController
# ---------------------------------------------------------------------------


class TestShutdown:
    def test_initial_state(self) -> None:
        controller = ShutdownController()
        assert (controller.requested, controller.signum, controller.exit_code) == (
            False,
            None,
            None,
        )
        controller.checkpoint("anywhere")  # no-op

    @pytest.mark.parametrize(("sig", "code"), [(signal.SIGINT, 130), (signal.SIGTERM, 143)])
    def test_first_signal_requests_abort_at_checkpoint(
        self, sig: signal.Signals, code: int
    ) -> None:
        controller = ShutdownController()
        controller.request(int(sig))
        assert controller.requested
        assert controller.exit_code == code == SIGNAL_EXIT_BASE + int(sig)
        with pytest.raises(AbortRequestedError) as exc:
            controller.checkpoint("backfill after id 42")
        assert (exc.value.signum, exc.value.checkpoint) == (int(sig), "backfill after id 42")

    def test_subsequent_signals_are_recorded_and_ignored(self) -> None:
        controller = ShutdownController()
        controller.request(int(signal.SIGTERM))
        controller.request(int(signal.SIGINT))
        controller.request(int(signal.SIGINT))
        assert controller.signum == int(signal.SIGTERM)
        assert controller.received == [15, 2, 2]

    def test_critical_section_defers_abort(self) -> None:
        controller = ShutdownController()
        with controller.critical():
            assert critical(controller)
            controller.request(int(signal.SIGINT))
            controller.checkpoint("during cutover")  # does not raise
            with controller.critical():
                controller.checkpoint("nested")
            controller.checkpoint("still critical")
        assert not critical(controller)
        with pytest.raises(AbortRequestedError):
            controller.checkpoint("after cutover")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
    async def test_real_signal_is_trapped(self, sig: signal.Signals) -> None:
        controller = ShutdownController()
        with controller.installed():
            os.kill(os.getpid(), sig)
            for _ in range(20):
                if controller.requested:
                    break
                await asyncio.sleep(0.005)
        assert controller.signum == int(sig)
        assert controller.exit_code == 128 + int(sig)

    @pytest.mark.asyncio
    async def test_install_twice_rejected_and_uninstall_idempotent(self) -> None:
        controller = ShutdownController(signals=(signal.SIGUSR1,))
        controller.install()
        with pytest.raises(RuntimeError, match="already installed"):
            controller.install()
        controller.uninstall()
        controller.uninstall()
        assert signal.getsignal(signal.SIGUSR1) == signal.SIG_DFL

    @pytest.mark.asyncio
    async def test_explicit_loop(self) -> None:
        controller = ShutdownController(signals=(signal.SIGUSR2,))
        controller.install(asyncio.get_running_loop())
        os.kill(os.getpid(), signal.SIGUSR2)
        await asyncio.sleep(0.01)
        controller.uninstall()
        assert controller.signum == int(signal.SIGUSR2)
