"""End-to-end orchestrator matrix: success, rollback, abort, and decommission failure."""

from __future__ import annotations

import asyncio
import os
import signal
from collections.abc import Callable, Mapping, Sequence

import pytest

from embedding_feature_store.core.errors import (
    NotFoundError,
    PermanentError,
    ServerError,
    StoreTimeoutError,
)
from embedding_feature_store.core.verification import ExpectationLedger, MigrationVerifier
from embedding_feature_store.pipeline.live_traffic import LiveTrafficSimulator, TrafficProfile
from embedding_feature_store.pipeline.orchestrator import MigrationResult
from embedding_feature_store.pipeline.reconcile import CdcReconciler
from embedding_feature_store.pipeline.writer import PointWriter
from embedding_feature_store.schemas import (
    AliasAction,
    CreateAliasAction,
    MigrationPhase,
    PointRecord,
    VectorSpec,
)
from embedding_feature_store.storage.faults import ALL_METHODS, FaultInjectingStore, FaultPlan
from embedding_feature_store.storage.memory import InMemoryVectorStore
from tests.pipeline_support import ALIAS, BLUE, GREEN, GREEN_SPEC, SOURCE_DIM, World, make_world

P = MigrationPhase
LIFECYCLE = [
    P.PROVISIONING,
    P.BACKFILLING,
    P.RECONCILING,
    P.DRAINING,
    P.VERIFYING,
    P.CUTTING_OVER,
    P.DECOMMISSIONING,
    P.COMPLETED,
]

Hook = Callable[..., None]


class ScriptedStore(InMemoryVectorStore):
    """In-memory store whose calls can be intercepted per method name."""

    def __init__(self) -> None:
        super().__init__()
        self.hooks: dict[str, Hook] = {}
        self.calls: dict[str, int] = {}

    def _hook(self, name: str, *args: object) -> None:
        self.calls[name] = self.calls.get(name, 0) + 1
        hook = self.hooks.get(name)
        if hook is not None:
            hook(*args)

    async def create_collection(self, name: str, spec: VectorSpec) -> None:
        self._hook("create_collection", name, spec)
        await super().create_collection(name, spec)

    async def delete_collection(self, name: str) -> None:
        self._hook("delete_collection", name)
        await super().delete_collection(name)

    async def upsert(self, collection: str, points: Sequence[PointRecord]) -> None:
        self._hook("upsert", collection, points)
        await super().upsert(collection, points)

    async def count(self, collection: str) -> int:
        self._hook("count", collection)
        return await super().count(collection)

    async def list_aliases(self) -> Mapping[str, str]:
        self._hook("list_aliases")
        return await super().list_aliases()

    async def update_aliases(self, actions: Sequence[AliasAction]) -> None:
        self._hook("update_aliases", actions)
        await super().update_aliases(actions)


def targets_green(actions: Sequence[AliasAction]) -> bool:
    return any(isinstance(a, CreateAliasAction) and a.collection == GREEN for a in actions)


async def scripted_world(rows: int = 120) -> tuple[World, ScriptedStore]:
    store = ScriptedStore()
    return await make_world(rows=rows, store=store), store


def phases(result: MigrationResult) -> list[MigrationPhase]:
    return [t.to_phase for t in result.state.history]


async def assert_rolled_back_to_blue(world: World, result: MigrationResult) -> None:
    assert await world.aliases() == {ALIAS: BLUE}
    assert not await world.store.collection_exists(GREEN)
    assert world.gate.is_open
    assert result.cutover is None or result.cutover.current == BLUE


# ---------------------------------------------------------------------------
# Success
# ---------------------------------------------------------------------------


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_completes_and_retires_blue(self) -> None:
        world = await make_world(rows=150)
        result = await world.orchestrator().run()
        assert result.state.phase is P.COMPLETED
        assert result.exit_code == 0
        assert phases(result) == LIFECYCLE
        assert result.report is not None
        assert result.report.passed
        assert result.cutover is not None
        assert (result.cutover.previous, result.cutover.current) == (BLUE, GREEN)
        assert result.handoff_lsn == await world.source.head_lsn()
        assert await world.aliases() == {ALIAS: GREEN}
        assert not await world.store.collection_exists(BLUE)
        info = await world.store.describe_collection(ALIAS)
        assert (info.spec, info.points_count) == (GREEN_SPEC, 150)
        assert world.gate.is_open
        snap = world.metrics.snapshot()
        assert snap.phase is P.COMPLETED
        assert snap.backfilled == 150
        assert snap.vectors_verified == 150
        assert snap.drain_ms is not None
        assert snap.cutover_ms is not None
        assert snap.verification is not None
        assert snap.verification.passed

    @pytest.mark.asyncio
    async def test_exposes_state_and_ledger(self) -> None:
        world = await make_world(rows=10)
        orchestrator = world.orchestrator()
        assert orchestrator.state.phase is P.PENDING
        result = await orchestrator.run()
        assert orchestrator.state == result.state
        assert orchestrator.ledger.count == 10

    @pytest.mark.asyncio
    async def test_completes_under_live_traffic_with_zero_loss(self) -> None:
        world = await make_world(rows=300, batch_size=16)
        profile = TrafficProfile(
            alias=ALIAS,
            feature_dim=SOURCE_DIM,
            id_range=400,
            seed=3,
            read_interval_s=0.0,
            write_interval_s=0.0,
        )
        traffic = LiveTrafficSimulator(
            world.store, world.source, world.gate, world.metrics, profile
        )
        async with traffic:
            result = await world.orchestrator().run()
            for _ in range(20):  # let writes held by the freeze land
                await asyncio.sleep(0)
        stats = traffic.stats
        assert result.state.phase is P.COMPLETED
        assert stats.read_errors == 0, traffic.last_read_error
        assert stats.dropped == 0
        assert stats.writes_committed > 0
        assert stats.writes_held > 0, "the freeze window must have held some writes"
        assert world.metrics.snapshot().cdc_events > 0

        # Steady-state sync resumes from the handoff LSN; afterwards Green must
        # equal the source of record exactly: no write was lost.
        ledger = ExpectationLedger()
        writer = PointWriter(
            world.store,
            ALIAS,
            projector=world.green_projector,
            ledger=ledger,
            retrier=world.retrier,
            metrics=world.metrics,
        )
        await CdcReconciler(world.source, writer, world.metrics).catch_up_to_head(
            result.handoff_lsn
        )
        actual = await MigrationVerifier(world.store).scan(ALIAS, GREEN_SPEC)
        assert actual == await world.expected_digest()

    @pytest.mark.asyncio
    async def test_survives_transient_store_chaos(self) -> None:
        inner = InMemoryVectorStore()
        chaos = FaultInjectingStore(
            inner,
            FaultPlan(0.2, seed=5, ambiguous_fraction=0.5, methods=ALL_METHODS),
        )
        world = await make_world(rows=100, store=chaos)
        world.retrier = type(world.retrier)(
            world.plan.retry.model_copy(update={"max_attempts": 40}),
            sleep=world.sleep,
            seed=1,
            on_retry=world.metrics.on_retry,
        )
        result = await world.orchestrator().run()
        assert result.state.phase is P.COMPLETED, result.state.error
        assert sum(chaos.injected.values()) > 0
        assert world.metrics.snapshot().retries > 0
        assert await inner.list_aliases() == {ALIAS: GREEN}

    @pytest.mark.asyncio
    async def test_lost_create_response_is_tolerated(self) -> None:
        world, store = await scripted_world()
        lost: list[str] = []

        async def create_then_timeout(name: str, spec: VectorSpec) -> None:
            await InMemoryVectorStore.create_collection(store, name, spec)
            if name == GREEN and not lost:  # first Green create commits, reply is lost
                lost.append(name)
                raise StoreTimeoutError("response lost")

        store.create_collection = create_then_timeout  # type: ignore[method-assign]
        result = await world.orchestrator().run()
        assert result.state.phase is P.COMPLETED


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


class TestRollback:
    @pytest.mark.asyncio
    async def test_verification_failure_rolls_back_without_touching_alias(self) -> None:
        world, store = await scripted_world()
        dropped: list[int] = []

        # Silently loses one point: a store bug that verification must catch.
        async def lossy_upsert(collection: str, points: Sequence[PointRecord]) -> None:
            if collection == GREEN and not dropped:
                dropped.append(points[0].point_id)
                points = points[1:]
            await InMemoryVectorStore.upsert(store, collection, points)

        store.upsert = lossy_upsert  # type: ignore[method-assign]
        aliases_before = store.calls.get("update_aliases", 0)
        result = await world.orchestrator().run()

        assert result.state.phase is P.ROLLED_BACK
        assert result.exit_code == 1
        assert result.state.error is not None
        assert "verification failed" in result.state.error
        assert result.report is not None
        assert not result.report.passed
        assert result.cutover is None
        assert store.calls.get("update_aliases", 0) == aliases_before, "alias never touched"
        assert phases(result)[-3:] == [P.VERIFYING, P.FAILED, P.ROLLED_BACK]
        await assert_rolled_back_to_blue(world, result)
        assert world.metrics.snapshot().verification is not None

    @pytest.mark.asyncio
    async def test_target_already_exists_is_not_deleted(self) -> None:
        world, store = await scripted_world()
        await store.create_collection(GREEN, GREEN_SPEC)
        result = await world.orchestrator().run()
        assert result.state.phase is P.ROLLED_BACK
        assert result.state.error is not None
        assert "already exists" in result.state.error
        assert await store.collection_exists(GREEN), "someone else's collection is left alone"
        assert await world.aliases() == {ALIAS: BLUE}

    @pytest.mark.asyncio
    async def test_concurrent_conflicting_create_is_rejected(self) -> None:
        world, store = await scripted_world()

        def someone_else_creates(name: str, spec: VectorSpec) -> None:
            if name == GREEN:
                # Another actor creates Green (wrong spec) right before our create lands.
                store.hooks.pop("create_collection")
                store._collections[GREEN] = type(store._collections[BLUE])(
                    VectorSpec(dimension=3, distance=spec.distance)
                )

        store.hooks["create_collection"] = someone_else_creates
        result = await world.orchestrator().run()
        assert result.state.phase is P.ROLLED_BACK
        assert result.state.error is not None
        assert "appeared concurrently" in result.state.error
        assert await store.collection_exists(GREEN)

    @pytest.mark.asyncio
    async def test_cutover_rejected_rolls_back(self) -> None:
        world, store = await scripted_world()

        def reject(actions: Sequence[AliasAction]) -> None:
            if targets_green(actions):
                raise ServerError("alias service down", status_code=503)

        store.hooks["update_aliases"] = reject
        result = await world.orchestrator().run()
        assert result.state.phase is P.ROLLED_BACK
        assert result.state.error is not None
        assert "CutoverError" in result.state.error
        assert phases(result)[-3:] == [P.CUTTING_OVER, P.FAILED, P.ROLLED_BACK]
        await assert_rolled_back_to_blue(world, result)

    @pytest.mark.asyncio
    async def test_partial_swap_with_failed_restore_is_repaired(self) -> None:
        world, store = await scripted_world()
        script = ["partial", "fail_restore"]

        def misbehave(actions: Sequence[AliasAction]) -> None:
            # 1st call, the swap: the delete applies, the create is lost.
            # 2nd call, the coordinator's restore: fails.
            # 3rd call, the orchestrator's compensation: succeeds.
            step = script.pop(0) if script else "ok"
            if step == "partial":
                store._aliases.pop(ALIAS)
                raise ServerError("half applied", status_code=500)
            if step == "fail_restore":
                raise ServerError("still failing", status_code=500)

        store.hooks["update_aliases"] = misbehave
        result = await world.orchestrator().run()
        assert result.state.phase is P.ROLLED_BACK, result.state.error
        await assert_rolled_back_to_blue(world, result)

    @pytest.mark.asyncio
    async def test_alias_hijacked_without_initial_binding_is_unbound_again(self) -> None:
        store = ScriptedStore()
        world = await make_world(store=store, bootstrap=False)
        await store.create_collection("intruder", GREEN_SPEC)
        script = ["hijack", "fail_restore"]

        def misbehave(actions: Sequence[AliasAction]) -> None:
            step = script.pop(0) if script else "ok"
            if step == "hijack":
                store._aliases[ALIAS] = "intruder"
                raise ServerError("misrouted", status_code=500)
            if step == "fail_restore":
                raise ServerError("still failing", status_code=500)

        store.hooks["update_aliases"] = misbehave
        result = await world.orchestrator().run()
        assert result.state.phase is P.ROLLED_BACK, result.state.error
        assert await world.aliases() == {}, "restored to the initial state: no binding"
        assert not await store.collection_exists(GREEN)

    @pytest.mark.asyncio
    async def test_silently_ignored_repair_ends_aborted(self) -> None:
        world, store = await scripted_world()
        calls = [0]

        async def flaky_aliases(actions: Sequence[AliasAction]) -> None:
            calls[0] += 1
            if calls[0] == 1:  # the swap: delete applied, create lost
                store._aliases.pop(ALIAS)
                raise ServerError("half applied", status_code=500)
            if calls[0] == 2:  # the coordinator's restore
                raise ServerError("still failing", status_code=500)
            # the repair is acknowledged but never applied

        store.update_aliases = flaky_aliases  # type: ignore[method-assign]
        result = await world.orchestrator().run()
        assert result.state.phase is P.ABORTED
        assert ALIAS not in await world.aliases()
        assert await store.collection_exists(GREEN)

    @pytest.mark.asyncio
    async def test_unrepairable_alias_ends_aborted_and_keeps_green(self) -> None:
        world, store = await scripted_world()

        def break_aliases(actions: Sequence[AliasAction]) -> None:
            if targets_green(actions):
                store._aliases.pop(ALIAS, None)
            if store.calls["update_aliases"] > 1:
                raise ServerError("alias service down", status_code=503)

        store.hooks["update_aliases"] = break_aliases
        result = await world.orchestrator().run()
        assert result.state.phase is P.ABORTED
        assert result.exit_code == 1
        assert result.state.error is not None
        assert ALIAS not in await world.aliases()
        assert await store.collection_exists(GREEN), "never drop Green without a safe alias"
        assert world.gate.is_open

    @pytest.mark.asyncio
    async def test_failed_green_cleanup_ends_aborted(self) -> None:
        world, store = await scripted_world()

        # A tier-1 count mismatch fails verification.
        async def short_count(collection: str) -> int:
            n = await InMemoryVectorStore.count(store, collection)
            return n - 1 if collection == GREEN else n

        def refuse_drop(name: str) -> None:
            if name == GREEN:
                raise PermanentError("drop refused")

        store.count = short_count  # type: ignore[method-assign]
        store.hooks["delete_collection"] = refuse_drop
        result = await world.orchestrator().run()
        assert result.state.phase is P.ABORTED
        assert await world.aliases() == {ALIAS: BLUE}
        assert await store.collection_exists(GREEN)

    @pytest.mark.asyncio
    async def test_drain_timeout_rolls_back(self) -> None:
        world = await make_world()
        release, entered = asyncio.Event(), asyncio.Event()

        async def stuck_writer() -> None:
            async with world.gate.write():
                entered.set()
                await release.wait()

        writer = asyncio.create_task(stuck_writer())
        await entered.wait()
        result = await world.orchestrator(drain_timeout_s=0.02).run()
        release.set()
        await writer
        assert result.state.phase is P.ROLLED_BACK
        assert result.state.error is not None
        assert "DrainTimeoutError" in result.state.error
        await assert_rolled_back_to_blue(world, result)

    @pytest.mark.asyncio
    async def test_unreadable_aliases_at_start(self) -> None:
        world, store = await scripted_world()

        def down() -> None:
            raise NotFoundError("aliases endpoint missing")

        store.hooks["list_aliases"] = down
        result = await world.orchestrator().run()
        assert result.state.phase is P.ROLLED_BACK
        assert phases(result) == [P.FAILED, P.ROLLED_BACK]
        assert not await store.collection_exists(GREEN)

    @pytest.mark.asyncio
    async def test_unexpected_exception_propagates_with_gate_open(self) -> None:
        world, store = await scripted_world()

        def bug(collection: str) -> None:
            raise KeyError(collection)

        store.hooks["count"] = bug
        with pytest.raises(KeyError):
            await world.orchestrator().run()
        assert world.gate.is_open


# ---------------------------------------------------------------------------
# Cooperative abort
# ---------------------------------------------------------------------------


class TestAbort:
    @pytest.mark.asyncio
    async def test_sigterm_mid_backfill_aborts_and_leaves_alias_on_blue(self) -> None:
        world, store = await scripted_world(rows=200)
        batches = [0]

        def signal_on_third(collection: str, points: Sequence[PointRecord]) -> None:
            if collection == GREEN:
                batches[0] += 1
                if batches[0] == 3:
                    world.shutdown.request(int(signal.SIGTERM))

        store.hooks["upsert"] = signal_on_third
        result = await world.orchestrator().run()
        assert result.state.phase is P.ABORTED
        assert result.aborted_by_signal == signal.SIGTERM
        assert result.exit_code == 143
        assert phases(result)[-2:] == [P.BACKFILLING, P.ABORTED]
        assert result.state.backfill_cursor == 3 * world.plan.batch_size
        assert result.state.history[-1].reason.startswith("abort requested by signal 15")
        await assert_rolled_back_to_blue(world, result)
        assert world.metrics.snapshot().backfilled == 3 * world.plan.batch_size

    @pytest.mark.asyncio
    async def test_real_sigint_aborts_at_checkpoint(self) -> None:
        world, store = await scripted_world()
        sent = [False]

        def interrupt(collection: str, points: Sequence[PointRecord]) -> None:
            if collection == GREEN and not sent[0]:
                sent[0] = True
                os.kill(os.getpid(), signal.SIGINT)

        store.hooks["upsert"] = interrupt
        with world.shutdown.installed():
            result = await world.orchestrator().run()
        assert result.state.phase is P.ABORTED
        assert result.exit_code == 130
        await assert_rolled_back_to_blue(world, result)

    @pytest.mark.asyncio
    async def test_abort_inside_freeze_window_reopens_gate(self) -> None:
        world, store = await scripted_world()

        def signal_during_verify(collection: str) -> None:
            if collection == GREEN:
                world.shutdown.request(int(signal.SIGTERM))

        store.hooks["count"] = signal_during_verify
        result = await world.orchestrator().run()
        assert result.state.phase is P.ABORTED
        assert phases(result)[-2:] == [P.VERIFYING, P.ABORTED]
        await assert_rolled_back_to_blue(world, result)

    @pytest.mark.asyncio
    async def test_signal_during_cutover_is_deferred_and_run_completes(self) -> None:
        world, store = await scripted_world()

        def signal_mid_swap(actions: Sequence[AliasAction]) -> None:
            if targets_green(actions):
                world.shutdown.request(int(signal.SIGINT))

        store.hooks["update_aliases"] = signal_mid_swap
        result = await world.orchestrator().run()
        assert result.state.phase is P.COMPLETED
        assert result.exit_code == 0
        assert world.shutdown.received == [int(signal.SIGINT)]
        assert await world.aliases() == {ALIAS: GREEN}

    @pytest.mark.asyncio
    async def test_abort_before_anything_was_created(self) -> None:
        world = await make_world()
        world.shutdown.request(int(signal.SIGTERM))
        result = await world.orchestrator().run()
        assert result.state.phase is P.ABORTED
        assert phases(result) == [P.PROVISIONING, P.ABORTED]
        assert await world.aliases() == {ALIAS: BLUE}


# ---------------------------------------------------------------------------
# Decommission
# ---------------------------------------------------------------------------


class TestDecommission:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error", [PermanentError("drop refused"), StoreTimeoutError("drop timed out")]
    )
    async def test_failure_after_cutover_cannot_roll_back(self, error: Exception) -> None:
        world, store = await scripted_world()

        def refuse(name: str) -> None:
            if name == BLUE:
                raise error

        store.hooks["delete_collection"] = refuse
        result = await world.orchestrator().run()
        assert result.state.phase is P.ABORTED
        assert result.exit_code == 1
        assert result.aborted_by_signal is None
        assert phases(result)[-3:] == [P.DECOMMISSIONING, P.FAILED, P.ABORTED]
        assert result.state.error is not None
        assert "left for manual cleanup" in result.state.error
        assert await world.aliases() == {ALIAS: GREEN}, "traffic stays on verified Green"
        assert await store.collection_exists(BLUE)
        assert result.report is not None
        assert result.report.passed
