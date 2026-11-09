"""Alias cutover: atomic swap, freeze-window lifecycle, ambiguous failures, rollback."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest

from embedding_feature_store.core.cutover import AliasCutoverCoordinator, CutoverOutcome
from embedding_feature_store.core.errors import (
    CutoverError,
    DrainTimeoutError,
    ServerError,
    StoreConnectionError,
    StoreTimeoutError,
)
from embedding_feature_store.core.retry import Retrier
from embedding_feature_store.core.write_gate import WriteGate
from embedding_feature_store.schemas import (
    CreateAliasAction,
    DeleteAliasAction,
    DistanceMetric,
    RetryPolicy,
    VectorSpec,
)
from tests.fakes import AliasFault, InMemoryVectorStore, RecordingSleep

SPEC = VectorSpec(dimension=2, distance=DistanceMetric.DOT)


def make(
    aliases: dict[str, str] | None = None, *, clock: Iterator[float] | None = None
) -> tuple[AliasCutoverCoordinator, InMemoryVectorStore, WriteGate]:
    store = InMemoryVectorStore(aliases=dict(aliases or {}))
    for name in ("old", "new", "other"):
        store.collections[name] = (SPEC, {})
    gate = WriteGate()
    store.observed_gate = gate
    retrier = Retrier(RetryPolicy(max_attempts=3), sleep=RecordingSleep(), seed=0)
    ticks = clock if clock is not None else iter(float(i) for i in range(1000))
    coordinator = AliasCutoverCoordinator(
        store, gate, retrier, drain_timeout_s=0.05, clock=lambda: next(ticks)
    )
    return coordinator, store, gate


class TestSuccessfulCutover:
    @pytest.mark.asyncio
    async def test_swap_is_one_atomic_call_inside_a_drained_freeze(self) -> None:
        coordinator, store, gate = make({"live": "old"})
        outcome = await coordinator.cutover("live", "new")

        assert store.aliases == {"live": "new"}
        assert outcome == CutoverOutcome(alias="live", previous="old", current="new", freeze_s=1.0)
        (call,) = store.alias_calls
        assert call.actions == (
            DeleteAliasAction(alias="live"),
            CreateAliasAction(alias="live", collection="new"),
        )
        assert call.gate_open is False
        assert call.in_flight == 0
        assert gate.is_open

    @pytest.mark.asyncio
    async def test_first_binding_only_creates(self) -> None:
        coordinator, store, _ = make()
        outcome = await coordinator.cutover("live", "new")
        assert outcome.previous is None
        assert store.alias_calls[0].actions == (CreateAliasAction(alias="live", collection="new"),)

    @pytest.mark.asyncio
    async def test_already_bound_is_a_noop(self) -> None:
        coordinator, store, gate = make({"live": "new"})
        outcome = await coordinator.cutover("live", "new")
        assert outcome == CutoverOutcome(alias="live", previous="new", current="new", freeze_s=0.0)
        assert store.alias_calls == []
        assert gate.is_open

    @pytest.mark.asyncio
    async def test_other_aliases_untouched(self) -> None:
        coordinator, store, _ = make({"live": "old", "shadow": "other"})
        await coordinator.cutover("live", "new")
        assert store.aliases == {"live": "new", "shadow": "other"}

    @pytest.mark.asyncio
    async def test_freeze_duration_from_injected_clock(self) -> None:
        coordinator, _, _ = make({"live": "old"}, clock=iter([10.0, 12.5]))
        outcome = await coordinator.cutover("live", "new")
        assert outcome.freeze_s == 2.5

    @pytest.mark.asyncio
    async def test_transient_alias_reads_are_retried(self) -> None:
        coordinator, store, _ = make({"live": "old"})
        store.list_alias_errors = [StoreConnectionError("reset"), StoreTimeoutError("t")]
        outcome = await coordinator.cutover("live", "new")
        assert outcome.current == "new"


class TestWritesDuringCutover:
    @pytest.mark.asyncio
    async def test_cutover_waits_for_in_flight_write(self) -> None:
        coordinator, store, gate = make({"live": "old"})
        release, entered = asyncio.Event(), asyncio.Event()

        async def slow_write() -> None:
            async with gate.write():
                entered.set()
                await release.wait()

        writer = asyncio.create_task(slow_write())
        await entered.wait()
        coordinator_task = asyncio.create_task(coordinator.cutover("live", "new"))
        for _ in range(5):
            await asyncio.sleep(0)
        assert store.alias_calls == []
        release.set()
        await writer
        await coordinator_task
        assert store.alias_calls[0].in_flight == 0

    @pytest.mark.asyncio
    async def test_writes_arriving_mid_swap_are_held_then_applied(self) -> None:
        coordinator, store, gate = make({"live": "old"})
        applied: list[str] = []
        late: list[asyncio.Task[None]] = []

        async def late_write() -> None:
            async with gate.write():
                applied.append(store.aliases["live"])

        store.on_update_aliases = lambda: late.append(asyncio.create_task(late_write()))
        await coordinator.cutover("live", "new")
        await asyncio.gather(*late)
        assert applied == ["new"], "held write must land after the swap, not be lost"

    @pytest.mark.asyncio
    async def test_drain_timeout_aborts_before_touching_aliases(self) -> None:
        coordinator, store, gate = make({"live": "old"})
        release, entered = asyncio.Event(), asyncio.Event()

        async def stuck_write() -> None:
            async with gate.write():
                entered.set()
                await release.wait()

        writer = asyncio.create_task(stuck_write())
        await entered.wait()
        with pytest.raises(DrainTimeoutError):
            await coordinator.cutover("live", "new")
        assert store.alias_calls == []
        assert store.aliases == {"live": "old"}
        assert gate.is_open
        release.set()
        await writer


class TestFailureSafety:
    @pytest.mark.asyncio
    async def test_missing_target_rejected_before_freeze(self) -> None:
        coordinator, store, gate = make({"live": "old"})
        with pytest.raises(CutoverError, match="missing collection"):
            await coordinator.cutover("live", "absent")
        assert store.alias_calls == []
        assert gate.is_open

    @pytest.mark.asyncio
    async def test_swap_rejected_leaves_old_binding(self) -> None:
        coordinator, store, gate = make({"live": "old"})
        boom = ServerError("down", status_code=503)
        store.alias_faults = [AliasFault("raise_before", boom)]
        with pytest.raises(CutoverError) as exc:
            await coordinator.cutover("live", "new")
        assert exc.value.__cause__ is boom
        assert store.aliases == {"live": "old"}
        assert len(store.alias_calls) == 1, "no restore needed: binding never changed"
        assert gate.is_open

    @pytest.mark.asyncio
    async def test_ambiguous_failure_resolved_by_read_back(self) -> None:
        coordinator, store, gate = make({"live": "old"})
        store.alias_faults = [AliasFault("raise_after", StoreTimeoutError("timeout"))]
        outcome = await coordinator.cutover("live", "new")
        assert outcome.current == "new"
        assert store.aliases == {"live": "new"}
        assert gate.is_open

    @pytest.mark.asyncio
    async def test_silent_noop_detected(self) -> None:
        coordinator, store, _ = make({"live": "old"})
        store.alias_faults = [AliasFault("noop")]
        with pytest.raises(CutoverError) as exc:
            await coordinator.cutover("live", "new")
        assert exc.value.__cause__ is None
        assert store.aliases == {"live": "old"}

    @pytest.mark.asyncio
    async def test_partial_swap_restores_previous_binding(self) -> None:
        coordinator, store, gate = make({"live": "old"})
        store.alias_faults = [AliasFault("partial", ServerError("half", status_code=500))]
        with pytest.raises(CutoverError, match="restored to 'old'"):
            await coordinator.cutover("live", "new")
        assert store.aliases == {"live": "old"}
        restore = store.alias_calls[1]
        assert restore.actions == (CreateAliasAction(alias="live", collection="old"),)
        assert restore.gate_open is False, "restore happens inside the freeze window"
        assert gate.is_open

    @pytest.mark.asyncio
    async def test_restore_failure_reported(self) -> None:
        coordinator, store, gate = make({"live": "old"})
        restore_error = ServerError("still down", status_code=500)
        store.alias_faults = [
            AliasFault("partial", ServerError("half", status_code=500)),
            AliasFault("raise_before", restore_error),
        ]
        with pytest.raises(CutoverError, match="restoring 'old' failed") as exc:
            await coordinator.cutover("live", "new")
        assert exc.value.__cause__ is restore_error
        assert gate.is_open

    @pytest.mark.asyncio
    async def test_unconfirmed_restore_reported(self) -> None:
        coordinator, store, _ = make({"live": "old"})
        store.alias_faults = [
            AliasFault("partial", ServerError("half", status_code=500)),
            AliasFault("noop"),
        ]
        with pytest.raises(CutoverError, match="could not be restored"):
            await coordinator.cutover("live", "new")

    @pytest.mark.asyncio
    async def test_bound_elsewhere_is_rebound_to_previous(self) -> None:
        coordinator, store, _ = make({"live": "old"})

        def hijack() -> None:
            store.aliases["live"] = "other"

        store.on_update_aliases = hijack
        store.alias_faults = [AliasFault("noop"), AliasFault("noop")]
        with pytest.raises(CutoverError):
            await coordinator.cutover("live", "new")
        assert store.alias_calls[1].actions == (
            DeleteAliasAction(alias="live"),
            CreateAliasAction(alias="live", collection="old"),
        )

    @pytest.mark.asyncio
    async def test_unexpected_exception_still_reopens_gate(self) -> None:
        coordinator, store, gate = make({"live": "old"})
        store.alias_faults = [AliasFault("raise_before", KeyError("bug"))]
        with pytest.raises(KeyError):
            await coordinator.cutover("live", "new")
        assert gate.is_open


class TestRollback:
    @pytest.mark.asyncio
    async def test_rollback_restores_previous_collection(self) -> None:
        coordinator, store, gate = make({"live": "old"})
        outcome = await coordinator.cutover("live", "new")
        undone = await coordinator.rollback(outcome)
        assert store.aliases == {"live": "old"}
        assert (undone.previous, undone.current) == ("new", "old")
        assert gate.is_open

    @pytest.mark.asyncio
    async def test_rollback_of_first_binding_removes_alias(self) -> None:
        coordinator, store, _ = make()
        outcome = await coordinator.cutover("live", "new")
        await coordinator.rollback(outcome)
        assert store.aliases == {}
        assert store.alias_calls[-1].actions == (DeleteAliasAction(alias="live"),)

    @pytest.mark.asyncio
    async def test_rollback_is_idempotent(self) -> None:
        coordinator, store, _ = make({"live": "old"})
        outcome = await coordinator.cutover("live", "new")
        await coordinator.rollback(outcome)
        again = await coordinator.rollback(outcome)
        assert again.freeze_s == 0.0
        assert len(store.alias_calls) == 2
