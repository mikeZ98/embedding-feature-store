"""Write gate: concurrent writers, drain, timeouts, reopening, zero write loss."""

from __future__ import annotations

import asyncio

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from embedding_feature_store.core.errors import (
    DrainTimeoutError,
    GateClosedError,
    GateStateError,
    TransientError,
)
from embedding_feature_store.core.write_gate import WriteGate


async def hold(gate: WriteGate, release: asyncio.Event, entered: asyncio.Event) -> None:
    async with gate.write():
        entered.set()
        await release.wait()


async def settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_starts_open_and_idle() -> None:
    gate = WriteGate()
    assert gate.is_open
    assert gate.in_flight == 0


@pytest.mark.asyncio
async def test_counts_in_flight_and_notifies_observer() -> None:
    depths: list[int] = []
    gate = WriteGate(observer=depths.append)
    async with gate.write():
        assert gate.in_flight == 1
        async with gate.write():
            assert gate.in_flight == 2
    assert gate.in_flight == 0
    assert depths == [1, 2, 1, 0]


@pytest.mark.asyncio
async def test_slot_released_when_write_raises() -> None:
    gate = WriteGate()
    with pytest.raises(RuntimeError):
        async with gate.write():
            raise RuntimeError
    assert gate.in_flight == 0


@pytest.mark.asyncio
async def test_drain_waits_for_in_flight_writes() -> None:
    gate = WriteGate()
    release, entered = asyncio.Event(), asyncio.Event()
    writer = asyncio.create_task(hold(gate, release, entered))
    await entered.wait()

    drain = asyncio.create_task(gate.close_and_drain(timeout_s=5))
    await settle()
    assert not drain.done()
    assert not gate.is_open
    assert gate.in_flight == 1

    release.set()
    await drain
    await writer
    assert gate.in_flight == 0
    assert not gate.is_open


@pytest.mark.asyncio
async def test_writers_are_held_not_rejected_while_closed() -> None:
    gate = WriteGate()
    await gate.close_and_drain(timeout_s=1)
    done: list[int] = []

    async def write(i: int) -> None:
        async with gate.write():
            done.append(i)

    tasks = [asyncio.create_task(write(i)) for i in range(5)]
    await settle()
    assert done == []
    assert gate.in_flight == 0
    gate.reopen()
    await asyncio.gather(*tasks)
    assert sorted(done) == list(range(5))


@pytest.mark.asyncio
async def test_writer_timeout_while_closed() -> None:
    gate = WriteGate()
    await gate.close_and_drain(timeout_s=1)
    with pytest.raises(GateClosedError) as exc:
        async with gate.write(timeout_s=0.01):
            pytest.fail("entered a closed gate")
    assert isinstance(exc.value, TransientError)
    assert gate.in_flight == 0


@pytest.mark.asyncio
async def test_writer_timeout_irrelevant_when_open() -> None:
    gate = WriteGate()
    async with gate.write(timeout_s=0.0):
        assert gate.in_flight == 1


@pytest.mark.asyncio
async def test_drain_timeout_reopens_gate() -> None:
    gate = WriteGate()
    release, entered = asyncio.Event(), asyncio.Event()
    writer = asyncio.create_task(hold(gate, release, entered))
    await entered.wait()

    with pytest.raises(DrainTimeoutError) as exc:
        await gate.close_and_drain(timeout_s=0.01)
    assert exc.value.in_flight == 1
    assert gate.is_open, "a failed freeze must not strand writers"

    release.set()
    await writer
    assert gate.in_flight == 0


@pytest.mark.asyncio
async def test_double_close_rejected() -> None:
    gate = WriteGate()
    await gate.close_and_drain(timeout_s=1)
    with pytest.raises(GateStateError):
        await gate.close_and_drain(timeout_s=1)


@pytest.mark.asyncio
async def test_reopen_is_idempotent() -> None:
    gate = WriteGate()
    gate.reopen()
    await gate.close_and_drain(timeout_s=1)
    gate.reopen()
    gate.reopen()
    assert gate.is_open


@pytest.mark.asyncio
async def test_frozen_reopens_on_error() -> None:
    gate = WriteGate()

    async def failing_body() -> None:
        async with gate.frozen(timeout_s=1):
            assert not gate.is_open
            raise RuntimeError

    with pytest.raises(RuntimeError):
        await failing_body()
    assert gate.is_open


@pytest.mark.asyncio
async def test_reclosed_before_waiter_resumes() -> None:
    """A held writer woken by reopen() must re-check: the gate may have closed again."""
    gate = WriteGate()
    await gate.close_and_drain(timeout_s=1)
    entered = asyncio.Event()

    async def write() -> None:
        async with gate.write():
            entered.set()

    task = asyncio.create_task(write())
    await settle()
    gate.reopen()
    await gate.close_and_drain(timeout_s=1)  # completes without yielding: gate is idle
    await settle()
    assert not entered.is_set()
    gate.reopen()
    await task
    assert entered.is_set()


async def run_freeze_scenario(durations: list[int], freeze_after: int) -> tuple[int, int, int]:
    """Writers of varying length race a freeze window.

    Returns (writes completed, writes observed inside the frozen window,
    in_flight at the moment the drain returned).
    """
    gate = WriteGate()
    frozen = False
    violations = 0
    completed = 0
    drained_in_flight = -1

    async def writer(ticks: int) -> None:
        nonlocal violations, completed
        async with gate.write():
            for _ in range(ticks):
                if frozen:
                    violations += 1
                await asyncio.sleep(0)
            completed += 1

    async def freezer() -> None:
        nonlocal frozen, drained_in_flight
        for _ in range(freeze_after):
            await asyncio.sleep(0)
        async with gate.frozen(timeout_s=5):
            drained_in_flight = gate.in_flight
            frozen = True
            for _ in range(3):
                await asyncio.sleep(0)
            frozen = False

    await asyncio.gather(freezer(), *(writer(d) for d in durations))
    return completed, violations, drained_in_flight


@settings(max_examples=50)
@given(st.lists(st.integers(0, 6), min_size=1, max_size=25), st.integers(0, 8))
def test_zero_write_loss_under_freeze(durations: list[int], freeze_after: int) -> None:
    completed, violations, drained = asyncio.run(run_freeze_scenario(durations, freeze_after))
    assert completed == len(durations)
    assert violations == 0
    assert drained == 0
