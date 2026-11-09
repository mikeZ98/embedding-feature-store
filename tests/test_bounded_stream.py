"""Bounded producer/consumer bridge: bounds, backpressure, errors, cleanup."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator, AsyncIterator

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from embedding_feature_store.storage.bounded_stream import bounded_stream


class Telemetry:
    def __init__(self) -> None:
        self.events: list[tuple[int, int, bool]] = []

    def __call__(self, depth: int, capacity: int, blocked: bool) -> None:
        self.events.append((depth, capacity, blocked))

    @property
    def max_depth(self) -> int:
        return max((d for d, _, _ in self.events), default=0)


class Source:
    """Async generator source that records production and closure."""

    def __init__(self, items: list[int], *, fail_after: int | None = None) -> None:
        self.items = items
        self.fail_after = fail_after
        self.produced = 0
        self.closed = False
        self.error = ValueError("source exploded")

    async def __call__(self) -> AsyncGenerator[int]:
        try:
            for item in self.items:
                if self.fail_after is not None and self.produced == self.fail_after:
                    raise self.error
                self.produced += 1
                yield item
        finally:
            self.closed = True


class PlainIterator:
    """A non-generator async iterator (has no aclose)."""

    def __init__(self, n: int) -> None:
        self.i = 0
        self.n = n

    def __aiter__(self) -> AsyncIterator[int]:
        return self

    async def __anext__(self) -> int:
        if self.i >= self.n:
            raise StopAsyncIteration
        self.i += 1
        return self.i


async def settle(rounds: int = 20) -> None:
    for _ in range(rounds):
        await asyncio.sleep(0)


async def collect(stream: AsyncIterator[int]) -> list[int]:
    return [item async for item in stream]


@pytest.mark.asyncio
async def test_capacity_validated() -> None:
    with pytest.raises(ValueError, match="capacity"):
        await collect(bounded_stream(Source([1])(), capacity=0))


@pytest.mark.asyncio
async def test_passes_items_through_in_order() -> None:
    source = Source(list(range(50)))
    assert await collect(bounded_stream(source(), capacity=3)) == list(range(50))
    assert source.closed


@pytest.mark.asyncio
async def test_empty_source() -> None:
    assert await collect(bounded_stream(Source([])(), capacity=1)) == []


@pytest.mark.asyncio
async def test_plain_async_iterator_source() -> None:
    assert await collect(bounded_stream(PlainIterator(4), capacity=2)) == [1, 2, 3, 4]


@pytest.mark.asyncio
async def test_backpressure_blocks_producer_at_capacity() -> None:
    telemetry = Telemetry()
    source = Source(list(range(100)))
    stream = bounded_stream(source(), capacity=4, observer=telemetry)
    first = await anext(stream)
    await settle()
    # 1 consumed + 4 queued + 1 held by the producer blocked on put().
    assert first == 0
    assert source.produced == 6
    assert telemetry.max_depth == 4
    assert (4, 4, True) in telemetry.events
    await stream.aclose()


@pytest.mark.asyncio
async def test_slow_consumer_bounds_lag() -> None:
    telemetry = Telemetry()
    source = Source(list(range(40)))
    received = 0
    async with contextlib.aclosing(bounded_stream(source(), capacity=3, observer=telemetry)) as s:
        async for _ in s:
            received += 1
            await settle(3)
            assert source.produced - received <= 3 + 1
    assert received == 40
    assert telemetry.max_depth <= 3
    assert all(cap == 3 for _, cap, _ in telemetry.events)


@pytest.mark.asyncio
async def test_producer_error_delivered_after_preceding_items() -> None:
    source = Source(list(range(10)), fail_after=3)
    stream = bounded_stream(source(), capacity=2)
    received = [await anext(stream) for _ in range(3)]
    with pytest.raises(ValueError, match="source exploded") as exc:
        await anext(stream)
    assert received == [0, 1, 2]
    assert exc.value is source.error
    assert source.closed


@pytest.mark.asyncio
async def test_producer_error_waits_behind_full_queue() -> None:
    source = Source(list(range(10)), fail_after=5)
    stream = bounded_stream(source(), capacity=1)
    received = [await anext(stream)]
    await settle()
    received += [await anext(stream) for _ in range(4)]
    with pytest.raises(ValueError, match="source exploded"):
        await anext(stream)
    assert received == [0, 1, 2, 3, 4]


@pytest.mark.asyncio
async def test_early_exit_closes_source_and_producer() -> None:
    source = Source(list(range(1000)))
    stream = bounded_stream(source(), capacity=2)
    assert await anext(stream) == 0
    await stream.aclose()
    assert source.closed
    produced = source.produced
    await settle()
    assert source.produced == produced, "producer must stop after aclose()"


@pytest.mark.asyncio
async def test_aclose_before_start_is_harmless() -> None:
    source = Source([1, 2])
    stream = bounded_stream(source(), capacity=1)
    await stream.aclose()
    assert source.produced == 0


@pytest.mark.asyncio
async def test_consumer_cancellation_cleans_up() -> None:
    source = Source(list(range(1000)))
    started = asyncio.Event()

    async def consume() -> None:
        # aclosing() is the documented contract: a bare ``async for`` leaves the
        # generator suspended until garbage collection when its body is cancelled.
        async with contextlib.aclosing(bounded_stream(source(), capacity=2)) as stream:
            async for _ in stream:
                started.set()
                await asyncio.Event().wait()  # stall until cancelled

    task = asyncio.create_task(consume())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert source.closed


@settings(max_examples=40)
@given(st.integers(1, 6), st.lists(st.integers(), max_size=60), st.lists(st.integers(0, 3)))
def test_lossless_and_bounded(capacity: int, items: list[int], pauses: list[int]) -> None:
    async def scenario() -> tuple[list[int], int]:
        telemetry = Telemetry()

        async def source() -> AsyncGenerator[int]:
            for i, item in enumerate(items):
                if pauses and pauses[i % len(pauses)]:
                    await asyncio.sleep(0)
                yield item

        out: list[int] = []
        async for item in bounded_stream(source(), capacity=capacity, observer=telemetry):
            out.append(item)
            if pauses:
                await settle(pauses[len(out) % len(pauses)])
        return out, telemetry.max_depth

    out, max_depth = asyncio.run(scenario())
    assert out == items
    assert max_depth <= capacity
