"""Producer/consumer bridge over a bounded :class:`asyncio.Queue`.

:func:`bounded_stream` runs ``source`` in a background producer task that
pushes into a queue of at most ``capacity`` items. When the consumer falls
behind, the queue fills and the producer blocks on ``put`` — so backpressure
reaches the source and at most ``capacity`` items (plus the one being
produced) are ever buffered.

Producer exceptions travel *in band*: items produced before the failure are
delivered first, then the exception is raised from the consumer's iteration.

The returned object is an async generator. Close it when stopping early —
``async with contextlib.aclosing(bounded_stream(...)) as rows`` — which cancels
the producer and ``aclose()``s ``source`` if it is itself an async generator,
releasing its cursor.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator, AsyncIterable
from dataclasses import dataclass
from typing import Final

from embedding_feature_store.core.ports import QueueObserver

__all__ = ["bounded_stream"]


@dataclass(frozen=True, slots=True)
class _Failure:
    error: Exception


class _End:
    __slots__ = ()


_END: Final = _End()


async def bounded_stream[T](
    source: AsyncIterable[T],
    *,
    capacity: int,
    observer: QueueObserver | None = None,
) -> AsyncGenerator[T]:
    """Re-yield ``source`` through a bounded queue filled by a producer task."""
    if capacity < 1:
        msg = "capacity must be >= 1"
        raise ValueError(msg)
    queue: asyncio.Queue[T | _Failure | _End] = asyncio.Queue(maxsize=capacity)

    def notify(*, blocked: bool) -> None:
        if observer is not None:
            observer(queue.qsize(), capacity, blocked)

    async def put(message: T | _Failure | _End) -> None:
        if queue.full():
            notify(blocked=True)
            await queue.put(message)
        else:
            queue.put_nowait(message)
        notify(blocked=False)

    async def produce() -> None:
        try:
            async for item in source:
                await put(item)
        except Exception as exc:  # noqa: BLE001 - forwarded in band to the consumer
            await put(_Failure(exc))
            return
        finally:
            if isinstance(source, AsyncGenerator):
                await source.aclose()
        await put(_END)

    producer = asyncio.create_task(produce())
    try:
        while True:
            message = await queue.get()
            notify(blocked=False)
            if isinstance(message, _End):
                return
            if isinstance(message, _Failure):
                raise message.error
            yield message
    finally:
        producer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await producer
