"""A write gate that can be frozen without losing writes.

Writers wrap each write in ``async with gate.write():``. While the gate is
open, writers enter immediately and are counted in :attr:`WriteGate.in_flight`.

:meth:`WriteGate.close_and_drain` closes the gate and waits for every in-flight
write to finish. Writers that arrive while the gate is closed are *held*, not
rejected, and enter once :meth:`WriteGate.reopen` is called — so a freeze
window delays writes but never drops them. A writer only gives up (with the
retryable :class:`GateClosedError`) if it passed its own ``timeout_s``.

If draining times out the gate reopens itself before raising
:class:`DrainTimeoutError`, so a failed freeze never strands writers.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from embedding_feature_store.core.errors import DrainTimeoutError, GateClosedError, GateStateError
from embedding_feature_store.core.ports import QueueObserver

__all__ = ["WriteGate"]


class WriteGate:
    def __init__(self, *, observer: QueueObserver | None = None) -> None:
        self._open = asyncio.Event()
        self._open.set()
        self._idle = asyncio.Event()
        self._idle.set()
        self._in_flight = 0
        self._observer = observer

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def is_open(self) -> bool:
        return self._open.is_set()

    def _set_in_flight(self, value: int) -> None:
        self._in_flight = value
        if value == 0:
            self._idle.set()
        else:
            self._idle.clear()
        if self._observer is not None:
            self._observer(value)

    async def _wait_open(self) -> None:
        # Loop: another task may close the gate between our wake-up and resumption.
        while not self._open.is_set():
            await self._open.wait()

    @asynccontextmanager
    async def write(self, *, timeout_s: float | None = None) -> AsyncIterator[None]:
        """Hold a write slot for the duration of the block.

        Waits (up to ``timeout_s``; forever if None) while the gate is closed.
        """
        try:
            async with asyncio.timeout(timeout_s):
                await self._wait_open()
        except TimeoutError:
            msg = f"write gate still closed after {timeout_s}s"
            raise GateClosedError(msg) from None
        self._set_in_flight(self._in_flight + 1)
        try:
            yield
        finally:
            self._set_in_flight(self._in_flight - 1)

    async def close_and_drain(self, timeout_s: float) -> None:
        """Close the gate and wait until no write is in flight.

        Raises :class:`DrainTimeoutError` (after reopening the gate) if writes
        are still in flight after ``timeout_s``.
        """
        if not self._open.is_set():
            msg = "write gate is already closed"
            raise GateStateError(msg)
        self._open.clear()
        try:
            async with asyncio.timeout(timeout_s):
                await self._idle.wait()
        except TimeoutError:
            stuck = self._in_flight
            self.reopen()
            msg = f"{stuck} write(s) still in flight after {timeout_s}s"
            raise DrainTimeoutError(msg, in_flight=stuck) from None

    def reopen(self) -> None:
        """Open the gate and release held writers. Idempotent."""
        self._open.set()

    @asynccontextmanager
    async def frozen(self, timeout_s: float) -> AsyncIterator[None]:
        """Close and drain for the duration of the block; always reopen on exit."""
        await self.close_and_drain(timeout_s)
        try:
            yield
        finally:
            self.reopen()
