"""Cooperative SIGINT/SIGTERM handling.

The first signal *requests* an abort; nothing is interrupted. Long-running work
calls :meth:`ShutdownController.checkpoint` at safe points (between batches,
between phases), which raises :class:`AbortRequestedError` once an abort is
pending — so a migration only ever stops between fully applied units of work
and can run its compensation.

Inside :meth:`ShutdownController.critical` (the atomic cutover and what follows
it) checkpoints never raise: the work is committed to finishing, and signals
are only recorded. Signals after the first are recorded and otherwise ignored.

Exit codes follow the shell convention ``128 + signum`` (130 for SIGINT, 143
for SIGTERM).
"""

from __future__ import annotations

import asyncio
import logging
import signal
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

from embedding_feature_store.core.errors import AbortRequestedError

__all__ = ["SIGNAL_EXIT_BASE", "ShutdownController"]

SIGNAL_EXIT_BASE = 128
_log = logging.getLogger(__name__)


class ShutdownController:
    def __init__(self, signals: Sequence[signal.Signals] = (signal.SIGINT, signal.SIGTERM)) -> None:
        self._signals = tuple(signals)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._signum: int | None = None
        self._critical = 0
        self.received: list[int] = []

    @property
    def requested(self) -> bool:
        return self._signum is not None

    @property
    def signum(self) -> int | None:
        return self._signum

    @property
    def exit_code(self) -> int | None:
        return None if self._signum is None else SIGNAL_EXIT_BASE + self._signum

    @property
    def in_critical_section(self) -> bool:
        return self._critical > 0

    def request(self, signum: int) -> None:
        """Signal handler body; also callable directly (tests, programmatic stop)."""
        self.received.append(signum)
        if self._signum is not None:
            _log.warning("signal %d ignored: abort already pending", signum)
            return
        self._signum = signum
        if self.in_critical_section:
            _log.warning("signal %d deferred: inside critical section", signum)
        else:
            _log.warning("signal %d: aborting at the next checkpoint", signum)

    def checkpoint(self, where: str) -> None:
        """Raise :class:`AbortRequestedError` if an abort is pending and allowed here."""
        if self._signum is not None and not self.in_critical_section:
            raise AbortRequestedError(self._signum, where)

    @contextmanager
    def critical(self) -> Iterator[None]:
        """Defer aborts for the duration of the block (re-entrant)."""
        self._critical += 1
        try:
            yield
        finally:
            self._critical -= 1

    def install(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Route the configured signals to :meth:`request` on ``loop``."""
        if self._loop is not None:
            msg = "signal handlers already installed"
            raise RuntimeError(msg)
        self._loop = loop if loop is not None else asyncio.get_running_loop()
        for sig in self._signals:
            self._loop.add_signal_handler(sig, self.request, int(sig))

    def uninstall(self) -> None:
        """Restore default handling of the configured signals. Idempotent."""
        if self._loop is None:
            return
        for sig in self._signals:
            self._loop.remove_signal_handler(sig)
        self._loop = None

    @contextmanager
    def installed(self) -> Iterator[ShutdownController]:
        self.install()
        try:
            yield self
        finally:
            self.uninstall()
