"""Atomic alias cutover inside a write freeze window.

Sequence::

    pre-flight (outside the window): target exists, read current binding
    close gate -> drain in-flight writes
        update_aliases([delete alias, create alias -> target])   # one atomic call
        confirm the binding by reading it back
    reopen gate (always, also on failure)

The read-back makes the swap safe under ambiguous failures: if
``update_aliases`` raises but the store *did* apply it (e.g. a timeout after
commit), the cutover is reported as successful. If the alias ends up bound to
neither the old nor the new collection, the coordinator restores the previous
binding before raising :class:`CutoverError`, so readers are never left
without a collection.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from embedding_feature_store.core.errors import CutoverError, FeatureStoreError
from embedding_feature_store.core.ports import Clock, VectorStore
from embedding_feature_store.core.retry import Retrier
from embedding_feature_store.core.write_gate import WriteGate
from embedding_feature_store.schemas import AliasAction, CreateAliasAction, DeleteAliasAction

__all__ = ["AliasCutoverCoordinator", "CutoverOutcome"]


@dataclass(frozen=True, slots=True)
class CutoverOutcome:
    alias: str
    previous: str | None
    current: str | None
    freeze_s: float


def _rebind_actions(alias: str, previous: str | None, target: str | None) -> list[AliasAction]:
    actions: list[AliasAction] = []
    if previous is not None:
        actions.append(DeleteAliasAction(alias=alias))
    if target is not None:
        actions.append(CreateAliasAction(alias=alias, collection=target))
    return actions


class AliasCutoverCoordinator:
    def __init__(
        self,
        store: VectorStore,
        gate: WriteGate,
        retrier: Retrier,
        *,
        drain_timeout_s: float,
        clock: Clock = time.monotonic,
    ) -> None:
        self._store = store
        self._gate = gate
        self._retrier = retrier
        self._drain_timeout_s = drain_timeout_s
        self._clock = clock

    async def _binding(self, alias: str) -> str | None:
        aliases = await self._retrier.run(self._store.list_aliases)
        return aliases.get(alias)

    async def cutover(self, alias: str, target: str) -> CutoverOutcome:
        """Atomically point ``alias`` at ``target``."""
        if not await self._retrier.run(lambda: self._store.collection_exists(target)):
            msg = f"cannot cut {alias!r} over to missing collection {target!r}"
            raise CutoverError(msg)
        return await self._rebind(alias, target)

    async def rollback(self, outcome: CutoverOutcome) -> CutoverOutcome:
        """Restore the binding that ``outcome`` replaced (removing the alias if none)."""
        return await self._rebind(outcome.alias, outcome.previous)

    async def _rebind(self, alias: str, target: str | None) -> CutoverOutcome:
        previous = await self._binding(alias)
        if previous == target:
            return CutoverOutcome(alias=alias, previous=previous, current=target, freeze_s=0.0)

        started = self._clock()
        async with self._gate.frozen(self._drain_timeout_s):
            swap_error: FeatureStoreError | None = None
            try:
                await self._store.update_aliases(_rebind_actions(alias, previous, target))
            except FeatureStoreError as exc:
                swap_error = exc
            bound = await self._binding(alias)
            if bound != target:
                if bound != previous:
                    await self._restore(alias, bound, previous)
                msg = (
                    f"alias {alias!r} not rebound to {target!r} "
                    f"(restored to {previous!r}): {swap_error!r}"
                )
                raise CutoverError(msg) from swap_error
        return CutoverOutcome(
            alias=alias, previous=previous, current=target, freeze_s=self._clock() - started
        )

    async def _restore(self, alias: str, bound: str | None, previous: str | None) -> None:
        try:
            await self._store.update_aliases(_rebind_actions(alias, bound, previous))
        except FeatureStoreError as exc:
            msg = f"alias {alias!r} left bound to {bound!r}; restoring {previous!r} failed"
            raise CutoverError(msg) from exc
        if await self._binding(alias) != previous:
            msg = f"alias {alias!r} could not be restored to {previous!r}"
            raise CutoverError(msg)
