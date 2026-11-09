"""Blue/green migration driver.

Lifecycle (each step a state-machine transition, mirrored into telemetry)::

    PROVISIONING     create the Green collection with the target spec
    BACKFILLING      snapshot at the source head LSN; stream, embed and write it
    RECONCILING      CDC catch-up from the snapshot LSN to the current head
    DRAINING         close the write gate, drain in-flight writes, final CDC
                     catch-up to the (now frozen) head            -- freeze --
    VERIFYING        two-tier verification of Green vs the ledger -- freeze --
    CUTTING_OVER     atomic alias swap Blue -> Green, reopen gate -- freeze --
    DECOMMISSIONING  drop Blue
    COMPLETED

Failure handling:

* Any failure (or failed verification) **before or during** the alias swap
  rolls back: the gate is reopened, the alias is restored to the binding seen
  at start (Blue), Green is dropped, and the run ends ROLLED_BACK with the
  error kept. If the alias cannot be restored or Green cannot be dropped, the
  run ends ABORTED instead, so a partial rollback is never reported as clean.
* A cooperative abort (first SIGINT/SIGTERM) is honoured at checkpoints
  before the swap and compensated the same way, ending ABORTED.
* From CUTTING_OVER on the run is inside a critical section: signals are
  deferred. A failure while dropping Blue cannot be rolled back (traffic is
  already on Green); the run ends ABORTED with Blue left for manual cleanup.

The gate is reopened on every exit path, including unexpected exceptions.
The Green collection is never dropped while the alias points at it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from embedding_feature_store.core.cutover import AliasCutoverCoordinator, CutoverOutcome
from embedding_feature_store.core.errors import (
    AbortRequestedError,
    ConflictError,
    FeatureStoreError,
    SchemaMismatchError,
)
from embedding_feature_store.core.ports import Clock, EmbeddingAdapter, SourceReader, VectorStore
from embedding_feature_store.core.projection import PointProjector
from embedding_feature_store.core.retry import Retrier
from embedding_feature_store.core.state_machine import advance, fail, transition
from embedding_feature_store.core.verification import ExpectationLedger, MigrationVerifier
from embedding_feature_store.core.write_gate import WriteGate
from embedding_feature_store.pipeline.backfill import BackfillEngine
from embedding_feature_store.pipeline.reconcile import CdcReconciler
from embedding_feature_store.pipeline.shutdown import ShutdownController
from embedding_feature_store.pipeline.writer import PointWriter
from embedding_feature_store.schemas import (
    AliasAction,
    CreateAliasAction,
    DeleteAliasAction,
    MigrationPhase,
    MigrationPlan,
    MigrationState,
    VerificationReport,
    evolve,
)
from embedding_feature_store.telemetry.metrics import MigrationMetrics

__all__ = ["MigrationOrchestrator", "MigrationResult"]

_log = logging.getLogger(__name__)
P = MigrationPhase


@dataclass(frozen=True, slots=True)
class MigrationResult:
    state: MigrationState
    report: VerificationReport | None
    cutover: CutoverOutcome | None
    handoff_lsn: int
    """CDC position up to which Green is guaranteed complete; steady-state sync resumes here."""
    aborted_by_signal: int | None

    @property
    def exit_code(self) -> int:
        if self.state.phase is P.COMPLETED:
            return 0
        if self.aborted_by_signal is not None:
            return 128 + self.aborted_by_signal
        return 1


class _VerificationFailedError(FeatureStoreError):
    pass


class MigrationOrchestrator:
    def __init__(  # noqa: PLR0913 - explicit dependencies of the composition root
        self,
        plan: MigrationPlan,
        *,
        store: VectorStore,
        source: SourceReader,
        embedder: EmbeddingAdapter,
        gate: WriteGate,
        metrics: MigrationMetrics,
        shutdown: ShutdownController,
        retrier: Retrier,
        drain_timeout_s: float = 10.0,
        cdc_page_size: int = 512,
        clock: Clock = time.monotonic,
    ) -> None:
        self._plan = plan
        self._store = store
        self._source = source
        self._gate = gate
        self._metrics = metrics
        self._shutdown = shutdown
        self._retrier = retrier
        self._drain_timeout_s = drain_timeout_s
        self._clock = clock
        self._ledger = ExpectationLedger()
        self._projector = PointProjector(plan.target_spec, embedder)
        self._writer = PointWriter(
            store,
            plan.target_collection,
            projector=self._projector,
            ledger=self._ledger,
            retrier=retrier,
            metrics=metrics,
        )
        self._reconciler = CdcReconciler(
            source,
            self._writer,
            metrics,
            page_size=cdc_page_size,
            checkpoint=lambda lsn: shutdown.checkpoint(f"cdc lsn {lsn}"),
        )
        self._coordinator = AliasCutoverCoordinator(
            store, gate, retrier, drain_timeout_s=drain_timeout_s, clock=clock
        )
        self._verifier = MigrationVerifier(store, retrier=retrier)
        self._state = MigrationState(plan=plan)
        self._green_created = False
        self._initial_binding: str | None = None
        self._initial_binding_known = False
        self._report: VerificationReport | None = None
        self._outcome: CutoverOutcome | None = None

    @property
    def state(self) -> MigrationState:
        return self._state

    @property
    def ledger(self) -> ExpectationLedger:
        return self._ledger

    # -- state bookkeeping -------------------------------------------------------

    def _set(self, state: MigrationState) -> None:
        self._state = state
        self._metrics.observe_state(state)
        _log.info("phase %s%s", state.phase.value, f": {state.error}" if state.error else "")

    def _advance(self, reason: str) -> None:
        self._set(advance(self._state, reason=reason))

    def _applied(self, lsn: int) -> None:
        self._state = evolve(self._state, applied_lsn=lsn)
        self._metrics.observe_state(self._state)

    def _result(self, signum: int | None = None) -> MigrationResult:
        return MigrationResult(
            state=self._state,
            report=self._report,
            cutover=self._outcome,
            handoff_lsn=self._state.applied_lsn,
            aborted_by_signal=signum,
        )

    # -- lifecycle ---------------------------------------------------------------

    async def run(self) -> MigrationResult:
        try:
            try:
                await self._pre_cutover()
            except AbortRequestedError as exc:
                await self._compensate()
                self._set(transition(self._state, P.ABORTED, reason=str(exc)))
                return self._result(exc.signum)
            except FeatureStoreError as exc:
                self._set(fail(self._state, f"{type(exc).__name__}: {exc}"))
                if await self._compensate():
                    self._set(transition(self._state, P.ROLLED_BACK, reason="compensated"))
                else:
                    self._set(transition(self._state, P.ABORTED, reason="compensation incomplete"))
                return self._result()
            with self._shutdown.critical():
                await self._decommission()
            return self._result()
        finally:
            self._gate.reopen()

    async def _pre_cutover(self) -> None:
        plan, checkpoint = self._plan, self._shutdown.checkpoint
        aliases = await self._retrier.run(self._store.list_aliases)
        self._initial_binding = aliases.get(plan.alias)
        self._initial_binding_known = True

        self._advance("provision target")
        checkpoint("provisioning")
        await self._provision()

        self._advance("backfill snapshot")
        snapshot_lsn = await self._source.head_lsn()
        self._metrics.set_estimated_total(await self._source.estimate_rows())

        def on_batch(last_id: int) -> None:
            self._state = evolve(self._state, backfill_cursor=last_id)
            checkpoint(f"backfill after id {last_id}")

        engine = BackfillEngine(
            self._source,
            self._writer,
            self._metrics,
            batch_size=plan.batch_size,
            checkpoint=on_batch,
        )
        await engine.run(snapshot_lsn)
        self._applied(snapshot_lsn)

        self._advance("cdc catch-up")
        checkpoint("reconciling")
        self._applied(await self._reconciler.catch_up_to_head(self._state.applied_lsn))

        self._advance("freeze writes")
        checkpoint("draining")
        started = self._clock()
        await self._gate.close_and_drain(self._drain_timeout_s)
        self._metrics.record_drain(self._clock() - started)
        self._applied(await self._reconciler.catch_up_to_head(self._state.applied_lsn))

        self._advance("verify target")
        checkpoint("verifying")
        self._report = await self._verifier.verify(
            plan.target_collection, plan.target_spec, self._ledger
        )
        self._metrics.record_verification(self._report)
        if not self._report.passed:
            msg = (
                f"verification failed: expected {self._report.source_count} points, "
                f"found {self._report.target_count}"
            )
            raise _VerificationFailedError(msg)
        checkpoint("before cutover")

        with self._shutdown.critical():
            self._advance("swap alias")
            self._outcome = await self._coordinator.cutover(
                plan.alias, plan.target_collection, frozen_by_caller=True
            )
            self._metrics.record_cutover(self._outcome.freeze_s)
            self._gate.reopen()

    async def _provision(self) -> None:
        """Create Green; tolerant of a create whose success response was lost.

        Pre-existence is checked first, so a ConflictError *during* the retried
        create can only come from our own earlier attempt that did commit.
        """
        plan, store = self._plan, self._store
        if await self._retrier.run(lambda: store.collection_exists(plan.target_collection)):
            msg = f"target collection {plan.target_collection!r} already exists"
            raise ConflictError(msg)
        try:
            await self._retrier.run(
                lambda: store.create_collection(plan.target_collection, plan.target_spec)
            )
        except ConflictError:
            info = await self._retrier.run(
                lambda: store.describe_collection(plan.target_collection)
            )
            if info.spec != plan.target_spec:
                msg = f"{plan.target_collection!r} appeared concurrently with spec {info.spec}"
                raise SchemaMismatchError(msg) from None
            _log.warning("create of %r was applied by an earlier attempt", plan.target_collection)
        self._green_created = True

    async def _decommission(self) -> None:
        self._advance("drop source collection")
        try:
            await self._retrier.run(
                lambda: self._store.delete_collection(self._plan.source_collection)
            )
        except FeatureStoreError as exc:
            error = (
                f"decommission failed ({type(exc).__name__}: {exc}); traffic is on "
                f"{self._plan.target_collection!r}, {self._plan.source_collection!r} "
                "left for manual cleanup"
            )
            self._set(fail(self._state, error))
            self._set(transition(self._state, P.ABORTED, reason="point of no return passed"))
            return
        self._advance("done")

    async def _compensate(self) -> bool:
        """Undo a pre-cutover migration: reopen writes, restore the alias, drop Green.

        The alias is put back to the binding observed when the run started
        (normally Blue). Green is dropped only once the alias is confirmed not
        to serve it. Returns False when either step could not be completed; the
        caller then ends ABORTED rather than ROLLED_BACK, so the outcome is
        never overstated.
        """
        self._gate.reopen()
        plan = self._plan
        try:
            if self._initial_binding_known:
                bound = (await self._retrier.run(self._store.list_aliases)).get(plan.alias)
                if bound != self._initial_binding:
                    _log.warning(
                        "restoring alias %r: %r -> %r", plan.alias, bound, self._initial_binding
                    )
                    actions: list[AliasAction] = []
                    if bound is not None:
                        actions.append(DeleteAliasAction(alias=plan.alias))
                    if self._initial_binding is not None:
                        actions.append(
                            CreateAliasAction(alias=plan.alias, collection=self._initial_binding)
                        )
                    await self._retrier.run(lambda: self._store.update_aliases(actions))
                    bound = (await self._retrier.run(self._store.list_aliases)).get(plan.alias)
                    if bound != self._initial_binding:
                        _log.error("alias %r could not be restored (now %r)", plan.alias, bound)
                        return False
            if self._green_created:
                await self._retrier.run(
                    lambda: self._store.delete_collection(plan.target_collection)
                )
        except FeatureStoreError as exc:
            _log.error("compensation incomplete: %s", exc)
            return False
        return True
