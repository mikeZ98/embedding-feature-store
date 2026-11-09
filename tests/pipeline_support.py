"""Shared wiring for pipeline tests: a seeded source, a Blue collection, an orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field

from embedding_feature_store.core.ports import VectorStore
from embedding_feature_store.core.projection import PointProjector
from embedding_feature_store.core.retry import Retrier
from embedding_feature_store.core.verification import MultisetDigest
from embedding_feature_store.core.write_gate import WriteGate
from embedding_feature_store.embedding import RandomProjectionEmbedder
from embedding_feature_store.pipeline.bootstrap import bootstrap_blue
from embedding_feature_store.pipeline.orchestrator import MigrationOrchestrator
from embedding_feature_store.pipeline.shutdown import ShutdownController
from embedding_feature_store.schemas import (
    DistanceMetric,
    MigrationPlan,
    RetryPolicy,
    VectorSpec,
)
from embedding_feature_store.storage.memory import InMemoryVectorStore
from embedding_feature_store.storage.postgres_sim import PostgresSimulator
from embedding_feature_store.telemetry.metrics import MigrationMetrics
from tests.fakes import RecordingSleep

ALIAS = "features_active"
BLUE = "features_blue"
GREEN = "features_green"
SOURCE_DIM = 6
BLUE_SPEC = VectorSpec(dimension=4, distance=DistanceMetric.COSINE)
GREEN_SPEC = VectorSpec(dimension=12, distance=DistanceMetric.COSINE)


def make_plan(batch_size: int = 32) -> MigrationPlan:
    return MigrationPlan(
        migration_id="mig-test",
        source_collection=BLUE,
        target_collection=GREEN,
        alias=ALIAS,
        target_spec=GREEN_SPEC,
        batch_size=batch_size,
        retry=RetryPolicy(max_attempts=4),
    )


@dataclass
class World:
    store: VectorStore
    source: PostgresSimulator
    plan: MigrationPlan
    metrics: MigrationMetrics
    gate: WriteGate
    shutdown: ShutdownController
    retrier: Retrier
    green_embedder: RandomProjectionEmbedder
    sleep: RecordingSleep = field(default_factory=RecordingSleep)

    @property
    def green_projector(self) -> PointProjector:
        return PointProjector(self.plan.target_spec, self.green_embedder)

    def orchestrator(
        self, *, drain_timeout_s: float = 5.0, cdc_page_size: int = 64
    ) -> MigrationOrchestrator:
        return MigrationOrchestrator(
            self.plan,
            store=self.store,
            source=self.source,
            embedder=self.green_embedder,
            gate=self.gate,
            metrics=self.metrics,
            shutdown=self.shutdown,
            retrier=self.retrier,
            drain_timeout_s=drain_timeout_s,
            cdc_page_size=cdc_page_size,
        )

    async def aliases(self) -> dict[str, str]:
        return dict(await self.store.list_aliases())

    async def expected_digest(self) -> MultisetDigest:
        """Digest of the source at its head, projected exactly as Green should hold it."""
        head = await self.source.head_lsn()
        records = sorted(self.source.snapshot(head).values(), key=lambda r: r.record_id)
        points = await self.green_projector.project(records)
        return MultisetDigest.of(p.canonical_identity for p in points)


async def make_world(
    *,
    rows: int = 120,
    store: VectorStore | None = None,
    batch_size: int = 32,
    seed: int = 7,
    bootstrap: bool = True,
) -> World:
    store = store if store is not None else InMemoryVectorStore()
    source = PostgresSimulator(queue_capacity=2)
    source.populate(rows, SOURCE_DIM, seed=seed, tenants=("a", "b"))
    plan = make_plan(batch_size)
    metrics = MigrationMetrics(plan)
    sleep = RecordingSleep()
    retrier = Retrier(plan.retry, sleep=sleep, seed=seed, on_retry=metrics.on_retry)
    if bootstrap:
        blue = PointProjector(BLUE_SPEC, RandomProjectionEmbedder(SOURCE_DIM, 4, seed=seed))
        await bootstrap_blue(
            store, source, blue, retrier, collection=BLUE, alias=ALIAS, batch_size=batch_size
        )
    return World(
        store=store,
        source=source,
        plan=plan,
        metrics=metrics,
        gate=WriteGate(),
        shutdown=ShutdownController(),
        retrier=retrier,
        green_embedder=RandomProjectionEmbedder(SOURCE_DIM, GREEN_SPEC.dimension, seed=seed + 1),
        sleep=sleep,
    )
