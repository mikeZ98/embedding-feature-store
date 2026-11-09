"""Projection (normalisation, payload, dimension checks) and two-tier verification."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence

import pytest
from hypothesis import given
from hypothesis import strategies as st

from embedding_feature_store.core.errors import (
    InvalidVectorError,
    SchemaMismatchError,
    VerificationError,
)
from embedding_feature_store.core.projection import PointProjector, l2_normalise
from embedding_feature_store.core.verification import (
    ExpectationLedger,
    MigrationVerifier,
    MultisetDigest,
)
from embedding_feature_store.schemas import (
    DistanceMetric,
    PointRecord,
    ScrollPage,
    SourceRecord,
    VectorSpec,
    to_float32,
)
from tests.fakes import FakeEmbedder, InMemoryVectorStore

COSINE3 = VectorSpec(dimension=3, distance=DistanceMetric.COSINE)
DOT3 = VectorSpec(dimension=3, distance=DistanceMetric.DOT)

components = st.floats(-1e6, 1e6, allow_nan=False)
vectors3 = st.tuples(components, components, components).filter(lambda v: math.hypot(*v) > 1e-3)
identities = st.text(max_size=12)


def row(rid: int = 1, features: tuple[float, ...] = (3.0, 0.0, 4.0), lsn: int = 7) -> SourceRecord:
    return SourceRecord(record_id=rid, tenant="acme", features=features, lsn=lsn)


def point(pid: int, *vector: float) -> PointRecord:
    return PointRecord(point_id=pid, vector=vector)


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


class TestNormalise:
    def test_unit_length(self) -> None:
        assert l2_normalise((3.0, 4.0)) == (0.6, 0.8)

    @pytest.mark.parametrize("vector", [(0.0, 0.0), (math.inf, 1.0), (math.nan, 1.0)])
    def test_rejects_degenerate_norm(self, vector: tuple[float, ...]) -> None:
        norm = math.hypot(*vector)
        with pytest.raises(InvalidVectorError) as exc:
            l2_normalise(vector)
        assert str(exc.value) == f"cannot normalise a vector with norm {norm}"

    def test_large_components_do_not_overflow(self) -> None:
        unit = l2_normalise((1e308, 1e308))
        assert math.isclose(math.hypot(*unit), 1.0)

    @given(vectors3)
    def test_cosine_points_are_unit_after_float32(self, vector: tuple[float, float, float]) -> None:
        projector = PointProjector(COSINE3, FakeEmbedder(dimension=3))
        stored = projector.build_point(row(), vector).vector
        assert math.isclose(math.hypot(*stored), 1.0, rel_tol=1e-6)


class TestProjector:
    def test_embedder_dimension_must_match_spec(self) -> None:
        with pytest.raises(SchemaMismatchError, match="produces 2-d"):
            PointProjector(COSINE3, FakeEmbedder(dimension=2))

    def test_spec_exposed(self) -> None:
        assert PointProjector(DOT3, FakeEmbedder(dimension=3)).spec is DOT3

    def test_cosine_prenormalised(self) -> None:
        p = PointProjector(COSINE3, FakeEmbedder(dimension=3)).build_point(row(), (3.0, 0.0, 4.0))
        assert p.vector == to_float32((0.6, 0.0, 0.8))

    def test_dot_stored_as_float32_without_normalising(self) -> None:
        p = PointProjector(DOT3, FakeEmbedder(dimension=3)).build_point(row(), (3.0, 0.1, 4.0))
        assert p.vector == to_float32((3.0, 0.1, 4.0))

    def test_payload_and_id(self) -> None:
        p = PointProjector(DOT3, FakeEmbedder(dimension=3, model_id="m@2")).build_point(
            row(rid=42, lsn=9), (1.0, 2.0, 3.0)
        )
        assert p.point_id == 42
        assert p.payload == (("model_id", "m@2"), ("source_lsn", 9), ("tenant", "acme"))

    def test_identity_tracks_model_and_version(self) -> None:
        a = PointProjector(DOT3, FakeEmbedder(dimension=3, model_id="m@1"))
        b = PointProjector(DOT3, FakeEmbedder(dimension=3, model_id="m@2"))
        v = (1.0, 2.0, 3.0)
        base = a.build_point(row(lsn=1), v).canonical_identity
        assert base != b.build_point(row(lsn=1), v).canonical_identity
        assert base != a.build_point(row(lsn=2), v).canonical_identity

    @pytest.mark.parametrize("vector", [(1.0, 2.0), (1.0, 2.0, 3.0, 4.0)])
    def test_dimension_error(self, vector: tuple[float, ...]) -> None:
        projector = PointProjector(DOT3, FakeEmbedder(dimension=3))
        with pytest.raises(SchemaMismatchError, match="expected 3"):
            projector.build_point(row(), vector)

    @pytest.mark.parametrize("bad", [math.nan, math.inf])
    def test_non_finite_rejected(self, bad: float) -> None:
        projector = PointProjector(DOT3, FakeEmbedder(dimension=3))
        with pytest.raises(InvalidVectorError, match="non-finite"):
            projector.build_point(row(), (1.0, bad, 0.0))

    def test_float32_overflow_rejected(self) -> None:
        projector = PointProjector(DOT3, FakeEmbedder(dimension=3))
        with pytest.raises(InvalidVectorError, match="cannot be stored"):
            projector.build_point(row(), (1e39, 0.0, 0.0))

    def test_zero_vector_rejected_under_cosine(self) -> None:
        projector = PointProjector(COSINE3, FakeEmbedder(dimension=3))
        with pytest.raises(InvalidVectorError):
            projector.build_point(row(), (0.0, 0.0, 0.0))

    @pytest.mark.asyncio
    async def test_batch_preserves_order(self) -> None:
        embedder = FakeEmbedder(dimension=3, transform=lambda f: [x * 2 for x in f])
        projector = PointProjector(DOT3, embedder)
        records = [row(rid=i, features=(float(i), 1.0, 1.0)) for i in (3, 1, 2)]
        points = await projector.project(records)
        assert [p.point_id for p in points] == [3, 1, 2]
        assert points[0].vector == (6.0, 2.0, 2.0)

    @pytest.mark.asyncio
    async def test_empty_batch_skips_embedder(self) -> None:
        embedder = FakeEmbedder(dimension=3)
        assert await PointProjector(DOT3, embedder).project([]) == ()
        assert embedder.calls == 0

    @pytest.mark.asyncio
    async def test_embedder_arity_mismatch(self) -> None:
        projector = PointProjector(DOT3, FakeEmbedder(dimension=3, drop_last=True))
        with pytest.raises(SchemaMismatchError, match="returned 1 vectors for 2"):
            await projector.project([row(rid=1), row(rid=2)])


# ---------------------------------------------------------------------------
# MultisetDigest
# ---------------------------------------------------------------------------


class TestMultisetDigest:
    @given(st.lists(identities), st.randoms(use_true_random=False))
    def test_permutation_invariant(self, items: list[str], rnd: object) -> None:
        shuffled = items[:]
        rnd.shuffle(shuffled)  # type: ignore[attr-defined]
        a, b = MultisetDigest.of(items), MultisetDigest.of(shuffled)
        assert a == b
        assert a.hexdigest() == b.hexdigest()
        assert hash(a) == hash(b)

    @given(st.lists(identities), identities)
    def test_remove_inverts_add(self, items: list[str], extra: str) -> None:
        digest = MultisetDigest.of(items)
        before = digest.hexdigest()
        digest.add(extra)
        assert digest.hexdigest() != before
        digest.remove(extra)
        assert digest.hexdigest() == before

    @given(st.lists(identities, min_size=1, unique=True))
    def test_multiplicity_matters(self, items: list[str]) -> None:
        assert MultisetDigest.of(items) != MultisetDigest.of([*items, items[0]])

    def test_sensitive_to_content(self) -> None:
        assert MultisetDigest.of(["a", "b"]) != MultisetDigest.of(["a", "c"])

    def test_empty(self) -> None:
        digest = MultisetDigest()
        assert digest.count == 0
        assert len(digest.hexdigest()) == 64
        with pytest.raises(ValueError, match="empty"):
            digest.remove("a")

    def test_not_equal_to_other_types(self) -> None:
        assert MultisetDigest() != "digest"

    def test_repr(self) -> None:
        assert repr(MultisetDigest.of(["a"])).startswith("MultisetDigest(count=1, hex=")


# ---------------------------------------------------------------------------
# ExpectationLedger
# ---------------------------------------------------------------------------


class TestLedger:
    def test_last_write_wins(self) -> None:
        ledger = ExpectationLedger()
        ledger.record_upsert(point(1, 1.0))
        ledger.record_upsert(point(1, 2.0))
        ledger.record_upsert(point(2, 3.0))
        assert ledger.count == 2
        expected = MultisetDigest.of(
            [point(1, 2.0).canonical_identity, point(2, 3.0).canonical_identity]
        )
        assert ledger.digest == expected

    def test_delete_and_delete_missing(self) -> None:
        ledger = ExpectationLedger()
        ledger.record_upsert(point(1, 1.0))
        ledger.record_delete(1)
        ledger.record_delete(99)
        assert ledger.count == 0
        assert ledger.digest == MultisetDigest()


# ---------------------------------------------------------------------------
# MigrationVerifier
# ---------------------------------------------------------------------------


async def seeded_store(points: Sequence[PointRecord], spec: VectorSpec) -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    await store.create_collection("dst", spec)
    await store.upsert("dst", points)
    return store


def ledger_of(points: Sequence[PointRecord]) -> ExpectationLedger:
    ledger = ExpectationLedger()
    for p in points:
        ledger.record_upsert(p)
    return ledger


SPEC1 = VectorSpec(dimension=1, distance=DistanceMetric.DOT)
POINTS = [point(i, float(i)) for i in range(10)]


class TestVerifier:
    def test_page_size_validated(self) -> None:
        with pytest.raises(ValueError, match="page_size"):
            MigrationVerifier(InMemoryVectorStore(), page_size=0)

    @pytest.mark.asyncio
    async def test_passes_across_pages(self) -> None:
        store = await seeded_store(POINTS, SPEC1)
        report = await MigrationVerifier(store, page_size=3).require(
            "dst", SPEC1, ledger_of(POINTS)
        )
        assert report.passed
        assert report.source_count == report.target_count == 10
        assert report.source_checksum == report.target_checksum

    @pytest.mark.asyncio
    async def test_empty_collection_matches_empty_ledger(self) -> None:
        store = await seeded_store([], SPEC1)
        report = await MigrationVerifier(store).verify("dst", SPEC1, ExpectationLedger())
        assert report.passed

    @pytest.mark.asyncio
    async def test_count_mismatch_fails(self) -> None:
        store = await seeded_store(POINTS[:-1], SPEC1)
        report = await MigrationVerifier(store).verify("dst", SPEC1, ledger_of(POINTS))
        assert not report.passed
        assert (report.source_count, report.target_count) == (10, 9)

    @pytest.mark.asyncio
    async def test_content_mismatch_with_equal_count_fails(self) -> None:
        tampered = [*POINTS[:-1], point(9, 9.5)]
        store = await seeded_store(tampered, SPEC1)
        report = await MigrationVerifier(store).verify("dst", SPEC1, ledger_of(POINTS))
        assert not report.passed
        assert report.source_count == report.target_count
        assert report.source_checksum != report.target_checksum

    @pytest.mark.asyncio
    async def test_require_raises_with_report(self) -> None:
        store = await seeded_store(POINTS[:5], SPEC1)
        with pytest.raises(VerificationError) as exc:
            await MigrationVerifier(store).require("dst", SPEC1, ledger_of(POINTS))
        assert exc.value.report is not None
        assert not exc.value.report.passed

    @pytest.mark.asyncio
    async def test_dimension_error_raised(self) -> None:
        store = await seeded_store([point(1, 1.0), point(2, 1.0, 2.0)], SPEC1)
        with pytest.raises(SchemaMismatchError, match="point 2 has 2 dimensions"):
            await MigrationVerifier(store).scan("dst", SPEC1)

    @pytest.mark.asyncio
    async def test_non_advancing_scroll_detected(self) -> None:
        class StuckStore(InMemoryVectorStore):
            async def scroll(
                self, collection: str, *, offset: int | None, limit: int
            ) -> ScrollPage:
                return ScrollPage(next_offset=5)

        with pytest.raises(VerificationError, match="did not advance past offset 5"):
            await MigrationVerifier(StuckStore()).scan("dst", SPEC1)


ops = st.lists(
    st.tuples(st.booleans(), st.integers(0, 15), st.floats(-10, 10, allow_nan=False)),
    max_size=60,
)


@given(ops, st.integers(1, 7))
def test_ledger_tracks_any_upsert_delete_history(
    history: list[tuple[bool, int, float]], page_size: int
) -> None:
    async def scenario() -> bool:
        store = await seeded_store([], SPEC1)
        ledger = ExpectationLedger()
        for is_upsert, pid, value in history:
            if is_upsert:
                p = point(pid, value)
                await store.upsert("dst", [p])
                ledger.record_upsert(p)
            else:
                await store.delete_points("dst", [pid])
                ledger.record_delete(pid)
        report = await MigrationVerifier(store, page_size=page_size).verify("dst", SPEC1, ledger)
        return report.passed

    assert asyncio.run(scenario())
