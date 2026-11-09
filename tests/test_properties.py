"""Property-based fuzzing of the invariants verification and replay depend on.

* float32 canonicalisation at its exact IEEE-754 boundaries (overflow
  threshold, subnormal underflow, rounding error, sign symmetry);
* canonical identity: equal exactly when (id, vector, typed payload) are equal;
* MultisetDigest: permutation invariance and add/remove cancellation under
  arbitrary interleavings, checked against a ``Counter`` model;
* the keyset cursor: ascending, gap-free, full pages, parameterised SQL;
* the WAL: strictly increasing LSNs under any paged or resumed replay, and
  replaying a prefix reproduces the MVCC snapshot at that LSN.

Complements the example-based tests in the per-module files.
"""

from __future__ import annotations

import asyncio
import math
import re
import struct
from collections import Counter
from typing import Any

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from pydantic import ValidationError
from pydantic_core import PydanticCustomError

from embedding_feature_store.core.verification import MultisetDigest
from embedding_feature_store.schemas import CdcEvent, PayloadValue, PointRecord, to_float32
from embedding_feature_store.storage.postgres_sim import KeysetQuery, PostgresSimulator

# ---------------------------------------------------------------------------
# float32 boundaries
# ---------------------------------------------------------------------------

FLT_MAX = struct.unpack("<f", b"\xff\xff\x7f\x7f")[0]  # (2 - 2**-23) * 2**127
# FLT_MAX plus half an ulp. Its mantissa is odd, so the tie rounds to even (2**128 = inf):
# every finite |x| below this rounds to at most FLT_MAX, everything at or above overflows.
OVERFLOW_AT = 2.0**128 - 2.0**103
MIN_SUBNORMAL = 2.0**-149
UNDERFLOW_TIE = 2.0**-150  # half the smallest subnormal: rounds (to even) to zero
MIN_NORMAL = 2.0**-126

BOUNDARY_VALUES = [
    0.0,
    -0.0,
    MIN_SUBNORMAL,
    -MIN_SUBNORMAL,
    MIN_NORMAL,
    -MIN_NORMAL,
    MIN_NORMAL - MIN_SUBNORMAL,  # largest subnormal
    1.0,
    math.nextafter(1.0, 2.0),
    FLT_MAX,
    -FLT_MAX,
    math.nextafter(OVERFLOW_AT, 0.0),
    -math.nextafter(OVERFLOW_AT, 0.0),
]

finite64 = st.floats(allow_nan=False, allow_infinity=False)
in_range = st.floats(
    min_value=-OVERFLOW_AT, max_value=OVERFLOW_AT, exclude_min=True, exclude_max=True
)
boundary_vectors = st.lists(
    st.sampled_from(BOUNDARY_VALUES) | st.floats(width=32, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=8,
)


def f32(x: float) -> float:
    (value,) = to_float32((x,))
    return value


class TestFloat32Boundaries:
    @given(st.floats(min_value=FLT_MAX, max_value=OVERFLOW_AT, exclude_max=True))
    def test_just_below_the_overflow_threshold_saturates_to_flt_max(self, x: float) -> None:
        assert f32(x) == FLT_MAX
        assert f32(-x) == -FLT_MAX

    @given(st.floats(min_value=OVERFLOW_AT, allow_infinity=False), st.booleans())
    def test_at_or_above_the_threshold_is_rejected(self, x: float, negate: bool) -> None:
        with pytest.raises(PydanticCustomError) as exc:
            to_float32((1.0, -x if negate else x))
        assert exc.value.type == "float32_overflow"

    @given(st.floats(min_value=-UNDERFLOW_TIE, max_value=UNDERFLOW_TIE))
    def test_underflow_is_positive_zero(self, x: float) -> None:
        value = f32(x)
        assert value == 0.0
        assert math.copysign(1.0, value) == 1.0, "negative zero must be normalised"

    @given(st.floats(min_value=UNDERFLOW_TIE, max_value=MIN_SUBNORMAL, exclude_min=True))
    def test_just_above_the_underflow_tie_is_the_smallest_subnormal(self, x: float) -> None:
        assert f32(x) == MIN_SUBNORMAL
        assert f32(-x) == -MIN_SUBNORMAL

    @given(in_range)
    def test_rounding_error_is_at_most_half_an_ulp(self, x: float) -> None:
        # Normal range: 24-bit significand. Subnormal range: absolute spacing 2**-149.
        assert abs(f32(x) - x) <= max(abs(x) * 2.0**-24, UNDERFLOW_TIE)

    @given(in_range, in_range)
    def test_rounding_is_monotonic(self, x: float, y: float) -> None:
        lo, hi = sorted((x, y))
        assert f32(lo) <= f32(hi)

    @given(in_range)
    def test_sign_symmetric(self, x: float) -> None:
        assert f32(-x) == -f32(x)

    @given(boundary_vectors)
    def test_boundary_vectors_survive_json_with_identical_identity(
        self, vector: list[float]
    ) -> None:
        record = PointRecord(point_id=1, vector=tuple(vector))
        clone = PointRecord.model_validate_json(record.model_dump_json())
        assert clone.vector == record.vector
        assert clone.canonical_identity == record.canonical_identity

    @given(st.lists(finite64, min_size=1, max_size=8))
    def test_record_accepts_exactly_what_to_float32_accepts(self, vector: list[float]) -> None:
        try:
            expected: tuple[float, ...] | None = to_float32(tuple(vector))
        except PydanticCustomError:
            expected = None
        if expected is None:
            with pytest.raises(ValidationError):
                PointRecord(point_id=1, vector=tuple(vector))
        else:
            assert PointRecord(point_id=1, vector=tuple(vector)).vector == expected


# ---------------------------------------------------------------------------
# Canonical identity
# ---------------------------------------------------------------------------

SHA256_HEX = re.compile(r"[0-9a-f]{64}")

# A deliberately tiny domain, so that equal and unequal pairs both occur often.
# 0, 0.0, False and "0" are ==-equal or look alike but must hash differently.
small_values: st.SearchStrategy[PayloadValue] = st.sampled_from(
    [None, False, True, 0, 1, 0.0, 1.0, -0.0, "0", "1", ""]
)
small_payloads = st.dictionaries(st.sampled_from(["a", "b"]), small_values, max_size=2)
small_records = st.builds(
    lambda pid, vec, payload: PointRecord(point_id=pid, vector=vec, payload=tuple(payload.items())),
    st.integers(0, 2),
    st.sampled_from([(0.0,), (-0.0,), (1.0,), (0.0, 1.0), (1.0, 0.0), (MIN_SUBNORMAL,)]),
    small_payloads,
)


def typed_key(record: PointRecord) -> tuple[Any, ...]:
    """What identity must capture: ids, float32 vector values, and payload *types*."""
    payload = tuple((k, type(v).__name__, v) for k, v in record.payload)
    return (record.point_id, record.vector, payload)


class TestCanonicalIdentity:
    @given(small_records, small_records)
    def test_equal_exactly_when_id_vector_and_typed_payload_are_equal(
        self, a: PointRecord, b: PointRecord
    ) -> None:
        assert (a.canonical_identity == b.canonical_identity) == (typed_key(a) == typed_key(b))

    @given(small_records)
    def test_is_a_sha256_hex_digest(self, record: PointRecord) -> None:
        assert SHA256_HEX.fullmatch(record.canonical_identity)

    @given(st.integers(min_value=0, max_value=2**53), st.text(min_size=1, max_size=4))
    def test_payload_scalar_types_never_collide(self, n: int, key: str) -> None:
        variants: list[PayloadValue] = [n, float(n), str(n)]
        if n in {0, 1}:
            variants.append(bool(n))
        identities = {
            PointRecord(point_id=1, vector=(1.0,), payload=((key, v),)).canonical_identity
            for v in variants
        }
        assert len(identities) == len(variants)


# ---------------------------------------------------------------------------
# MultisetDigest
# ---------------------------------------------------------------------------

alphabet = st.text(alphabet="abc", max_size=2)


@st.composite
def interleavings(draw: st.DrawFn) -> tuple[list[tuple[str, str]], Counter[str]]:
    """A sequence of add/remove ops (removes only of present items) and its result."""
    ops: list[tuple[str, str]] = []
    model: Counter[str] = Counter()
    for _ in range(draw(st.integers(0, 40))):
        present = sorted(model.elements())
        remove = draw(st.booleans()) if present else False
        if remove:
            item = draw(st.sampled_from(present))
            ops.append(("remove", item))
            model[item] -= 1
        else:
            item = draw(alphabet)
            ops.append(("add", item))
            model[item] += 1
    return ops, +model


class TestMultisetDigest:
    @given(interleavings())
    def test_any_interleaving_equals_the_digest_of_the_final_multiset(
        self, case: tuple[list[tuple[str, str]], Counter[str]]
    ) -> None:
        ops, model = case
        digest = MultisetDigest()
        for op, item in ops:
            if op == "add":
                digest.add(item)
            else:
                digest.remove(item)
        expected = MultisetDigest.of(model.elements())
        assert digest == expected
        assert digest.hexdigest() == expected.hexdigest()
        assert digest.count == model.total()

    @given(st.lists(alphabet), st.lists(alphabet), st.randoms(use_true_random=False))
    def test_adding_then_removing_a_batch_in_any_order_cancels(
        self, base: list[str], batch: list[str], rnd: Any
    ) -> None:
        digest = MultisetDigest.of(base)
        before = digest.hexdigest()
        for item in batch:
            digest.add(item)
        undo = list(batch)
        rnd.shuffle(undo)
        for item in undo:
            digest.remove(item)
        assert digest.hexdigest() == before

    @given(st.lists(alphabet), st.lists(alphabet))
    def test_equal_exactly_when_the_multisets_are_equal(self, a: list[str], b: list[str]) -> None:
        assert (MultisetDigest.of(a) == MultisetDigest.of(b)) == (Counter(a) == Counter(b))

    @given(st.lists(alphabet), st.randoms(use_true_random=False))
    def test_permutation_invariant_hexdigest(self, items: list[str], rnd: Any) -> None:
        shuffled = list(items)
        rnd.shuffle(shuffled)
        assert MultisetDigest.of(items).hexdigest() == MultisetDigest.of(shuffled).hexdigest()


# ---------------------------------------------------------------------------
# Keyset pagination and WAL replay
# ---------------------------------------------------------------------------

STATEMENTS = {
    "SELECT * FROM features ORDER BY id LIMIT $1",
    "SELECT * FROM features WHERE id > $1 ORDER BY id LIMIT $2",
}


async def stream(
    db: PostgresSimulator, *, snapshot_lsn: int, after_id: int | None, batch_size: int
) -> list[list[int]]:
    rows = db.stream_rows(snapshot_lsn=snapshot_lsn, after_id=after_id, batch_size=batch_size)
    return [[r.record_id for r in batch] async for batch in rows]


class TestKeysetPagination:
    @given(
        st.sets(st.integers(1, 10**6), max_size=40),
        st.integers(1, 7),
        st.none() | st.integers(0, 10**6),
    )
    def test_pages_are_ascending_full_and_resume_from_the_last_id(
        self, ids: set[int], batch_size: int, after_id: int | None
    ) -> None:
        db = PostgresSimulator()
        for rid in sorted(ids, reverse=True):  # insertion order must not matter
            db.upsert(rid, "t", (float(rid),))

        batches = asyncio.run(
            stream(db, snapshot_lsn=len(ids), after_id=after_id, batch_size=batch_size)
        )
        flat = [rid for batch in batches for rid in batch]
        expected = sorted(i for i in ids if after_id is None or i > after_id)

        assert flat == expected, "ascending, no gaps, no duplicates"
        assert all(batches), "no empty pages are yielded"
        assert all(len(b) == batch_size for b in batches[:-1]), "only the last page is short"
        assert len(db.queries) == len(expected) // batch_size + 1

        resumed_from = [q.after_id for q in db.queries]
        assert resumed_from == [after_id, *(b[-1] for b in batches)][: len(db.queries)]
        assert all(isinstance(q, KeysetQuery) for q in db.queries)
        assert all(q.statement in STATEMENTS for q in db.queries), "values are bound, not inlined"
        assert all(q.params[-1] == batch_size for q in db.queries)


wal_ops = st.lists(
    st.tuples(st.booleans(), st.integers(1, 6), st.integers(-3, 3)), min_size=1, max_size=40
)


def build_wal(history: list[tuple[bool, int, int]]) -> PostgresSimulator:
    db = PostgresSimulator()
    for is_upsert, rid, value in history:
        if is_upsert:
            db.upsert(rid, "t", (float(value),))
        else:
            db.delete(rid)  # deleting a missing row is a no-op with no WAL entry
    return db


async def paged_replay(db: PostgresSimulator, start: int, limits: list[int]) -> list[CdcEvent]:
    """Tail the WAL from ``start``, resuming from the last LSN seen each time."""
    seen: list[CdcEvent] = []
    cursor = start
    for limit in limits:
        page = await db.changes_since(cursor, limit=limit)
        if not page:
            break
        seen.extend(page)
        cursor = page[-1].lsn
    return seen


class TestWalReplay:
    @given(wal_ops, st.data())
    def test_paged_replay_from_any_lsn_is_strictly_increasing_and_gap_free(
        self, history: list[tuple[bool, int, int]], data: st.DataObject
    ) -> None:
        db = build_wal(history)
        head = asyncio.run(db.head_lsn())
        start = data.draw(st.integers(0, head), label="start")
        limits = data.draw(st.lists(st.integers(1, 5), min_size=head + 1, max_size=head + 1))

        lsns = [e.lsn for e in asyncio.run(paged_replay(db, start, limits))]
        assert lsns == list(range(start + 1, head + 1))

    @given(wal_ops, st.data())
    def test_resuming_after_a_crash_never_skips_or_repeats(
        self, history: list[tuple[bool, int, int]], data: st.DataObject
    ) -> None:
        db = build_wal(history)
        head = asyncio.run(db.head_lsn())
        everything = asyncio.run(db.changes_since(0, limit=head + 1))
        checkpoint = data.draw(st.integers(0, head), label="checkpoint")

        before = [e for e in everything if e.lsn <= checkpoint]
        after = asyncio.run(db.changes_since(checkpoint, limit=head + 1))
        assert [*before, *after] == list(everything)

    @given(wal_ops)
    def test_replaying_any_prefix_reproduces_the_snapshot_at_that_lsn(
        self, history: list[tuple[bool, int, int]]
    ) -> None:
        db = build_wal(history)
        head = asyncio.run(db.head_lsn())
        events = asyncio.run(db.changes_since(0, limit=head + 1))

        table: dict[int, Any] = {}
        assert db.snapshot(0) == table
        for event in events:
            if event.record is None:
                del table[event.record_id]  # a logged delete always hits a live row
            else:
                assert event.record.lsn == event.lsn, "rows are stamped with their commit LSN"
                table[event.record_id] = event.record
            assert db.snapshot(event.lsn) == table
        assert all(row.lsn <= head for row in table.values())

    @given(wal_ops)
    def test_snapshot_rows_never_come_from_the_future(
        self, history: list[tuple[bool, int, int]]
    ) -> None:
        db = build_wal(history)
        head = asyncio.run(db.head_lsn())
        assume(head > 0)
        for lsn in range(head + 1):
            assert all(row.lsn <= lsn for row in db.snapshot(lsn).values())
