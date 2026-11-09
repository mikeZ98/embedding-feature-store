"""Unit and property tests for the data contracts."""

from __future__ import annotations

import itertools
import math
import struct
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import TypeAdapter, ValidationError
from pydantic_core import PydanticCustomError

from embedding_feature_store.schemas import (
    TERMINAL_PHASES,
    AliasAction,
    CdcEvent,
    CdcOp,
    CollectionInfo,
    CreateAliasAction,
    DeleteAliasAction,
    DistanceMetric,
    MigrationPhase,
    MigrationPlan,
    MigrationState,
    PayloadValue,
    PhaseTransition,
    PointRecord,
    RetryPolicy,
    ScrollPage,
    SourceRecord,
    VectorSpec,
    VerificationReport,
    canonicalize_payload,
    evolve,
    to_float32,
)

FLT_MAX = struct.unpack("<f", b"\xff\xff\x7f\x7f")[0]
NON_FINITE = [math.nan, math.inf, -math.inf]
SHA_A = "a" * 64
SHA_B = "b" * 64

f32 = st.floats(width=32, allow_nan=False, allow_infinity=False)
f64_in_f32_range = st.floats(min_value=-FLT_MAX, max_value=FLT_MAX)
overflowing = st.floats(min_value=3.5e38, allow_infinity=False, allow_nan=False) | st.floats(
    max_value=-3.5e38, allow_infinity=False, allow_nan=False
)
payload_values: st.SearchStrategy[PayloadValue] = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(max_size=8),
)
payload_dicts = st.dictionaries(st.text(min_size=1, max_size=8), payload_values, max_size=6)


def error_types(exc: ValidationError) -> set[str]:
    return {str(e["type"]) for e in exc.errors()}


def make_plan(**overrides: Any) -> MigrationPlan:
    fields: dict[str, Any] = {
        "migration_id": "m1",
        "source_collection": "src",
        "target_collection": "dst",
        "alias": "live",
        "target_spec": VectorSpec(dimension=3, distance=DistanceMetric.COSINE),
    }
    fields.update(overrides)
    return MigrationPlan(**fields)


def transitions(*phases: MigrationPhase) -> tuple[PhaseTransition, ...]:
    chain = (MigrationPhase.PENDING, *phases)
    return tuple(
        PhaseTransition(sequence=i, from_phase=a, to_phase=b)
        for i, (a, b) in enumerate(itertools.pairwise(chain))
    )


# ---------------------------------------------------------------------------
# Base model behaviour
# ---------------------------------------------------------------------------


class TestFrozenStrictClosed:
    def test_instances_are_frozen(self) -> None:
        spec = VectorSpec(dimension=3, distance=DistanceMetric.DOT)
        with pytest.raises(ValidationError) as exc:
            spec.dimension = 4  # type: ignore[misc]
        assert error_types(exc.value) == {"frozen_instance"}

    def test_unknown_fields_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc:
            VectorSpec(dimension=3, distance=DistanceMetric.DOT, extra=1)  # type: ignore[call-arg]
        assert error_types(exc.value) == {"extra_forbidden"}

    @pytest.mark.parametrize("bad", ["3", 3.0, True])
    def test_strict_int_rejects_coercion(self, bad: object) -> None:
        with pytest.raises(ValidationError):
            VectorSpec(dimension=bad, distance=DistanceMetric.DOT)  # type: ignore[arg-type]

    def test_strict_enum_rejects_raw_string_in_python_mode(self) -> None:
        with pytest.raises(ValidationError):
            VectorSpec(dimension=3, distance="dot")  # type: ignore[arg-type]

    def test_json_mode_accepts_enum_values(self) -> None:
        spec = VectorSpec.model_validate_json('{"dimension": 3, "distance": "euclid"}')
        assert spec.distance is DistanceMetric.EUCLID

    def test_dimension_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            VectorSpec(dimension=0, distance=DistanceMetric.DOT)

    def test_hashable_and_value_equal(self) -> None:
        a = VectorSpec(dimension=3, distance=DistanceMetric.DOT)
        b = VectorSpec(dimension=3, distance=DistanceMetric.DOT)
        assert a == b
        assert hash(a) == hash(b)


# ---------------------------------------------------------------------------
# Float rejection and float32 canonicalisation
# ---------------------------------------------------------------------------


class TestFloatRejection:
    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_vector_rejects_non_finite(self, bad: float) -> None:
        with pytest.raises(ValidationError) as exc:
            PointRecord(point_id=1, vector=(0.0, bad))
        assert error_types(exc.value) == {"finite_number"}

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_features_reject_non_finite(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            SourceRecord(record_id=1, tenant="t", features=(bad,), lsn=0)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_payload_rejects_non_finite(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            PointRecord(point_id=1, vector=(1.0,), payload=(("x", bad),))

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_scalar_floats_reject_non_finite(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            RetryPolicy(base_delay_s=bad)

    @pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
    def test_json_non_finite_rejected(self, token: str) -> None:
        with pytest.raises(ValidationError):
            PointRecord.model_validate_json(f'{{"point_id": 1, "vector": [{token}]}}')

    @given(overflowing)
    def test_vector_rejects_float32_overflow(self, big: float) -> None:
        with pytest.raises(ValidationError) as exc:
            PointRecord(point_id=1, vector=(0.0, big))
        assert error_types(exc.value) == {"float32_overflow"}

    def test_flt_max_is_accepted(self) -> None:
        assert PointRecord(point_id=1, vector=(FLT_MAX, -FLT_MAX)).vector == (FLT_MAX, -FLT_MAX)

    def test_empty_vector_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PointRecord(point_id=1, vector=())

    def test_list_vector_rejected_in_strict_python_mode(self) -> None:
        with pytest.raises(ValidationError):
            PointRecord(point_id=1, vector=[1.0])  # type: ignore[arg-type]


class TestToFloat32:
    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_direct_call_rejects_non_finite(self, bad: float) -> None:
        with pytest.raises(PydanticCustomError) as exc:
            to_float32((1.0, bad))
        assert exc.value.type == "finite_number"

    def test_direct_call_rejects_overflow(self) -> None:
        with pytest.raises(PydanticCustomError) as exc:
            to_float32((1.0, -1e39))
        assert exc.value.type == "float32_overflow"
        assert "-1e+39" in str(exc.value)

    def test_rounds_to_float32(self) -> None:
        (value,) = to_float32((0.1,))
        assert value != 0.1
        assert value == struct.unpack("<f", struct.pack("<f", 0.1))[0]

    def test_negative_zero_normalised(self) -> None:
        (value,) = to_float32((-0.0,))
        assert math.copysign(1.0, value) == 1.0

    def test_ints_become_floats(self) -> None:
        record = PointRecord(point_id=1, vector=(1, 2))
        assert all(type(v) is float for v in record.vector)

    @given(st.lists(f32, min_size=1, max_size=16))
    def test_exact_float32_values_are_preserved(self, values: list[float]) -> None:
        assert to_float32(tuple(values)) == tuple(values)

    @given(st.lists(f64_in_f32_range, min_size=1, max_size=16))
    def test_idempotent(self, values: list[float]) -> None:
        once = to_float32(tuple(values))
        assert to_float32(once) == once
        assert all(abs(v) <= FLT_MAX for v in once)


# ---------------------------------------------------------------------------
# Payload canonicalisation and identity
# ---------------------------------------------------------------------------


class TestPayload:
    def test_sorted_by_key(self) -> None:
        record = PointRecord(point_id=1, vector=(1.0,), payload=(("b", 2), ("a", 1)))
        assert record.payload == (("a", 1), ("b", 2))

    def test_mapping_input_accepted(self) -> None:
        record = PointRecord.model_validate(
            {"point_id": 1, "vector": (1.0,), "payload": {"z": "x", "a": None}}
        )
        assert record.payload == (("a", None), ("z", "x"))

    def test_duplicate_keys_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc:
            PointRecord(point_id=1, vector=(1.0,), payload=(("a", 1), ("a", 2)))
        assert error_types(exc.value) == {"duplicate_payload_key"}

    def test_empty_key_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PointRecord(point_id=1, vector=(1.0,), payload=(("", 1),))

    def test_value_types_preserved(self) -> None:
        record = PointRecord(
            point_id=1, vector=(1.0,), payload=(("b", True), ("i", 1), ("f", 1.5), ("s", "x"))
        )
        kinds = {k: type(v) for k, v in record.payload}
        assert kinds == {"b": bool, "f": float, "i": int, "s": str}

    def test_negative_zero_payload_normalised(self) -> None:
        ((_, value),) = canonicalize_payload((("x", -0.0),))
        assert isinstance(value, float)
        assert math.copysign(1.0, value) == 1.0

    def test_json_payload_is_an_object(self) -> None:
        record = PointRecord(point_id=1, vector=(1.0,), payload=(("b", 1), ("a", True)))
        expected = '{"point_id":1,"vector":[1.0],"payload":{"a":true,"b":1}}'
        assert record.model_dump_json() == expected
        assert record.model_dump()["payload"] == (("a", True), ("b", 1))

    def test_nested_values_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PointRecord.model_validate({"point_id": 1, "vector": (1.0,), "payload": {"x": [1]}})

    @given(payload_dicts, st.randoms(use_true_random=False))
    def test_order_independent(self, payload: dict[str, PayloadValue], rnd: Any) -> None:
        items = list(payload.items())
        shuffled = items[:]
        rnd.shuffle(shuffled)
        a = PointRecord(point_id=7, vector=(1.0, 2.0), payload=tuple(items))
        b = PointRecord(point_id=7, vector=(1.0, 2.0), payload=tuple(shuffled))
        assert a == b
        assert a.payload == tuple(sorted(items, key=lambda kv: kv[0]))
        assert a.canonical_identity == b.canonical_identity


class TestCanonicalIdentity:
    def test_known_digest_is_stable(self) -> None:
        record = PointRecord(point_id=1, vector=(1.0, 0.5), payload=(("k", "v"),))
        assert record.canonical_identity == (
            "d5a99ae4edfe61b2253b0958d2ebade18a673389eb43cd5bbc4e1775a06d1dcd"
        )

    def test_type_aware(self) -> None:
        ids = {
            PointRecord(point_id=1, vector=(1.0,), payload=(("k", v),)).canonical_identity
            for v in (True, 1, 1.0, "1")
        }
        assert len(ids) == 4

    def test_negative_zero_does_not_split_identity(self) -> None:
        a = PointRecord(point_id=1, vector=(0.0,), payload=(("k", 0.0),))
        b = PointRecord(point_id=1, vector=(-0.0,), payload=(("k", -0.0),))
        assert a.canonical_identity == b.canonical_identity

    @given(
        st.integers(min_value=0, max_value=2**64 - 1),
        st.lists(f32, min_size=1, max_size=8),
        payload_dicts,
    )
    def test_deterministic_and_json_roundtrip_stable(
        self, pid: int, vec: list[float], payload: dict[str, PayloadValue]
    ) -> None:
        record = PointRecord(point_id=pid, vector=tuple(vec), payload=tuple(payload.items()))
        clone = PointRecord.model_validate_json(record.model_dump_json())
        assert clone == record
        assert clone.canonical_identity == record.canonical_identity
        assert len(record.canonical_identity) == 64

    @given(st.integers(min_value=0, max_value=2**64 - 2), st.lists(f32, min_size=1, max_size=8))
    def test_sensitive_to_point_id(self, pid: int, vec: list[float]) -> None:
        a = PointRecord(point_id=pid, vector=tuple(vec))
        b = PointRecord(point_id=pid + 1, vector=tuple(vec))
        assert a.canonical_identity != b.canonical_identity

    @given(st.lists(f32, min_size=1, max_size=8), st.lists(f32, min_size=1, max_size=8))
    def test_identity_equal_iff_vectors_equal(self, v1: list[float], v2: list[float]) -> None:
        a = PointRecord(point_id=1, vector=tuple(v1))
        b = PointRecord(point_id=1, vector=tuple(v2))
        assert (a.canonical_identity == b.canonical_identity) == (a.vector == b.vector)

    @pytest.mark.parametrize("pid", [-1, 2**64])
    def test_point_id_is_u64(self, pid: int) -> None:
        with pytest.raises(ValidationError):
            PointRecord(point_id=pid, vector=(1.0,))


# ---------------------------------------------------------------------------
# evolve()
# ---------------------------------------------------------------------------


class TestEvolve:
    def test_applies_changes_and_returns_new_instance(self) -> None:
        spec = VectorSpec(dimension=3, distance=DistanceMetric.DOT)
        changed = evolve(spec, dimension=8)
        assert changed == VectorSpec(dimension=8, distance=DistanceMetric.DOT)
        assert spec.dimension == 3

    def test_revalidates_field_constraints(self) -> None:
        spec = VectorSpec(dimension=3, distance=DistanceMetric.DOT)
        with pytest.raises(ValidationError):
            evolve(spec, dimension=0)

    def test_revalidates_model_invariants(self) -> None:
        report = VerificationReport(
            passed=True,
            source_count=1,
            target_count=1,
            source_checksum=SHA_A,
            target_checksum=SHA_A,
        )
        with pytest.raises(ValidationError) as exc:
            evolve(report, target_count=2)
        assert error_types(exc.value) == {"inconsistent_verdict"}

    def test_rejects_unknown_fields(self) -> None:
        spec = VectorSpec(dimension=3, distance=DistanceMetric.DOT)
        with pytest.raises(ValidationError) as exc:
            evolve(spec, dimensions=4)
        assert error_types(exc.value) == {"extra_forbidden"}

    def test_recanonicalises(self) -> None:
        record = PointRecord(point_id=1, vector=(1.0,))
        changed = evolve(record, vector=(0.1,), payload=(("b", 1), ("a", 2)))
        assert changed.vector == to_float32((0.1,))
        assert changed.payload == (("a", 2), ("b", 1))

    def test_model_copy_would_have_bypassed_validation(self) -> None:
        spec = VectorSpec(dimension=3, distance=DistanceMetric.DOT)
        assert spec.model_copy(update={"dimension": 0}).dimension == 0
        with pytest.raises(ValidationError):
            evolve(spec, dimension=0)

    @given(st.integers())
    def test_matches_direct_construction(self, dim: int) -> None:
        spec = VectorSpec(dimension=1, distance=DistanceMetric.COSINE)
        try:
            direct = VectorSpec(dimension=dim, distance=DistanceMetric.COSINE)
        except ValidationError:
            with pytest.raises(ValidationError):
                evolve(spec, dimension=dim)
        else:
            assert evolve(spec, dimension=dim) == direct


# ---------------------------------------------------------------------------
# Scroll / collection / alias contracts
# ---------------------------------------------------------------------------


class TestScrollPage:
    def test_empty_defaults(self) -> None:
        page = ScrollPage()
        assert page.points == ()
        assert page.next_offset is None

    def test_valid_page(self) -> None:
        pts = tuple(PointRecord(point_id=i, vector=(1.0,)) for i in (1, 3))
        assert ScrollPage(points=pts, next_offset=4).next_offset == 4

    def test_offset_without_points_allowed(self) -> None:
        assert ScrollPage(next_offset=10).next_offset == 10

    @pytest.mark.parametrize("ids", [(2, 1), (1, 1)])
    def test_rejects_unordered_ids(self, ids: tuple[int, int]) -> None:
        pts = tuple(PointRecord(point_id=i, vector=(1.0,)) for i in ids)
        with pytest.raises(ValidationError) as exc:
            ScrollPage(points=pts)
        assert error_types(exc.value) == {"scroll_order"}

    def test_rejects_stale_offset(self) -> None:
        pts = (PointRecord(point_id=5, vector=(1.0,)),)
        with pytest.raises(ValidationError) as exc:
            ScrollPage(points=pts, next_offset=5)
        assert error_types(exc.value) == {"scroll_offset"}


class TestCollectionAndAlias:
    def test_collection_info(self) -> None:
        info = CollectionInfo(
            name="c1", spec=VectorSpec(dimension=2, distance=DistanceMetric.EUCLID), points_count=0
        )
        assert info.points_count == 0

    @pytest.mark.parametrize("name", ["", "-lead", "has space", "x" * 129, "../etc"])
    def test_identifier_rejects_bad_names(self, name: str) -> None:
        with pytest.raises(ValidationError):
            CollectionInfo(
                name=name, spec=VectorSpec(dimension=2, distance=DistanceMetric.DOT), points_count=0
            )

    def test_alias_union_discriminates(self) -> None:
        adapter: TypeAdapter[AliasAction] = TypeAdapter(AliasAction)
        create = adapter.validate_json('{"kind": "create", "alias": "a", "collection": "c"}')
        delete = adapter.validate_json('{"kind": "delete", "alias": "a"}')
        assert create == CreateAliasAction(alias="a", collection="c")
        assert delete == DeleteAliasAction(alias="a")

    def test_alias_union_rejects_unknown_kind(self) -> None:
        adapter: TypeAdapter[AliasAction] = TypeAdapter(AliasAction)
        with pytest.raises(ValidationError):
            adapter.validate_json('{"kind": "rename", "alias": "a"}')

    def test_delete_rejects_collection_field(self) -> None:
        with pytest.raises(ValidationError):
            DeleteAliasAction(alias="a", collection="c")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Source / CDC contracts
# ---------------------------------------------------------------------------


class TestCdc:
    record = SourceRecord(record_id=9, tenant="acme", features=(0.5, 1.5), lsn=42)

    def test_features_keep_float64_precision(self) -> None:
        rec = SourceRecord(record_id=1, tenant="t", features=(0.1,), lsn=0)
        assert rec.features == (0.1,)

    def test_upsert_valid(self) -> None:
        event = CdcEvent(lsn=42, op=CdcOp.UPSERT, record_id=9, record=self.record)
        assert event.record is self.record

    def test_delete_valid(self) -> None:
        assert CdcEvent(lsn=43, op=CdcOp.DELETE, record_id=9).record is None

    def test_delete_with_record_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc:
            CdcEvent(lsn=42, op=CdcOp.DELETE, record_id=9, record=self.record)
        assert error_types(exc.value) == {"cdc_delete_record"}

    def test_upsert_without_record_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc:
            CdcEvent(lsn=42, op=CdcOp.UPSERT, record_id=9)
        assert error_types(exc.value) == {"cdc_upsert_record"}

    @pytest.mark.parametrize(("lsn", "rid"), [(41, 9), (42, 8)])
    def test_upsert_mismatch_rejected(self, lsn: int, rid: int) -> None:
        with pytest.raises(ValidationError) as exc:
            CdcEvent(lsn=lsn, op=CdcOp.UPSERT, record_id=rid, record=self.record)
        assert error_types(exc.value) == {"cdc_record_mismatch"}

    def test_negative_lsn_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CdcEvent(lsn=-1, op=CdcOp.DELETE, record_id=1)

    def test_json_roundtrip(self) -> None:
        event = CdcEvent(lsn=42, op=CdcOp.UPSERT, record_id=9, record=self.record)
        assert CdcEvent.model_validate_json(event.model_dump_json()) == event


# ---------------------------------------------------------------------------
# Migration contracts
# ---------------------------------------------------------------------------


class TestRetryPolicy:
    def test_defaults(self) -> None:
        policy = RetryPolicy()
        assert (policy.max_attempts, policy.base_delay_s, policy.max_delay_s) == (5, 0.1, 10.0)

    def test_max_below_base_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc:
            RetryPolicy(base_delay_s=2.0, max_delay_s=1.0)
        assert error_types(exc.value) == {"retry_delays"}

    @pytest.mark.parametrize(
        "kwargs", [{"max_attempts": 0}, {"base_delay_s": 0.0}, {"multiplier": 0.5}]
    )
    def test_bounds(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            RetryPolicy(**kwargs)


class TestPhaseTransition:
    def test_phase_set_is_complete(self) -> None:
        assert len(MigrationPhase) == 12
        assert {
            MigrationPhase.COMPLETED,
            MigrationPhase.ROLLED_BACK,
            MigrationPhase.ABORTED,
        } == TERMINAL_PHASES

    def test_self_transition_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc:
            PhaseTransition(
                sequence=0, from_phase=MigrationPhase.PENDING, to_phase=MigrationPhase.PENDING
            )
        assert error_types(exc.value) == {"self_transition"}

    @pytest.mark.parametrize("terminal", sorted(TERMINAL_PHASES))
    def test_cannot_leave_terminal(self, terminal: MigrationPhase) -> None:
        with pytest.raises(ValidationError) as exc:
            PhaseTransition(sequence=0, from_phase=terminal, to_phase=MigrationPhase.PENDING)
        assert error_types(exc.value) == {"terminal_transition"}

    def test_failed_is_not_terminal(self) -> None:
        step = PhaseTransition(
            sequence=0, from_phase=MigrationPhase.FAILED, to_phase=MigrationPhase.ROLLED_BACK
        )
        assert step.to_phase is MigrationPhase.ROLLED_BACK


class TestMigrationPlan:
    def test_defaults(self) -> None:
        plan = make_plan()
        assert plan.batch_size == 256
        assert plan.retry == RetryPolicy()

    @pytest.mark.parametrize(
        "overrides",
        [{"target_collection": "src"}, {"alias": "src"}, {"alias": "dst"}],
    )
    def test_names_must_be_distinct(self, overrides: dict[str, str]) -> None:
        with pytest.raises(ValidationError) as exc:
            make_plan(**overrides)
        assert error_types(exc.value) == {"plan_names"}

    @pytest.mark.parametrize("size", [0, 10_001])
    def test_batch_size_bounds(self, size: int) -> None:
        with pytest.raises(ValidationError):
            make_plan(batch_size=size)


class TestMigrationState:
    def test_initial_state(self) -> None:
        state = MigrationState(plan=make_plan())
        assert state.phase is MigrationPhase.PENDING
        assert state.history == ()

    def test_consistent_history(self) -> None:
        history = transitions(MigrationPhase.PROVISIONING, MigrationPhase.BACKFILLING)
        state = MigrationState(plan=make_plan(), phase=MigrationPhase.BACKFILLING, history=history)
        assert state.phase is MigrationPhase.BACKFILLING

    def test_phase_without_history_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc:
            MigrationState(plan=make_plan(), phase=MigrationPhase.BACKFILLING)
        assert error_types(exc.value) == {"history_phase"}

    def test_broken_chain_rejected(self) -> None:
        a, _ = transitions(MigrationPhase.PROVISIONING, MigrationPhase.BACKFILLING)
        gap = PhaseTransition(
            sequence=1, from_phase=MigrationPhase.RECONCILING, to_phase=MigrationPhase.DRAINING
        )
        with pytest.raises(ValidationError) as exc:
            MigrationState(plan=make_plan(), phase=MigrationPhase.DRAINING, history=(a, gap))
        assert error_types(exc.value) == {"history_chain"}

    def test_bad_sequence_rejected(self) -> None:
        step = PhaseTransition(
            sequence=5, from_phase=MigrationPhase.PENDING, to_phase=MigrationPhase.PROVISIONING
        )
        with pytest.raises(ValidationError) as exc:
            MigrationState(plan=make_plan(), phase=MigrationPhase.PROVISIONING, history=(step,))
        assert error_types(exc.value) == {"history_chain"}

    def test_failed_requires_error(self) -> None:
        history = transitions(MigrationPhase.FAILED)
        with pytest.raises(ValidationError) as exc:
            MigrationState(plan=make_plan(), phase=MigrationPhase.FAILED, history=history)
        assert error_types(exc.value) == {"missing_error"}
        state = MigrationState(
            plan=make_plan(), phase=MigrationPhase.FAILED, history=history, error="boom"
        )
        assert state.error == "boom"

    def test_error_forbidden_in_healthy_phase(self) -> None:
        with pytest.raises(ValidationError) as exc:
            MigrationState(plan=make_plan(), error="boom")
        assert error_types(exc.value) == {"unexpected_error"}

    def test_error_kept_after_rollback(self) -> None:
        history = transitions(MigrationPhase.FAILED, MigrationPhase.ROLLED_BACK)
        state = MigrationState(
            plan=make_plan(), phase=MigrationPhase.ROLLED_BACK, history=history, error="boom"
        )
        assert state.phase in TERMINAL_PHASES

    def test_evolve_advances_state(self) -> None:
        state = MigrationState(plan=make_plan())
        advanced = evolve(
            state,
            phase=MigrationPhase.PROVISIONING,
            history=transitions(MigrationPhase.PROVISIONING),
        )
        assert advanced.phase is MigrationPhase.PROVISIONING
        with pytest.raises(ValidationError):
            evolve(state, phase=MigrationPhase.PROVISIONING)

    def test_json_roundtrip(self) -> None:
        history = transitions(MigrationPhase.PROVISIONING)
        state = MigrationState(
            plan=make_plan(), phase=MigrationPhase.PROVISIONING, history=history, applied_lsn=7
        )
        assert MigrationState.model_validate_json(state.model_dump_json()) == state


class TestVerificationReport:
    @pytest.mark.parametrize(
        ("passed", "counts", "sums"),
        [
            (True, (3, 3), (SHA_A, SHA_A)),
            (False, (3, 4), (SHA_A, SHA_A)),
            (False, (3, 3), (SHA_A, SHA_B)),
        ],
    )
    def test_consistent_verdicts(
        self, passed: bool, counts: tuple[int, int], sums: tuple[str, str]
    ) -> None:
        report = VerificationReport(
            passed=passed,
            source_count=counts[0],
            target_count=counts[1],
            source_checksum=sums[0],
            target_checksum=sums[1],
        )
        assert report.passed is passed

    @pytest.mark.parametrize(
        ("passed", "counts", "sums"),
        [
            (False, (3, 3), (SHA_A, SHA_A)),
            (True, (3, 4), (SHA_A, SHA_A)),
            (True, (3, 3), (SHA_A, SHA_B)),
        ],
    )
    def test_inconsistent_verdicts_rejected(
        self, passed: bool, counts: tuple[int, int], sums: tuple[str, str]
    ) -> None:
        with pytest.raises(ValidationError) as exc:
            VerificationReport(
                passed=passed,
                source_count=counts[0],
                target_count=counts[1],
                source_checksum=sums[0],
                target_checksum=sums[1],
            )
        assert error_types(exc.value) == {"inconsistent_verdict"}

    @pytest.mark.parametrize("checksum", ["A" * 64, "a" * 63, "g" * 64])
    def test_checksum_format(self, checksum: str) -> None:
        with pytest.raises(ValidationError):
            VerificationReport(
                passed=False,
                source_count=0,
                target_count=0,
                source_checksum=checksum,
                target_checksum=SHA_A,
            )
