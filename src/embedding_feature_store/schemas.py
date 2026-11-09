"""Immutable, strictly validated data contracts.

Every model is frozen, forbids unknown fields, runs in Pydantic strict mode and
rejects NaN / +-Inf on every float. Vectors are canonicalised to IEEE-754
float32 so that what is hashed, compared and shipped is exactly what a float32
vector store will persist.

Instances are never mutated; derive new ones with :func:`evolve`, which runs
full validation on the result.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import struct
from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Any, Final, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    StringConstraints,
    model_validator,
)
from pydantic_core import PydanticCustomError

__all__ = [
    "TERMINAL_PHASES",
    "AliasAction",
    "CdcEvent",
    "CdcOp",
    "CollectionInfo",
    "CreateAliasAction",
    "DeleteAliasAction",
    "DistanceMetric",
    "FrozenModel",
    "MigrationPhase",
    "MigrationPlan",
    "MigrationState",
    "PayloadPairs",
    "PayloadValue",
    "PhaseTransition",
    "PointRecord",
    "RetryPolicy",
    "ScrollPage",
    "SourceRecord",
    "VectorSpec",
    "VerificationReport",
    "canonicalize_payload",
    "evolve",
    "to_float32",
]

_IDENTITY_SCHEMA: Final = "efs.point.v1"
_U64_MAX: Final = 2**64 - 1


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------


class FrozenModel(BaseModel):
    """Base for every contract: immutable, closed, strict, finite."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        validate_default=True,
    )


def evolve[M: BaseModel](model: M, /, **changes: Any) -> M:  # noqa: ANN401
    """Return a copy of ``model`` with ``changes`` applied, fully re-validated.

    Unlike ``model_copy(update=...)``, which skips validation, every field and
    model-level invariant is checked again; unknown field names are rejected.
    """
    cls = type(model)
    data = {name: getattr(model, name) for name in cls.model_fields}
    data.update(changes)
    return cls.model_validate(data)


# ---------------------------------------------------------------------------
# Scalar and vector types
# ---------------------------------------------------------------------------


def _float32_overflow(value: float) -> PydanticCustomError:
    return PydanticCustomError(
        "float32_overflow",
        "value {value} is outside the IEEE-754 float32 range",
        {"value": value},
    )


def to_float32(values: tuple[float, ...]) -> tuple[float, ...]:
    """Round ``values`` to the nearest IEEE-754 float32, rejecting non-finite input.

    Values whose float32 rounding would overflow to infinity are rejected rather
    than silently saturated. Negative zero is normalised to positive zero so that
    equal vectors always have equal canonical bytes.
    """
    for value in values:
        if not math.isfinite(value):
            raise PydanticCustomError(
                "finite_number", "vector components must be finite, got {value}", {"value": value}
            )
    try:
        packed = struct.pack(f"<{len(values)}f", *values)
    except OverflowError:
        worst = max(values, key=abs)
        raise _float32_overflow(worst) from None
    return tuple(v + 0.0 for v in struct.unpack(f"<{len(values)}f", packed))


type U64 = Annotated[int, Field(ge=0, le=_U64_MAX)]
type Lsn = Annotated[int, Field(ge=0)]
type NonNegativeInt = Annotated[int, Field(ge=0)]
type FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
type Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")]
type Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
type Float32Vector = Annotated[
    tuple[FiniteFloat, ...], Field(min_length=1), AfterValidator(to_float32)
]
type Features = Annotated[tuple[FiniteFloat, ...], Field(min_length=1)]

type PayloadKey = Annotated[str, StringConstraints(min_length=1, max_length=256)]
type PayloadValue = bool | int | FiniteFloat | str | None
"""A scalar payload value. ``bool`` is listed first so it is never coerced to ``int``."""


def _payload_from_mapping(value: object) -> object:
    if isinstance(value, Mapping):
        return tuple(value.items())
    return value


def canonicalize_payload(
    pairs: tuple[tuple[str, PayloadValue], ...],
) -> tuple[tuple[str, PayloadValue], ...]:
    """Sort payload pairs by key, reject duplicate keys, normalise ``-0.0`` to ``0.0``."""
    ordered = sorted(pairs, key=lambda kv: kv[0])
    for (prev, _), (key, _) in itertools.pairwise(ordered):
        if prev == key:
            raise PydanticCustomError(
                "duplicate_payload_key", "duplicate payload key {key!r}", {"key": key}
            )
    return tuple((key, val + 0.0 if isinstance(val, float) else val) for key, val in ordered)


type PayloadPairs = Annotated[
    tuple[tuple[PayloadKey, PayloadValue], ...],
    BeforeValidator(_payload_from_mapping),
    AfterValidator(canonicalize_payload),
    PlainSerializer(dict, return_type=dict[str, PayloadValue], when_used="json"),
]
"""Canonical payload: key-sorted, unique-key pairs. Accepts a mapping or pairs;
serialises to a JSON object (key order preserved)."""


# ---------------------------------------------------------------------------
# Vector store contracts
# ---------------------------------------------------------------------------


class DistanceMetric(StrEnum):
    COSINE = "cosine"
    DOT = "dot"
    EUCLID = "euclid"


class VectorSpec(FrozenModel):
    dimension: Annotated[int, Field(ge=1)]
    distance: DistanceMetric


class PointRecord(FrozenModel):
    """A point as persisted in the vector store."""

    point_id: U64
    vector: Float32Vector
    payload: PayloadPairs = ()

    @property
    def canonical_identity(self) -> str:
        """SHA-256 over a canonical, type-aware encoding of the point.

        The digest is independent of payload insertion order. It distinguishes
        ``True`` from ``1`` and ``1`` from ``1.0`` in payload values, which plain
        ``==`` does not.
        """
        doc = {
            "schema": _IDENTITY_SCHEMA,
            "id": self.point_id,
            "vector_f32le": struct.pack(f"<{len(self.vector)}f", *self.vector).hex(),
            "payload": [[key, val] for key, val in self.payload],
        }
        encoded = json.dumps(
            doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


class ScrollPage(FrozenModel):
    """One page of a scroll over a collection, ordered by ascending point id."""

    points: tuple[PointRecord, ...] = ()
    next_offset: U64 | None = None

    @model_validator(mode="after")
    def _check_order(self) -> Self:
        ids = [p.point_id for p in self.points]
        if any(a >= b for a, b in itertools.pairwise(ids)):
            raise PydanticCustomError(
                "scroll_order", "point ids must be strictly ascending within a page"
            )
        if self.next_offset is not None and ids and self.next_offset <= ids[-1]:
            raise PydanticCustomError(
                "scroll_offset", "next_offset must exceed the last point id on the page"
            )
        return self


class CollectionInfo(FrozenModel):
    name: Identifier
    spec: VectorSpec
    points_count: NonNegativeInt


class CreateAliasAction(FrozenModel):
    kind: Literal["create"] = "create"
    alias: Identifier
    collection: Identifier


class DeleteAliasAction(FrozenModel):
    kind: Literal["delete"] = "delete"
    alias: Identifier


type AliasAction = Annotated[CreateAliasAction | DeleteAliasAction, Field(discriminator="kind")]


# ---------------------------------------------------------------------------
# Source / CDC contracts
# ---------------------------------------------------------------------------


class SourceRecord(FrozenModel):
    record_id: U64
    tenant: Identifier
    features: Features
    lsn: Lsn


class CdcOp(StrEnum):
    UPSERT = "upsert"
    DELETE = "delete"


class CdcEvent(FrozenModel):
    """A change-data-capture event. UPSERT carries the record; DELETE carries none."""

    lsn: Lsn
    op: CdcOp
    record_id: U64
    record: SourceRecord | None = None

    @model_validator(mode="after")
    def _check_record(self) -> Self:
        if self.op is CdcOp.DELETE:
            if self.record is not None:
                raise PydanticCustomError(
                    "cdc_delete_record", "DELETE events must not carry a record"
                )
            return self
        if self.record is None:
            raise PydanticCustomError("cdc_upsert_record", "UPSERT events must carry a record")
        if self.record.record_id != self.record_id or self.record.lsn != self.lsn:
            raise PydanticCustomError(
                "cdc_record_mismatch", "record_id and lsn must match the carried record"
            )
        return self


# ---------------------------------------------------------------------------
# Migration contracts
# ---------------------------------------------------------------------------


class MigrationPhase(StrEnum):
    PENDING = "pending"
    PROVISIONING = "provisioning"
    BACKFILLING = "backfilling"
    RECONCILING = "reconciling"
    DRAINING = "draining"
    VERIFYING = "verifying"
    CUTTING_OVER = "cutting_over"
    DECOMMISSIONING = "decommissioning"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    ABORTED = "aborted"


TERMINAL_PHASES: Final[frozenset[MigrationPhase]] = frozenset(
    {MigrationPhase.COMPLETED, MigrationPhase.ROLLED_BACK, MigrationPhase.ABORTED}
)
_ERROR_PHASES: Final[frozenset[MigrationPhase]] = frozenset(
    {MigrationPhase.FAILED, MigrationPhase.ROLLED_BACK, MigrationPhase.ABORTED}
)


class RetryPolicy(FrozenModel):
    max_attempts: Annotated[int, Field(ge=1, le=100)] = 5
    base_delay_s: Annotated[FiniteFloat, Field(gt=0)] = 0.1
    max_delay_s: Annotated[FiniteFloat, Field(gt=0)] = 10.0
    multiplier: Annotated[FiniteFloat, Field(ge=1)] = 2.0

    @model_validator(mode="after")
    def _check_delays(self) -> Self:
        if self.max_delay_s < self.base_delay_s:
            raise PydanticCustomError("retry_delays", "max_delay_s must be >= base_delay_s")
        return self


class PhaseTransition(FrozenModel):
    sequence: NonNegativeInt
    from_phase: MigrationPhase
    to_phase: MigrationPhase
    reason: Annotated[str, StringConstraints(max_length=1024)] = ""

    @model_validator(mode="after")
    def _check_edge(self) -> Self:
        if self.from_phase is self.to_phase:
            raise PydanticCustomError("self_transition", "a transition must change the phase")
        if self.from_phase in TERMINAL_PHASES:
            raise PydanticCustomError(
                "terminal_transition",
                "cannot leave terminal phase {phase}",
                {"phase": self.from_phase.value},
            )
        return self


class MigrationPlan(FrozenModel):
    migration_id: Identifier
    source_collection: Identifier
    target_collection: Identifier
    alias: Identifier
    target_spec: VectorSpec
    batch_size: Annotated[int, Field(ge=1, le=10_000)] = 256
    retry: RetryPolicy = RetryPolicy()

    @model_validator(mode="after")
    def _check_names(self) -> Self:
        names = (self.source_collection, self.target_collection, self.alias)
        if len(set(names)) != len(names):
            raise PydanticCustomError(
                "plan_names", "source_collection, target_collection and alias must be distinct"
            )
        return self


class MigrationState(FrozenModel):
    """A migration's state, which must be reachable through its own history."""

    plan: MigrationPlan
    phase: MigrationPhase = MigrationPhase.PENDING
    history: tuple[PhaseTransition, ...] = ()
    applied_lsn: Lsn = 0
    backfill_cursor: U64 | None = None
    error: Annotated[str, StringConstraints(min_length=1)] | None = None

    @model_validator(mode="after")
    def _check_history(self) -> Self:
        current = MigrationPhase.PENDING
        for index, step in enumerate(self.history):
            if step.sequence != index or step.from_phase is not current:
                raise PydanticCustomError(
                    "history_chain",
                    "transition {index} does not continue the history",
                    {"index": index},
                )
            current = step.to_phase
        if current is not self.phase:
            raise PydanticCustomError(
                "history_phase", "phase does not match the last transition in history"
            )
        if self.phase is MigrationPhase.FAILED and self.error is None:
            raise PydanticCustomError("missing_error", "FAILED state requires an error")
        if self.error is not None and self.phase not in _ERROR_PHASES:
            raise PydanticCustomError(
                "unexpected_error", "error is only allowed in failed/rolled-back/aborted states"
            )
        return self


class VerificationReport(FrozenModel):
    """Source-vs-target comparison. ``passed`` must agree with the evidence."""

    passed: bool
    source_count: NonNegativeInt
    target_count: NonNegativeInt
    source_checksum: Sha256Hex
    target_checksum: Sha256Hex

    @model_validator(mode="after")
    def _check_verdict(self) -> Self:
        matches = (
            self.source_count == self.target_count and self.source_checksum == self.target_checksum
        )
        if self.passed is not matches:
            raise PydanticCustomError(
                "inconsistent_verdict", "passed must equal (counts match and checksums match)"
            )
        return self
