"""Source rows -> vector-store points.

Under the cosine metric vectors are L2-normalised *before* float32 rounding, so
the stored vector is the closest float32 approximation of the unit vector and
cosine reduces to a dot product in the store. DOT and EUCLID vectors are stored
as-is (rounded to float32).

Every point carries a fixed payload — ``tenant``, ``source_lsn`` and
``model_id`` — so its canonical identity changes whenever the source row, its
version or the embedding model changes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import ValidationError

from embedding_feature_store.core.errors import InvalidVectorError, SchemaMismatchError
from embedding_feature_store.core.ports import EmbeddingAdapter
from embedding_feature_store.schemas import (
    DistanceMetric,
    PointRecord,
    SourceRecord,
    VectorSpec,
)

__all__ = ["PointProjector", "l2_normalise"]


def l2_normalise(vector: Sequence[float]) -> tuple[float, ...]:
    """Scale ``vector`` to unit Euclidean length. Rejects zero and non-finite norms."""
    norm = math.hypot(*vector)
    if norm == 0.0 or not math.isfinite(norm):
        msg = f"cannot normalise a vector with norm {norm}"
        raise InvalidVectorError(msg)
    return tuple(component / norm for component in vector)


class PointProjector:
    """Builds :class:`PointRecord` objects for one target :class:`VectorSpec`."""

    def __init__(self, spec: VectorSpec, embedder: EmbeddingAdapter) -> None:
        if embedder.dimension != spec.dimension:
            msg = (
                f"embedder {embedder.model_id!r} produces {embedder.dimension}-d vectors, "
                f"target expects {spec.dimension}"
            )
            raise SchemaMismatchError(msg)
        self._spec = spec
        self._embedder = embedder

    @property
    def spec(self) -> VectorSpec:
        return self._spec

    def build_point(self, record: SourceRecord, vector: Sequence[float]) -> PointRecord:
        """Project one record with an already computed embedding."""
        if len(vector) != self._spec.dimension:
            msg = (
                f"record {record.record_id}: vector has {len(vector)} dimensions, "
                f"expected {self._spec.dimension}"
            )
            raise SchemaMismatchError(msg)
        if not all(math.isfinite(c) for c in vector):
            msg = f"record {record.record_id}: vector has non-finite components"
            raise InvalidVectorError(msg)
        if self._spec.distance is DistanceMetric.COSINE:
            vector = l2_normalise(vector)
        payload = (
            ("model_id", self._embedder.model_id),
            ("source_lsn", record.lsn),
            ("tenant", record.tenant),
        )
        try:
            return PointRecord(point_id=record.record_id, vector=tuple(vector), payload=payload)
        except ValidationError as exc:
            msg = f"record {record.record_id}: vector cannot be stored: {exc}"
            raise InvalidVectorError(msg) from exc

    async def project(self, records: Sequence[SourceRecord]) -> tuple[PointRecord, ...]:
        """Embed and project a batch, preserving input order."""
        if not records:
            return ()
        vectors = await self._embedder.embed([r.features for r in records])
        if len(vectors) != len(records):
            msg = f"embedder returned {len(vectors)} vectors for {len(records)} records"
            raise SchemaMismatchError(msg)
        return tuple(
            self.build_point(record, vector)
            for record, vector in zip(records, vectors, strict=True)
        )
