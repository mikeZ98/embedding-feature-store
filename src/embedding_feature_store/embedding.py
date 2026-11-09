"""Deterministic, network-free embedder: Gaussian random projection.

:class:`RandomProjectionEmbedder` maps R^k -> R^d with a fixed matrix whose
entries are drawn i.i.d. from N(0, 1/d) by a seeded generator. By the
Johnson-Lindenstrauss lemma, pairwise Euclidean distances are preserved up to a
factor (1 +- eps) with high probability once d = O(log n / eps^2), which makes
it a faithful stand-in for a learned embedding model in tests and simulations.

The matrix is an immutable tuple-of-tuples. :meth:`embed` snapshots its input
synchronously — at call time, before any await — and runs the matrix product
in a worker thread via :func:`asyncio.to_thread`, so the event loop is never
blocked and later mutation of the caller's buffers cannot leak into a result.
"""

from __future__ import annotations

import asyncio
import math
import random
from collections.abc import Coroutine, Sequence
from typing import Any

from embedding_feature_store.core.errors import SchemaMismatchError

__all__ = ["RandomProjectionEmbedder"]

type Matrix = tuple[tuple[float, ...], ...]


class RandomProjectionEmbedder:
    """Implements :class:`~embedding_feature_store.core.ports.EmbeddingAdapter`."""

    def __init__(self, input_dim: int, output_dim: int, *, seed: int) -> None:
        if input_dim < 1 or output_dim < 1:
            msg = "input_dim and output_dim must be >= 1"
            raise ValueError(msg)
        rng = random.Random(seed)  # noqa: S311 - reproducible projection, not cryptography
        scale = 1.0 / math.sqrt(output_dim)
        self._matrix: Matrix = tuple(
            tuple(rng.gauss(0.0, scale) for _ in range(input_dim)) for _ in range(output_dim)
        )
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._model_id = f"rp-gauss:{input_dim}x{output_dim}:seed={seed}"

    @property
    def dimension(self) -> int:
        return self._output_dim

    @property
    def input_dimension(self) -> int:
        return self._input_dim

    @property
    def model_id(self) -> str:
        return self._model_id

    def project_batch(self, batch: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
        """Synchronous projection of an already validated batch."""
        return tuple(tuple(math.sumprod(row, vector) for row in self._matrix) for vector in batch)

    def embed(
        self, features: Sequence[Sequence[float]]
    ) -> Coroutine[Any, Any, tuple[tuple[float, ...], ...]]:
        """Validate and snapshot ``features`` now; project them off the event loop."""
        snapshot = tuple(tuple(float(x) for x in vector) for vector in features)
        for index, vector in enumerate(snapshot):
            if len(vector) != self._input_dim:
                msg = f"input {index} has {len(vector)} features, expected {self._input_dim}"
                raise SchemaMismatchError(msg)
        return asyncio.to_thread(self.project_batch, snapshot)
