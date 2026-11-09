"""Two-tier verification of a target collection against what was written to it.

Tier 1 — cardinality: the store's exact count must equal the expected count.
Tier 2 — content: every point is scrolled, its dimension checked against the
target spec, and its canonical identity folded into an order-independent
:class:`MultisetDigest` that must equal the expected digest.

The expectation side is an :class:`ExpectationLedger` fed with every upsert and
delete the migration applies, so it tracks the final intended state (last
write per point id wins) without re-reading the source.
"""

from __future__ import annotations

import functools
import hashlib
from collections.abc import Awaitable, Callable, Iterable

from embedding_feature_store.core.errors import SchemaMismatchError, VerificationError
from embedding_feature_store.core.ports import VectorStore
from embedding_feature_store.core.retry import Retrier
from embedding_feature_store.schemas import PointRecord, VectorSpec, VerificationReport

__all__ = ["ExpectationLedger", "MigrationVerifier", "MultisetDigest"]

_MOD = 1 << 256
_ELEMENT_TAG = b"efs.multiset.v1\x00"
_FINAL_TAG = b"efs.multiset.final.v1\x00"


def _element_hash(item: str) -> int:
    return int.from_bytes(hashlib.sha256(_ELEMENT_TAG + item.encode()).digest(), "big")


class MultisetDigest:
    """Order-independent digest of a multiset of strings.

    Sums SHA-256 element hashes modulo 2**256, so insertion order does not
    matter and elements can be removed. Meant for integrity checking
    (accidental divergence), not as a collision-resistant commitment against an
    adversary choosing elements.
    """

    __slots__ = ("_acc", "_count")

    def __init__(self) -> None:
        self._acc = 0
        self._count = 0

    @classmethod
    def of(cls, items: Iterable[str]) -> MultisetDigest:
        digest = cls()
        for item in items:
            digest.add(item)
        return digest

    @property
    def count(self) -> int:
        return self._count

    def add(self, item: str) -> None:
        self._acc = (self._acc + _element_hash(item)) % _MOD
        self._count += 1

    def remove(self, item: str) -> None:
        """Remove one occurrence of ``item``; the caller must know it is present."""
        if self._count == 0:
            msg = "cannot remove from an empty multiset"
            raise ValueError(msg)
        self._acc = (self._acc - _element_hash(item)) % _MOD
        self._count -= 1

    def hexdigest(self) -> str:
        """SHA-256 over (count, accumulator); 64 lowercase hex characters."""
        body = self._count.to_bytes(8, "big") + self._acc.to_bytes(32, "big")
        return hashlib.sha256(_FINAL_TAG + body).hexdigest()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MultisetDigest):
            return NotImplemented
        return (self._count, self._acc) == (other._count, other._acc)

    def __hash__(self) -> int:
        return hash((self._count, self._acc))

    def __repr__(self) -> str:
        return f"MultisetDigest(count={self._count}, hex={self.hexdigest()[:12]}...)"


class ExpectationLedger:
    """The intended final contents of a target collection, as identities by point id."""

    def __init__(self) -> None:
        self._identities: dict[int, str] = {}
        self._digest = MultisetDigest()

    @property
    def count(self) -> int:
        return len(self._identities)

    @property
    def digest(self) -> MultisetDigest:
        return self._digest

    def record_upsert(self, point: PointRecord) -> None:
        identity = point.canonical_identity
        previous = self._identities.get(point.point_id)
        if previous is not None:
            self._digest.remove(previous)
        self._identities[point.point_id] = identity
        self._digest.add(identity)

    def record_delete(self, point_id: int) -> None:
        previous = self._identities.pop(point_id, None)
        if previous is not None:
            self._digest.remove(previous)


class MigrationVerifier:
    """Reads are idempotent; pass ``retrier`` to replay them on transient faults."""

    def __init__(
        self, store: VectorStore, *, page_size: int = 256, retrier: Retrier | None = None
    ) -> None:
        if page_size < 1:
            msg = "page_size must be >= 1"
            raise ValueError(msg)
        self._store = store
        self._page_size = page_size
        self._retrier = retrier

    async def _read[T](self, operation: Callable[[], Awaitable[T]]) -> T:
        if self._retrier is None:
            return await operation()
        return await self._retrier.run(operation)

    async def scan(self, collection: str, spec: VectorSpec) -> MultisetDigest:
        """Digest every point in ``collection``, enforcing ``spec.dimension``."""
        digest = MultisetDigest()
        offset: int | None = None
        while True:
            page = await self._read(
                functools.partial(
                    self._store.scroll, collection, offset=offset, limit=self._page_size
                )
            )
            for point in page.points:
                if len(point.vector) != spec.dimension:
                    msg = (
                        f"{collection}: point {point.point_id} has {len(point.vector)} "
                        f"dimensions, expected {spec.dimension}"
                    )
                    raise SchemaMismatchError(msg)
                digest.add(point.canonical_identity)
            if page.next_offset is None:
                return digest
            if offset is not None and page.next_offset <= offset:
                msg = f"{collection}: scroll did not advance past offset {offset}"
                raise VerificationError(msg)
            offset = page.next_offset

    async def verify(
        self, collection: str, spec: VectorSpec, ledger: ExpectationLedger
    ) -> VerificationReport:
        """Compare ``collection`` against ``ledger``. Never raises on a mismatch."""
        target_count = await self._read(functools.partial(self._store.count, collection))
        target = await self.scan(collection, spec)
        source_checksum = ledger.digest.hexdigest()
        target_checksum = target.hexdigest()
        return VerificationReport(
            passed=target_count == ledger.count and source_checksum == target_checksum,
            source_count=ledger.count,
            target_count=target_count,
            source_checksum=source_checksum,
            target_checksum=target_checksum,
        )

    async def require(
        self, collection: str, spec: VectorSpec, ledger: ExpectationLedger
    ) -> VerificationReport:
        """Like :meth:`verify`, but raise :class:`VerificationError` unless it passed."""
        report = await self.verify(collection, spec, ledger)
        if not report.passed:
            msg = (
                f"{collection}: expected {report.source_count} points "
                f"({report.source_checksum[:12]}), found {report.target_count} "
                f"({report.target_checksum[:12]})"
            )
            raise VerificationError(msg, report=report)
        return report
