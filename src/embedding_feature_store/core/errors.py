"""Exception taxonomy.

Every error raised by the core derives from :class:`FeatureStoreError` and is
classified as exactly one of:

* :class:`TransientError` — the same operation may succeed if replayed
  (timeouts, HTTP 408/429/5xx, connection resets, a gate that is draining).
  Only these are retried by :class:`~embedding_feature_store.core.retry.Retrier`.
* :class:`PermanentError` — replaying cannot help (validation, 404, schema or
  dimension mismatches, illegal state transitions, failed verification).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from embedding_feature_store.schemas import MigrationPhase, VerificationReport

__all__ = [
    "CutoverError",
    "DrainTimeoutError",
    "FeatureStoreError",
    "GateClosedError",
    "GateStateError",
    "IllegalTransitionError",
    "InvalidVectorError",
    "NotFoundError",
    "PermanentError",
    "RateLimitedError",
    "RetryExhaustedError",
    "SchemaMismatchError",
    "ServerError",
    "StoreConnectionError",
    "StoreTimeoutError",
    "TransientError",
    "ValidationFailedError",
    "VerificationError",
    "error_for_status",
]


class FeatureStoreError(Exception):
    """Root of every error raised by embedding_feature_store."""


# ---------------------------------------------------------------------------
# Transient (retryable)
# ---------------------------------------------------------------------------


class TransientError(FeatureStoreError):
    """A failure that may succeed if the identical operation is replayed."""


class StoreTimeoutError(TransientError):
    """The remote did not answer in time (includes HTTP 408)."""


class StoreConnectionError(TransientError):
    """The connection was refused, reset or dropped."""


class RateLimitedError(TransientError):
    """HTTP 429. ``retry_after`` is the server-requested delay in seconds, if any."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ServerError(TransientError):
    """HTTP 5xx."""

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class GateClosedError(TransientError):
    """A writer gave up waiting for a closed write gate to reopen."""


class DrainTimeoutError(TransientError):
    """In-flight writes did not drain in time; the freeze window was abandoned."""

    def __init__(self, message: str, *, in_flight: int) -> None:
        super().__init__(message)
        self.in_flight = in_flight


# ---------------------------------------------------------------------------
# Permanent (never retried)
# ---------------------------------------------------------------------------


class PermanentError(FeatureStoreError):
    """A failure that replaying the same operation cannot fix."""


class ValidationFailedError(PermanentError):
    """The remote rejected the request as malformed (HTTP 400/422)."""


class NotFoundError(PermanentError):
    """The addressed resource does not exist (HTTP 404)."""


class SchemaMismatchError(PermanentError):
    """Shapes disagree: vector dimension, distance metric or batch arity."""


class InvalidVectorError(PermanentError):
    """A vector cannot be stored (non-finite, float32 overflow, zero under cosine)."""


class GateStateError(PermanentError):
    """The write gate was driven through an invalid lifecycle (e.g. double close)."""


class IllegalTransitionError(PermanentError):
    """A migration phase transition outside the lifecycle DAG was requested."""

    def __init__(self, source: MigrationPhase, target: MigrationPhase, detail: str = "") -> None:
        message = f"illegal transition {source.value} -> {target.value}"
        super().__init__(f"{message}: {detail}" if detail else message)
        self.source = source
        self.target = target


class VerificationError(PermanentError):
    """Target contents do not match expectations."""

    def __init__(self, message: str, *, report: VerificationReport | None = None) -> None:
        super().__init__(message)
        self.report = report


class CutoverError(PermanentError):
    """The alias swap failed; the alias binding is reported in the message."""


class RetryExhaustedError(PermanentError):
    """A transient failure persisted through every allowed attempt."""

    def __init__(self, attempts: int, last_error: TransientError) -> None:
        super().__init__(f"gave up after {attempts} attempts: {last_error!r}")
        self.attempts = attempts
        self.last_error = last_error


# ---------------------------------------------------------------------------
# HTTP classification
# ---------------------------------------------------------------------------

_HTTP_TOO_MANY_REQUESTS = 429
_HTTP_REQUEST_TIMEOUT = 408
_HTTP_NOT_FOUND = 404
_HTTP_SERVER_ERROR_MIN = 500
_HTTP_CLIENT_ERROR_MIN = 400


def error_for_status(
    status_code: int, message: str, *, retry_after: float | None = None
) -> FeatureStoreError:
    """Map an HTTP error status to the taxonomy. ``status_code`` must be >= 400."""
    if status_code < _HTTP_CLIENT_ERROR_MIN:
        msg = f"status {status_code} is not an error"
        raise ValueError(msg)
    if status_code >= _HTTP_SERVER_ERROR_MIN:
        return ServerError(message, status_code=status_code)
    if status_code == _HTTP_TOO_MANY_REQUESTS:
        return RateLimitedError(message, retry_after=retry_after)
    if status_code == _HTTP_REQUEST_TIMEOUT:
        return StoreTimeoutError(message)
    if status_code == _HTTP_NOT_FOUND:
        return NotFoundError(message)
    return ValidationFailedError(message)
