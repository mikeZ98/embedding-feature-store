"""HTTP status classification and error payloads."""

from __future__ import annotations

import pytest

from embedding_feature_store.core.errors import (
    ConflictError,
    DrainTimeoutError,
    FeatureStoreError,
    IllegalTransitionError,
    NotFoundError,
    PermanentError,
    RateLimitedError,
    ServerError,
    StoreTimeoutError,
    TransientError,
    ValidationFailedError,
    VerificationError,
    error_for_status,
)
from embedding_feature_store.schemas import MigrationPhase


@pytest.mark.parametrize(
    ("status", "kind", "transient"),
    [
        (400, ValidationFailedError, False),
        (401, ValidationFailedError, False),
        (404, NotFoundError, False),
        (408, StoreTimeoutError, True),
        (409, ConflictError, False),
        (425, TransientError, True),
        (422, ValidationFailedError, False),
        (429, RateLimitedError, True),
        (500, ServerError, True),
        (503, ServerError, True),
        (599, ServerError, True),
    ],
)
def test_error_for_status(status: int, kind: type[FeatureStoreError], transient: bool) -> None:
    error = error_for_status(status, "msg")
    assert type(error) is kind
    assert isinstance(error, TransientError) is transient
    assert isinstance(error, PermanentError) is not transient
    assert str(error) == "msg"


def test_status_payloads() -> None:
    limited = error_for_status(429, "slow", retry_after=2.0)
    assert isinstance(limited, RateLimitedError)
    assert limited.retry_after == 2.0
    server = error_for_status(502, "bad gateway")
    assert isinstance(server, ServerError)
    assert server.status_code == 502


@pytest.mark.parametrize("status", [200, 302, 399])
def test_non_error_status_rejected(status: int) -> None:
    with pytest.raises(ValueError, match="not an error"):
        error_for_status(status, "ok")


def test_illegal_transition_message() -> None:
    plain = IllegalTransitionError(MigrationPhase.PENDING, MigrationPhase.COMPLETED)
    detailed = IllegalTransitionError(MigrationPhase.PENDING, MigrationPhase.FAILED, "why")
    assert str(plain) == "illegal transition pending -> completed"
    assert str(detailed) == "illegal transition pending -> failed: why"


def test_payload_attributes() -> None:
    assert DrainTimeoutError("x", in_flight=3).in_flight == 3
    assert VerificationError("x").report is None
