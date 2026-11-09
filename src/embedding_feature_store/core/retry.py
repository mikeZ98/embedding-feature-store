"""Exponential backoff with Full Jitter.

The delay before retry ``n`` (1-based) is drawn uniformly from
``[0, min(max_delay_s, base_delay_s * multiplier ** (n - 1))]`` — the "Full
Jitter" strategy, which spreads synchronised clients the furthest apart. A
server-supplied ``Retry-After`` raises the floor of that draw.

Only :class:`TransientError` is replayed; any other exception, including
:class:`PermanentError` and cancellation, propagates immediately. Operations
handed to :class:`Retrier` must therefore be safe to replay (idempotent).
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable

from embedding_feature_store.core.errors import (
    RateLimitedError,
    RetryExhaustedError,
    TransientError,
)
from embedding_feature_store.core.ports import Sleep
from embedding_feature_store.schemas import RetryPolicy

__all__ = ["OnRetry", "Retrier", "backoff_ceiling"]

type OnRetry = Callable[[int, float, TransientError], None]
"""Called before each sleep with (failed attempt number, delay, error)."""


def backoff_ceiling(policy: RetryPolicy, retry_number: int) -> float:
    """Upper bound of the jittered delay before retry ``retry_number`` (1-based)."""
    if retry_number < 1:
        msg = "retry_number is 1-based"
        raise ValueError(msg)
    try:
        raw = policy.base_delay_s * policy.multiplier ** (retry_number - 1)
    except OverflowError:
        return policy.max_delay_s
    return min(policy.max_delay_s, raw)


class Retrier:
    """Replays an idempotent async operation on :class:`TransientError`.

    Pass ``seed`` for a reproducible jitter sequence (tests, simulations). The
    default draws its seed from OS entropy so that independent processes do not
    retry in lockstep.
    """

    def __init__(
        self,
        policy: RetryPolicy,
        *,
        sleep: Sleep = asyncio.sleep,
        seed: int | None = None,
        on_retry: OnRetry | None = None,
    ) -> None:
        self._policy = policy
        self._sleep = sleep
        self._rng = random.Random(seed)  # noqa: S311 - jitter, not cryptography
        self._on_retry = on_retry

    @property
    def policy(self) -> RetryPolicy:
        return self._policy

    def delay_for(self, retry_number: int, error: TransientError | None = None) -> float:
        """Draw the jittered delay before retry ``retry_number`` (consumes randomness)."""
        delay = self._rng.uniform(0.0, backoff_ceiling(self._policy, retry_number))
        if isinstance(error, RateLimitedError) and error.retry_after is not None:
            return max(delay, error.retry_after)
        return delay

    async def run[T](self, operation: Callable[[], Awaitable[T]]) -> T:
        """Await ``operation()``, replaying it on transient failure."""
        attempt = 1
        while True:
            try:
                return await operation()
            except TransientError as exc:
                if attempt >= self._policy.max_attempts:
                    raise RetryExhaustedError(attempt, exc) from exc
                delay = self.delay_for(attempt, exc)
                if self._on_retry is not None:
                    self._on_retry(attempt, delay, exc)
                await self._sleep(delay)
                attempt += 1
