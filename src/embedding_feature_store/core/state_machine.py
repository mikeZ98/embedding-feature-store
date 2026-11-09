"""Deterministic migration lifecycle.

Happy path (a strict chain)::

    PENDING -> PROVISIONING -> BACKFILLING -> RECONCILING -> DRAINING
            -> VERIFYING -> CUTTING_OVER -> DECOMMISSIONING -> COMPLETED

Failure edges:

* ``FAILED`` is reachable from every active phase. It is not terminal: it must
  resolve to ``ROLLED_BACK`` or ``ABORTED``.
* ``ROLLED_BACK`` is reachable from PROVISIONING..CUTTING_OVER and from FAILED,
  but never once DECOMMISSIONING has begun (the old collection may be gone).
  That point-of-no-return rule also binds FAILED, via the state's history.
* ``ABORTED`` (stop without undoing) is reachable from PENDING..VERIFYING —
  before the alias is touched — and from FAILED.

COMPLETED, ROLLED_BACK and ABORTED are terminal sinks.

All functions are pure: they return a new, fully validated
:class:`MigrationState` and never mutate their input.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from embedding_feature_store.core.errors import IllegalTransitionError
from embedding_feature_store.schemas import (
    TERMINAL_PHASES,
    MigrationPhase,
    MigrationState,
    PhaseTransition,
    evolve,
)

__all__ = [
    "LIFECYCLE",
    "TRANSITIONS",
    "advance",
    "allowed_transitions",
    "fail",
    "is_terminal",
    "transition",
]

P = MigrationPhase

LIFECYCLE: Final[tuple[MigrationPhase, ...]] = (
    P.PENDING,
    P.PROVISIONING,
    P.BACKFILLING,
    P.RECONCILING,
    P.DRAINING,
    P.VERIFYING,
    P.CUTTING_OVER,
    P.DECOMMISSIONING,
    P.COMPLETED,
)
_ACTIVE: Final = LIFECYCLE[:-1]
_ROLLBACKABLE: Final = frozenset(LIFECYCLE[1:7])  # PROVISIONING..CUTTING_OVER
_ABORTABLE: Final = frozenset(LIFECYCLE[:6])  # PENDING..VERIFYING
_ERROR_CARRYING: Final = frozenset({P.FAILED, P.ROLLED_BACK, P.ABORTED})


def _build_table() -> Mapping[MigrationPhase, frozenset[MigrationPhase]]:
    table: dict[MigrationPhase, frozenset[MigrationPhase]] = {}
    for current, following in zip(_ACTIVE, LIFECYCLE[1:], strict=True):
        edges = {following, P.FAILED}
        if current in _ROLLBACKABLE:
            edges.add(P.ROLLED_BACK)
        if current in _ABORTABLE:
            edges.add(P.ABORTED)
        table[current] = frozenset(edges)
    table[P.FAILED] = frozenset({P.ROLLED_BACK, P.ABORTED})
    for sink in TERMINAL_PHASES:
        table[sink] = frozenset()
    return MappingProxyType(table)


TRANSITIONS: Final[Mapping[MigrationPhase, frozenset[MigrationPhase]]] = _build_table()
"""Static edge table. :func:`allowed_transitions` also applies history rules."""


def is_terminal(phase: MigrationPhase) -> bool:
    return phase in TERMINAL_PHASES


def _passed_point_of_no_return(state: MigrationState) -> bool:
    return any(step.to_phase is P.DECOMMISSIONING for step in state.history)


def allowed_transitions(state: MigrationState) -> frozenset[MigrationPhase]:
    """Phases reachable from ``state`` in one step."""
    edges = TRANSITIONS[state.phase]
    if state.phase is P.FAILED and _passed_point_of_no_return(state):
        return edges - {P.ROLLED_BACK}
    return edges


def transition(
    state: MigrationState,
    target: MigrationPhase,
    *,
    reason: str = "",
    error: str | None = None,
) -> MigrationState:
    """Move ``state`` to ``target``, appending to its history.

    ``error`` is required when entering FAILED and only permitted when entering
    FAILED, ROLLED_BACK or ABORTED. An existing error is carried forward into
    ROLLED_BACK / ABORTED unless replaced.
    """
    if target not in allowed_transitions(state):
        raise IllegalTransitionError(state.phase, target)
    if target is P.FAILED and error is None:
        raise IllegalTransitionError(state.phase, target, "entering FAILED requires an error")
    if error is not None and target not in _ERROR_CARRYING:
        raise IllegalTransitionError(state.phase, target, "only failure phases carry an error")

    step = PhaseTransition(
        sequence=len(state.history), from_phase=state.phase, to_phase=target, reason=reason
    )
    carried = state.error if target in _ERROR_CARRYING else None
    return evolve(
        state,
        phase=target,
        history=(*state.history, step),
        error=error if error is not None else carried,
    )


def advance(state: MigrationState, *, reason: str = "") -> MigrationState:
    """Take the single happy-path step from the current active phase."""
    if state.phase not in _ACTIVE:
        raise IllegalTransitionError(state.phase, state.phase, "no happy-path successor")
    following = LIFECYCLE[LIFECYCLE.index(state.phase) + 1]
    return transition(state, following, reason=reason)


def fail(state: MigrationState, error: str) -> MigrationState:
    """Enter FAILED with ``error``."""
    return transition(state, P.FAILED, reason="failure", error=error)
