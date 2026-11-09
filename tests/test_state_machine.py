"""Lifecycle DAG: every edge, every non-edge, terminal sinks, history rules."""

from __future__ import annotations

import itertools

import pytest
from hypothesis import given
from hypothesis import strategies as st

from embedding_feature_store.core.errors import IllegalTransitionError, PermanentError
from embedding_feature_store.core.state_machine import (
    LIFECYCLE,
    TRANSITIONS,
    advance,
    allowed_transitions,
    fail,
    is_terminal,
    transition,
)
from embedding_feature_store.schemas import (
    TERMINAL_PHASES,
    DistanceMetric,
    MigrationPhase,
    MigrationPlan,
    MigrationState,
    VectorSpec,
)

P = MigrationPhase

# Written out by hand, independently of the implementation's table construction.
EXPECTED: dict[MigrationPhase, set[MigrationPhase]] = {
    P.PENDING: {P.PROVISIONING, P.FAILED, P.ABORTED},
    P.PROVISIONING: {P.BACKFILLING, P.FAILED, P.ROLLED_BACK, P.ABORTED},
    P.BACKFILLING: {P.RECONCILING, P.FAILED, P.ROLLED_BACK, P.ABORTED},
    P.RECONCILING: {P.DRAINING, P.FAILED, P.ROLLED_BACK, P.ABORTED},
    P.DRAINING: {P.VERIFYING, P.FAILED, P.ROLLED_BACK, P.ABORTED},
    P.VERIFYING: {P.CUTTING_OVER, P.FAILED, P.ROLLED_BACK, P.ABORTED},
    P.CUTTING_OVER: {P.DECOMMISSIONING, P.FAILED, P.ROLLED_BACK},
    P.DECOMMISSIONING: {P.COMPLETED, P.FAILED},
    P.FAILED: {P.ROLLED_BACK, P.ABORTED},
    P.COMPLETED: set(),
    P.ROLLED_BACK: set(),
    P.ABORTED: set(),
}


def initial() -> MigrationState:
    plan = MigrationPlan(
        migration_id="m1",
        source_collection="src",
        target_collection="dst",
        alias="live",
        target_spec=VectorSpec(dimension=4, distance=DistanceMetric.COSINE),
    )
    return MigrationState(plan=plan)


def reach(phase: MigrationPhase) -> MigrationState:
    """A state in ``phase`` via the shortest legal path (FAILED from PENDING)."""
    state = initial()
    if phase is P.FAILED:
        return fail(state, "boom")
    if phase in {P.ROLLED_BACK, P.ABORTED}:
        return transition(fail(state, "boom"), phase)
    while state.phase is not phase:
        state = advance(state)
    return state


def test_lifecycle_is_the_exact_happy_path() -> None:
    assert LIFECYCLE == (
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


def test_transition_table_matches_specification() -> None:
    assert {k: set(v) for k, v in TRANSITIONS.items()} == EXPECTED


def test_table_is_read_only() -> None:
    with pytest.raises(TypeError):
        TRANSITIONS[P.PENDING] = frozenset()  # type: ignore[index]


def test_happy_path_end_to_end() -> None:
    state = initial()
    for _ in range(len(LIFECYCLE) - 1):
        state = advance(state, reason="ok")
    assert state.phase is P.COMPLETED
    assert [t.sequence for t in state.history] == list(range(8))
    assert [t.to_phase for t in state.history] == list(LIFECYCLE[1:])
    assert all(t.reason == "ok" for t in state.history)


@pytest.mark.parametrize(
    ("source", "target"), list(itertools.product(list(MigrationPhase), repeat=2))
)
def test_every_pair(source: MigrationPhase, target: MigrationPhase) -> None:
    state = reach(source)
    legal = target in EXPECTED[source]
    assert (target in allowed_transitions(state)) is legal
    if not legal:
        with pytest.raises(IllegalTransitionError) as exc:
            transition(state, target, error="x" if target is P.FAILED else None)
        assert (exc.value.source, exc.value.target) == (source, target)
        return
    error = "boom" if target is P.FAILED else None
    moved = transition(state, target, error=error)
    assert moved.phase is target
    assert moved.history[-1].from_phase is source
    assert len(moved.history) == len(state.history) + 1


@pytest.mark.parametrize("sink", sorted(TERMINAL_PHASES))
def test_terminal_sinks(sink: MigrationPhase) -> None:
    state = reach(sink)
    assert is_terminal(sink)
    assert allowed_transitions(state) == frozenset()
    with pytest.raises(IllegalTransitionError):
        advance(state)


def test_failed_is_not_terminal_and_has_no_happy_successor() -> None:
    state = reach(P.FAILED)
    assert not is_terminal(P.FAILED)
    with pytest.raises(IllegalTransitionError, match="no happy-path successor"):
        advance(state)


def test_illegal_transition_is_permanent() -> None:
    assert issubclass(IllegalTransitionError, PermanentError)


class TestErrors:
    def test_entering_failed_requires_error(self) -> None:
        with pytest.raises(IllegalTransitionError, match="requires an error"):
            transition(initial(), P.FAILED)

    def test_error_rejected_on_healthy_phase(self) -> None:
        with pytest.raises(IllegalTransitionError, match="only failure phases"):
            transition(initial(), P.PROVISIONING, error="boom")

    def test_fail_records_error(self) -> None:
        state = fail(advance(initial()), "disk full")
        assert state.phase is P.FAILED
        assert state.error == "disk full"
        assert state.history[-1].reason == "failure"

    def test_error_carried_into_rollback(self) -> None:
        state = transition(reach(P.FAILED), P.ROLLED_BACK)
        assert state.error == "boom"

    def test_error_can_be_replaced(self) -> None:
        state = transition(reach(P.FAILED), P.ABORTED, error="operator abort")
        assert state.error == "operator abort"

    def test_abort_without_failure_has_no_error(self) -> None:
        state = transition(initial(), P.ABORTED, reason="cancelled")
        assert state.error is None


class TestPointOfNoReturn:
    def test_rollback_forbidden_after_decommissioning_started(self) -> None:
        state = fail(reach(P.DECOMMISSIONING), "delete failed")
        assert allowed_transitions(state) == {P.ABORTED}
        with pytest.raises(IllegalTransitionError):
            transition(state, P.ROLLED_BACK)

    def test_rollback_allowed_when_failing_during_cutover(self) -> None:
        state = fail(reach(P.CUTTING_OVER), "alias swap failed")
        assert allowed_transitions(state) == {P.ROLLED_BACK, P.ABORTED}


def test_input_state_is_not_mutated() -> None:
    state = initial()
    before = state.model_dump()
    advance(state)
    fail(state, "x")
    assert state.model_dump() == before


@given(st.lists(st.integers(min_value=0), max_size=30))
def test_random_walks_stay_legal_and_deterministic(choices: list[int]) -> None:
    def walk() -> MigrationState:
        state = initial()
        for choice in choices:
            options = sorted(allowed_transitions(state))
            if not options:
                break
            target = options[choice % len(options)]
            state = transition(state, target, error="e" if target is P.FAILED else None)
        return state

    first, second = walk(), walk()
    assert first == second
    phases = [P.PENDING, *(t.to_phase for t in first.history)]
    for a, b in itertools.pairwise(phases):
        assert b in EXPECTED[a]
        assert not is_terminal(a)
    # Every path is bounded: at most the 8 happy steps plus FAILED and a sink.
    assert len(first.history) <= 10
