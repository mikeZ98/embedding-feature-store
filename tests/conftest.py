"""Shared test configuration.

Routes the Hypothesis example database under ./.cache/hypothesis and registers
deterministic profiles. Select a profile with HYPOTHESIS_PROFILE (default: dev;
also ci and mutation).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from hypothesis import HealthCheck, settings
from hypothesis.configuration import set_hypothesis_home_dir
from hypothesis.database import DirectoryBasedExampleDatabase

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HYPOTHESIS_DIR = PROJECT_ROOT / ".cache" / "hypothesis"

set_hypothesis_home_dir(HYPOTHESIS_DIR)

settings.register_profile(
    "dev",
    database=DirectoryBasedExampleDatabase(HYPOTHESIS_DIR / "examples"),
    max_examples=100,
)
settings.register_profile(
    "ci",
    database=None,
    derandomize=True,
    max_examples=500,
    deadline=None,
    print_blob=True,
    suppress_health_check=[HealthCheck.too_slow],
)
# Mutation runs re-execute the suite thousands of times: every example must be
# reproducible so a mutant is never "killed" or "survives" by chance. mutmut
# also re-runs the same test in one process on a fresh class instance, which is
# exactly what differing_executors flags; a plain pytest run never does that.
settings.register_profile(
    "mutation",
    database=None,
    derandomize=True,
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.differing_executors],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))


def pytest_configure(config: pytest.Config) -> None:
    """Create the parent of ``--basetemp`` (pytest only creates the leaf).

    Without this, tmp_path fails on a fresh clone and in the mutation
    workspace, where .cache/pytest/ does not exist yet.
    """
    basetemp = config.option.basetemp
    if basetemp:
        Path(basetemp).parent.mkdir(parents=True, exist_ok=True)
