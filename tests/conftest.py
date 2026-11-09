"""Shared test configuration.

Routes the Hypothesis example database under ./.cache/hypothesis and registers
deterministic profiles. Select a profile with HYPOTHESIS_PROFILE (default: dev).
"""

from __future__ import annotations

import os
from pathlib import Path

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
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))
