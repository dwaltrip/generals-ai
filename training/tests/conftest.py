"""
Session-scoped fixtures shared across `training/tests/`.

`list_sim_paths` walks ~165k file entries under the intermediate corpus —
~2.5s per call. Sharing the result across tests drops corpus-hitting tests
from ~2.5s each to subsecond.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bc.utils import list_sim_paths


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
INTERMEDIATE_DIR = _REPO_ROOT / "replay-parser" / "data" / "intermediate"


@pytest.fixture(scope="session")
def intermediate_root() -> Path:
    if not INTERMEDIATE_DIR.exists():
        pytest.skip(f"intermediate corpus not found at {INTERMEDIATE_DIR}")
    return INTERMEDIATE_DIR


@pytest.fixture(scope="session")
def sim_paths(intermediate_root: Path) -> list[Path]:
    """Cached sim-path list — computed once per session."""
    return list_sim_paths(intermediate_root)
