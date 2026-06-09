"""
Session-scoped fixtures shared across `packages/training/tests/`.

`list_sim_paths` walks ~165k file entries under the intermediate corpus —
~2.5s per call. Sharing the result across tests drops corpus-hitting tests
from ~2.5s each to subsecond.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from settings import INTERMEDIATE_DIR
from training.bc.filters import eligible_perspectives
from training.bc.splits import load_curated_names
from training.bc.utils import list_sim_paths, meta_path_for


# Cap how many sim files the `samples` fixture scans. Tests only ever pull
# a few hundred frames at most; full-corpus scans cost minutes for no benefit.
_SAMPLES_SCAN_LIMIT = 200


@pytest.fixture(scope="session")
def intermediate_root() -> Path:
    if not INTERMEDIATE_DIR.exists():
        pytest.skip(f"intermediate corpus not found at {INTERMEDIATE_DIR}")
    return INTERMEDIATE_DIR


@pytest.fixture(scope="session")
def sim_paths(intermediate_root: Path) -> list[Path]:
    """Cached sim-path list — computed once per session."""
    return list_sim_paths(intermediate_root)


@pytest.fixture(scope="session")
def samples(sim_paths: list[Path]) -> list[tuple[Path, int]]:
    """
    Cached `(sim_path, perspective_k)` list scanned from the first
    `_SAMPLES_SCAN_LIMIT` sim files. Mirrors what `bc.splits.build_manifest`
    produces, scoped down for test cost.
    """
    curated = load_curated_names()
    out: list[tuple[Path, int]] = []
    for sim_path in sim_paths[:_SAMPLES_SCAN_LIMIT]:
        for k in eligible_perspectives(sim_path, meta_path_for(sim_path), curated):
            out.append((sim_path, k))
    return out
