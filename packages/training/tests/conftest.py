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
from training.bc.splits import build_manifest, load_curated_names
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


# Note: If we need additional kwargs for `_make_manifest_small` we can do:
#   class MakeManifestOpts(TypedDict):
#       val_frac: NotRequired[float]
#       another_kwarg: NotRequired[SomeType]
#   # ...
#   def _make_manifest_small(seed: int, **kwargs: Unpack[MakeManifestOpts]):
#       # ...
#       return build_manifest(intermediate_root, seed=seed, sim_paths_for_tests=sub, **kwargs)
@pytest.fixture(scope="session")
def make_manifest_small(intermediate_root: Path, sim_paths: list[Path]):
    """Factory fixture for small subset of intermediate data"""

    def _make_manifest_small(seed: int, val_frac: None | float = None):
        sub = sim_paths[:100]
        if val_frac is not None:
            return build_manifest(
                intermediate_root,
                seed=seed,
                sim_paths_for_tests=sub,
                val_frac=val_frac,
            )
        return build_manifest(
            intermediate_root,
            seed=seed,
            sim_paths_for_tests=sub,
        )

    return _make_manifest_small


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
