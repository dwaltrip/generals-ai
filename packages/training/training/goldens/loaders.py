from __future__ import annotations

from pathlib import Path

import numpy as np

from training.bc.datapipe.sim_types import CorpusGame, PerspectiveMeta, SimGame
from training.goldens.hashes import ObsDigest
from training.goldens.registry import FIXTURES_DIR, FixtureRecord


def load_fixture(fixture: FixtureRecord) -> tuple[SimGame, PerspectiveMeta]:
    corpus = CorpusGame.load(FIXTURES_DIR / f"{fixture.replay_id}.npz")
    return corpus.sim, corpus.perspective_for_slot(fixture.slot)


def load_array(path: Path) -> np.ndarray | None:
    return np.load(path) if path.exists() else None


def load_obs_digest(path: Path) -> ObsDigest | None:
    if not path.exists():
        return None
    with np.load(path) as z:
        return ObsDigest(frame_hashes=z["frame_hashes"], channel_hashes=z["channel_hashes"])


def load_supervision_reference(paths: dict[str, Path]) -> dict[str, np.ndarray] | None:
    out = {}
    for key, path in paths.items():
        arr = load_array(path)
        if arr is None:
            return None
        out[key] = arr
    return out
