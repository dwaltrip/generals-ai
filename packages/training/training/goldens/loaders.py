from __future__ import annotations

from pathlib import Path

import numpy as np

from training.bc.datapipe.sim_types import GameMeta, PerspectiveMeta
from training.goldens.hashes import ObsDigest
from training.goldens.registry import FIXTURES_DIR, FixtureRecord


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def load_fixture(fixture: FixtureRecord) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    sim = _load_npz(FIXTURES_DIR / f"{fixture.replay_id}.npz")
    meta = _load_npz(FIXTURES_DIR / f"{fixture.replay_id}.meta.npz")
    return sim, meta


def perspective_for(game_meta: GameMeta, slot: int) -> PerspectiveMeta:
    matches = [p for p in game_meta.perspectives.values() if p.slot == slot]
    assert len(matches) == 1, f"slot {slot}: expected one recorded perspective, found {len(matches)}"
    return matches[0]


def load_array(path: Path) -> np.ndarray | None:
    return np.load(path) if path.exists() else None


def load_obs_digest(path: Path) -> ObsDigest | None:
    if not path.exists():
        return None
    z = _load_npz(path)
    return ObsDigest(frame_hashes=z["frame_hashes"], channel_hashes=z["channel_hashes"])


def load_supervision_reference(paths: dict[str, Path]) -> dict[str, np.ndarray] | None:
    out = {}
    for key, path in paths.items():
        arr = load_array(path)
        if arr is None:
            return None
        out[key] = arr
    return out
