"""Smoke test for `bc.dataset.IterableDataset` against the real corpus."""

from __future__ import annotations

from itertools import islice
from pathlib import Path

import numpy as np
import pytest

from bc.constants import ELIGIBLE_PLAYER_COUNT, MAX_BOARD_SIDE
from bc.dataset import Frame, IterableDataset, is_eligible


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
INTERMEDIATE_DIR = _REPO_ROOT / "replay-parser" / "data" / "intermediate"


@pytest.fixture(scope="module")
def intermediate_root() -> Path:
    if not INTERMEDIATE_DIR.exists():
        pytest.skip(f"intermediate corpus not found at {INTERMEDIATE_DIR}")
    return INTERMEDIATE_DIR


def test_yields_frames_with_expected_keys(intermediate_root: Path) -> None:
    ds = IterableDataset(intermediate_root, seed=0)
    frames = list(islice(iter(ds), 50))
    assert len(frames) > 0

    f = frames[0]
    assert isinstance(f, Frame)
    for key in ("ownership", "armies", "actions_source", "map_width", "map_height"):
        assert key in f.sim
    for key in ("perspective_player_ids", "placement"):
        assert key in f.meta


def test_walk_order_is_k_outer_t_inner(intermediate_root: Path) -> None:
    """Within one game, k is non-decreasing; for each k, t goes 0, 1, 2, ..."""
    ds = IterableDataset(intermediate_root, seed=0)
    frames = list(islice(iter(ds), 500))

    by_game: dict[int, list[Frame]] = {}
    for f in frames:
        by_game.setdefault(id(f.sim), []).append(f)

    multi_frame_games = [g for g in by_game.values() if len(g) > 1]
    assert multi_frame_games, "expected at least one game with multiple frames"

    for game_frames in multi_frame_games:
        prev_k = -1
        prev_t = -1
        for f in game_frames:
            if f.k == prev_k:
                assert f.t == prev_t + 1, (
                    f"t should increment within fixed k; got t={f.t} after t={prev_t}"
                )
            else:
                assert f.k > prev_k, (
                    f"k should be non-decreasing; got k={f.k} after k={prev_k}"
                )
                assert f.t == 0, f"t should reset to 0 at new k; got t={f.t}"
            prev_k, prev_t = f.k, f.t


def test_same_game_frames_share_sim_meta_refs(intermediate_root: Path) -> None:
    """Per-game eager load — every frame from one game points at the same dicts."""
    ds = IterableDataset(intermediate_root, seed=0)
    frames = list(islice(iter(ds), 200))

    by_game: dict[int, list[Frame]] = {}
    for f in frames:
        by_game.setdefault(id(f.sim), []).append(f)

    for game_frames in by_game.values():
        sim0, meta0 = game_frames[0].sim, game_frames[0].meta
        for f in game_frames:
            assert f.sim is sim0
            assert f.meta is meta0


def test_is_eligible_matches_filter(intermediate_root: Path) -> None:
    """Spot-check `is_eligible` against the underlying conditions on real files."""
    sample_paths: list[Path] = []
    for prefix in sorted(intermediate_root.iterdir())[:3]:
        if not prefix.is_dir():
            continue
        for p in sorted(prefix.iterdir()):
            if p.name.endswith(".npz") and not p.name.endswith(".meta.npz"):
                sample_paths.append(p)
                if len(sample_paths) >= 20:
                    break
        if len(sample_paths) >= 20:
            break

    assert sample_paths, "expected at least one sim file under intermediate root"

    for sim_path in sample_paths:
        with np.load(sim_path) as sim:
            w = int(sim["map_width"])
            h = int(sim["map_height"])
            p = sim["actions_source"].shape[0]
        expected = max(w, h) <= MAX_BOARD_SIDE and p == ELIGIBLE_PLAYER_COUNT
        assert is_eligible(sim_path) == expected
