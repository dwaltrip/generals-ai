"""
IterableDataset over the per-game parsed corpus.

Walks per-game (`<id>.npz`) + per-game-meta (`<id>.meta.npz`) sibling pairs
under `replay-parser/data/intermediate/`, yielding one `Frame` per
(perspective `k`, timestep `t`) within each game. Frames are raw — the
dataset itself does no encoding; downstream code reads `sim` + `meta` to
build training tensors.

Sampling: random shuffle over the eligible-file list per `__iter__`,
seeded by the constructor's `seed` plus an internal epoch counter so
successive epochs see different orderings while remaining deterministic.
Eligibility filtering is lazy — applied as the iterator walks the
shuffled file list. A pre-filtered manifest is the natural extension
when self-play / larger-corpus training arrives.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
from torch.utils.data import IterableDataset as TorchIterableDataset

from bc.constants import ELIGIBLE_PLAYER_COUNT, MAX_BOARD_SIDE


# Per-frame payload yielded by the IterableDataset. The dataset emits these
# raw; downstream callers read `sim` + `meta` to build training tensors.
# `Frame` is intentionally a generic name — rename if a more specific name
# emerges (e.g., `RawFrame` paired with an encoded `TrainingExample`).
@dataclass
class Frame:
    # All named arrays from the per-game sim npz, eagerly loaded.
    sim: dict[str, np.ndarray]
    # All named arrays from the per-game meta sidecar npz, eagerly loaded.
    meta: dict[str, np.ndarray]
    # Perspective index into `meta["perspective_player_ids"]`.
    k: int
    # Timestep in `[0, T-1)`.
    t: int


def _meta_path_for(sim_path: Path) -> Path:
    return sim_path.with_name(sim_path.stem + ".meta.npz")


def is_eligible(sim_path: Path) -> bool:
    """
    True iff the per-game sim file passes the spike's drop filter.

    Two conditions: `max(map_width, map_height) <= MAX_BOARD_SIDE` and
    player count == ELIGIBLE_PLAYER_COUNT. Map dims are checked first —
    they're 0-d scalars (~8 bytes), while player count requires reading
    `actions_source.shape` which materializes a ~32 KB array. The ~7% of
    games dropped on map dims short-circuit before paying the larger read.
    """
    with np.load(sim_path) as sim:
        w = int(sim["map_width"])
        h = int(sim["map_height"])
        if max(w, h) > MAX_BOARD_SIDE:
            return False
        p = sim["actions_source"].shape[0]
    return p == ELIGIBLE_PLAYER_COUNT


def _list_sim_paths(intermediate_root: Path) -> list[Path]:
    """Sorted list of sim file paths under the intermediate root."""
    out: list[Path] = []
    for prefix_dir in sorted(intermediate_root.iterdir()):
        if not prefix_dir.is_dir():
            continue
        for path in sorted(prefix_dir.iterdir()):
            name = path.name
            if name.endswith(".npz") and not name.endswith(".meta.npz"):
                out.append(path)
    return out


class IterableDataset(TorchIterableDataset):
    """
    Single-worker iterable over the per-game parsed corpus.

    Multi-worker semantics (per-worker split + per-worker seed offset)
    are not implemented — running with `num_workers > 1` will yield each
    frame `num_workers` times. Add worker-aware splitting when the
    training loop starts using DataLoader workers.
    """

    def __init__(self, intermediate_root: Path, seed: int) -> None:
        self._intermediate_root = intermediate_root
        self._seed = seed
        self._epoch_counter = 0

    def __iter__(self) -> Iterator[Frame]:
        rng = random.Random(self._seed + self._epoch_counter)
        self._epoch_counter += 1
        paths = _list_sim_paths(self._intermediate_root)
        rng.shuffle(paths)

        for sim_path in paths:
            if not is_eligible(sim_path):
                continue
            meta_path = _meta_path_for(sim_path)

            # Load every named array from both npz files eagerly. The iterator itself
            # only reads shapes (T from sim["ownership"], K from meta["perspective_player_ids"]);
            # downstream channel-assembly + target-encoding code consumes the rest. Zip-open
            # + DEFLATE setup dominates per-game I/O cost, so loading all entries is ~free
            # vs. a selective subset. Revisit if profiling shows per-game load cost matters.
            with np.load(sim_path) as sim_npz:
                sim = {key: sim_npz[key] for key in sim_npz.files}
            with np.load(meta_path) as meta_npz:
                meta = {key: meta_npz[key] for key in meta_npz.files}

            T = sim["ownership"].shape[0]
            K = meta["perspective_player_ids"].shape[0]
            for k in range(K):
                for t in range(T - 1):
                    yield Frame(sim=sim, meta=meta, k=k, t=t)
