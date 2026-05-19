"""
IterableDataset over the per-game parsed corpus.

Walks per-game (`<id>.npz`) + per-game-meta (`<id>.meta.npz`) sibling pairs
under `replay-parser/data/intermediate/`, producing training samples for the
behavioral-cloning policy/value network.

Two iteration entry points:
  - `iter_frames()` yields raw `Frame(sim, meta, k, t)` — public seam for
    tests and ad-hoc inspection.
  - `__iter__` yields encoded `dict[str, Tensor]` — what `DataLoader` collates.
    Internally calls `iter_frames` then `encode_frame`.

Sampling: random shuffle over the eligible-file list per iteration,
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
import torch
from torch.utils.data import IterableDataset as TorchIterableDataset

from bc import actions
from bc.constants import (
    ELIGIBLE_PLAYER_COUNT,
    MAX_BOARD_SIDE,
    W_PADDED,
)
from bc.mask import build_mask
from bc.obs import build_obs
from bc.utils import list_sim_paths, meta_path_for


# Per-frame payload yielded by `iter_frames`. Downstream callers read `sim` +
# `meta` to build training tensors; `encode_frame` does this on the production
# `__iter__` path. `Frame` is intentionally a generic name — rename if a more
# specific name emerges (e.g., `RawFrame` paired with an encoded `TrainingExample`).
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


def encode_frame(
    sim: dict[str, np.ndarray],
    meta: dict[str, np.ndarray],
    k: int,
    t: int,
) -> dict[str, torch.Tensor]:
    """
    One (game, perspective, timestep) → one training sample dict.

    Orchestrates `build_obs`, `build_mask`, `actions.encode` (for the action
    target), and the value-target extraction. DataLoader's default collate
    stacks the result keywise into batched tensors.
    """
    H = int(sim["map_height"])
    W = int(sim["map_width"])
    perspective_slot = int(meta["perspective_player_ids"][k])

    obs_np = build_obs(sim, t, perspective_slot, H, W)
    mask_np = build_mask(sim, t, perspective_slot, H, W)

    src = int(sim["actions_source"][perspective_slot, t])
    dst = int(sim["actions_dest"][perspective_slot, t])
    is50 = int(sim["actions_is50"][perspective_slot, t])
    is_pass, flat_idx = actions.encode(src, dst, is50, W, W_PADDED)

    # Value target: placement is 1..P (1st through Pth); shift to a 0-indexed
    # class label for F.cross_entropy.
    value_target = int(meta["placement"][k]) - 1

    return {
        "obs": torch.from_numpy(obs_np),
        "mask": torch.from_numpy(mask_np),
        "action_target": torch.tensor(flat_idx, dtype=torch.int64),
        "is_pass": torch.tensor(is_pass, dtype=torch.bool),
        "value_target": torch.tensor(value_target, dtype=torch.int64),
    }


class IterableDataset(TorchIterableDataset):
    """
    Single-worker iterable over the per-game parsed corpus.

    Multi-worker semantics (per-worker split + per-worker seed offset)
    are not implemented — running with `num_workers > 1` will yield each
    sample `num_workers` times. Add worker-aware splitting when the
    training loop starts using DataLoader workers.
    """

    def __init__(
        self,
        intermediate_root: Path,
        seed: int,
        sim_paths: list[Path] | None = None,
    ) -> None:
        """
        `sim_paths`, if provided, skips the per-iteration `list_sim_paths` glob
        and uses the supplied list instead. Hooks the same seam that a
        pre-filtered training manifest will use (and that tests use to share
        a session-scoped cached list across the suite).
        """
        self._intermediate_root = intermediate_root
        self._seed = seed
        self._sim_paths = sim_paths
        self._epoch_counter = 0

    def iter_frames(self) -> Iterator[Frame]:
        """
        Walk the corpus and yield raw `Frame` objects, one per (perspective, timestep).

        Public seam for tests and ad-hoc inspection. Production iteration goes
        through `__iter__` which calls this and then encodes.

        Per-perspective frame range stops at `elim_timestep[k]` for eliminated
        perspectives — once a player is out, their subsequent "actions" are
        all-pass and carry no training signal (would just teach the model to
        pass when dead).
        """
        rng = random.Random(self._seed + self._epoch_counter)
        self._epoch_counter += 1
        # Defensive copy: rng.shuffle mutates in place; never mutate the caller's list.
        paths = (
            list(self._sim_paths)
            if self._sim_paths is not None
            else list_sim_paths(self._intermediate_root)
        )
        rng.shuffle(paths)

        for sim_path in paths:
            if not is_eligible(sim_path):
                continue
            meta_path = meta_path_for(sim_path)

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
                elim_t = int(meta["elim_timestep"][k])
                end_t = T - 1 if elim_t == -1 else min(T - 1, elim_t)
                for t in range(end_t):
                    yield Frame(sim=sim, meta=meta, k=k, t=t)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for f in self.iter_frames():
            yield encode_frame(f.sim, f.meta, f.k, f.t)
