"""
IterableDataset over the per-game parsed corpus.

Walks per-game (`<id>.npz`) + per-game-meta (`<id>.meta.npz`) sibling pairs
under `replay-parser/data/intermediate/`, producing training samples for the
behavioral-cloning policy/value network.

Two iteration entry points:
  - `iter_frames()` yields raw `Frame(sim, meta, k, t)` — public seam for
    tests and ad-hoc inspection. No fog/memory state attached.
  - `__iter__` yields encoded `dict[str, Tensor]` — what `DataLoader` collates.
    Manages per-(game, k) `MemoryState` + `BFSCache` internally, calls
    `step_memory` + `encode_frame` each tick.

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

from bc import actions, bfs
from bc.constants import (
    ELIGIBLE_PLAYER_COUNT,
    MAX_BOARD_SIDE,
    W_PADDED,
)
from bc.mask import build_mask
from bc.obs import (
    MemoryState,
    build_obs,
    canonical_slot_order,
    init_memory,
    step_memory,
)
from bc.utils import list_sim_paths, meta_path_for
from bc.visibility import compute_visibility


@dataclass
class Frame:
    """
    Per-frame raw payload yielded by `iter_frames` (the test/inspection seam).

    The production `__iter__` path builds encoded tensor dicts directly and
    does not pass through `Frame` — it would need fog/memory state alongside
    the raw arrays, and that's a tighter coupling than the raw seam should own.
    """
    sim: dict[str, np.ndarray]
    meta: dict[str, np.ndarray]
    k: int
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
    perspective_slot: int,
    opp_slots: list[int],
    vis: np.ndarray,
    state: MemoryState,
    bfs_cache: bfs.BFSCache,
    H: int,
    W: int,
) -> dict[str, torch.Tensor]:
    """
    One (game, perspective, timestep) → one training sample dict.

    Orchestrates `build_obs` (89 channels, requires state + vis + bfs cache),
    `build_mask` (legality, stateless), `actions.encode` (action target), and
    the value-target extraction. DataLoader's default collate stacks the
    result keywise into batched tensors.

    Pure-read of `state` + `bfs_cache`. `step_memory` must already have been
    called for this `(t, vis)` — `__iter__` enforces this ordering.
    """
    obs_np = build_obs(sim, t, perspective_slot, opp_slots, vis, state, bfs_cache, H, W)
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

    def _shuffled_paths(self) -> list[Path]:
        """Shared shuffle + epoch-counter increment. Used by iter_frames + __iter__."""
        rng = random.Random(self._seed + self._epoch_counter)
        self._epoch_counter += 1
        # Defensive copy: rng.shuffle mutates in place; never mutate the caller's list.
        paths = (
            list(self._sim_paths)
            if self._sim_paths is not None
            else list_sim_paths(self._intermediate_root)
        )
        rng.shuffle(paths)
        return paths

    def iter_frames(self) -> Iterator[Frame]:
        """
        Walk the corpus and yield raw `Frame` objects, one per (perspective, timestep).

        Public seam for tests and ad-hoc inspection. Production iteration goes
        through `__iter__` which does its own state-managed walk.

        Per-perspective frame range stops at `elim_timestep[k]` for eliminated
        perspectives — once a player is out, their subsequent "actions" are
        all-pass and carry no training signal (would just teach the model to
        pass when dead).
        """
        for sim_path in self._shuffled_paths():
            if not is_eligible(sim_path):
                continue
            meta_path = meta_path_for(sim_path)

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
        """
        Production walk: manages per-(game, k) MemoryState + BFSCache, calls
        step_memory each tick, yields encoded sample dicts ready for DataLoader.
        """
        for sim_path in self._shuffled_paths():
            if not is_eligible(sim_path):
                continue
            meta_path = meta_path_for(sim_path)

            with np.load(sim_path) as sim_npz:
                sim = {key: sim_npz[key] for key in sim_npz.files}
            with np.load(meta_path) as meta_npz:
                meta = {key: meta_npz[key] for key in meta_npz.files}

            T = sim["ownership"].shape[0]
            K = meta["perspective_player_ids"].shape[0]
            H = int(sim["map_height"])
            W = int(sim["map_width"])

            for k in range(K):
                perspective_slot = int(meta["perspective_player_ids"][k])
                opp_slots = canonical_slot_order(perspective_slot)[1:]

                state = init_memory(sim, perspective_slot, H, W)
                bfs_cache = bfs.init_bfs_cache()

                elim_t = int(meta["elim_timestep"][k])
                end_t = T - 1 if elim_t == -1 else min(T - 1, elim_t)

                for t in range(end_t):
                    vis = compute_visibility(sim["ownership"][t], perspective_slot, H, W)
                    _graph_grew = step_memory(state, sim, t, vis, perspective_slot, H, W)
                    # v1 BFS-policy: city passability depends on per-tick army
                    # values (perspective total + per-city), so the BFS graph
                    # cache is invalidated unconditionally each frame.
                    # `step_memory`'s `graph_grew` return is unused for now.
                    # every-frame invalidation already covers it.
                    # TODO: Either use `graph_grew` in the future or remove it.
                    # And then clean up or remove `bfs_cache` as well.
                    bfs_cache.invalidate_graph()

                    yield encode_frame(
                        sim, meta, k, t,
                        perspective_slot, opp_slots, vis,
                        state, bfs_cache, H, W,
                    )
