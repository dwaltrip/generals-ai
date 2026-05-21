"""
IterableDataset over a manifest of `(sim_path, perspective_k)` pairs.

Consumed pairs are produced by `bc.splits.build_manifest` + resolved via
`bc.splits.samples_for_split`. The dataset has no opinion about eligibility —
the manifest is the contract for what trains. Pass only eligible pairs.

`__iter__` yields encoded `dict[str, Tensor]` — what `DataLoader` collates.
Manages per-(game, k) `MemoryState` + `BFSCache` internally, calls
`step_memory` + `encode_frame` each tick.

Iteration order: samples are grouped by `sim_path` so each game's listed
perspectives are walked back-to-back (one file open per game). Groups are
shuffled per-epoch by `seed + epoch_counter`. The split-time shuffle in
`bc.splits` determines train/val membership; this per-epoch shuffle
determines within-epoch ordering.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset as TorchIterableDataset

from bc import actions, bfs
from bc.constants import W_PADDED
from bc.mask import build_mask
from bc.obs import (
    MemoryState,
    build_obs,
    canonical_slot_order,
    init_memory,
    step_memory,
)
from bc.visibility import compute_visibility


def _group_by_path(samples: list[tuple[Path, int]]) -> list[tuple[Path, tuple[int, ...]]]:
    """
    Collapse a flat `(path, k)` list into one `(path, ks)` entry per unique path.

    Preserves first-seen order for paths and within-path order for k's
    (Python dicts are insertion-ordered). The dataset trusts caller-supplied
    k ordering — no internal sort.
    """
    by_path: dict[Path, list[int]] = {}
    for path, k in samples:
        by_path.setdefault(path, []).append(k)
    return [(p, tuple(ks)) for p, ks in by_path.items()]


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
    Single-worker iterable over a manifest of `(sim_path, perspective_k)` pairs.

    Multi-worker semantics (per-worker split + per-worker seed offset)
    are not implemented — running with `num_workers > 1` will yield each
    sample `num_workers` times. Add worker-aware splitting when the
    training loop starts using DataLoader workers.
    """

    def __init__(
        self,
        samples: list[tuple[Path, int]],
        seed: int,
    ) -> None:
        """
        `samples` is a list of `(sim_path, perspective_k)` pairs. Caller is
        responsible for filtering — this class trusts every pair is trainable.
        See `bc.splits.samples_for_split` for the production producer.
        """
        self._groups = _group_by_path(samples)
        self._seed = seed
        self._epoch_counter = 0

    def _shuffled_groups(self) -> list[tuple[Path, tuple[int, ...]]]:
        """Per-epoch shuffle of `(sim_path, k-tuple)` groups."""
        rng = random.Random(self._seed + self._epoch_counter)
        self._epoch_counter += 1
        groups = list(self._groups)
        rng.shuffle(groups)
        return groups

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """
        Production walk: manages per-(game, k) MemoryState + BFSCache, calls
        step_memory each tick, yields encoded sample dicts ready for DataLoader.

        Per-perspective frame range stops at `elim_timestep[k]` for eliminated
        perspectives — once a player is out, their subsequent "actions" are
        all-pass and carry no training signal (would just teach the model to
        pass when dead).
        """
        for sim_path, ks in self._shuffled_groups():
            meta_path = sim_path.with_name(sim_path.stem + ".meta.npz")

            with np.load(sim_path) as sim_npz:
                sim = {key: sim_npz[key] for key in sim_npz.files}
            with np.load(meta_path) as meta_npz:
                meta = {key: meta_npz[key] for key in meta_npz.files}

            T = sim["ownership"].shape[0]
            H = int(sim["map_height"])
            W = int(sim["map_width"])

            for k in ks:
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
