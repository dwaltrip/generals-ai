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
determines within-epoch group ordering.

Within a perspective, frames are generated in causal order (state at t
depends on having stepped 0..t-1). An optional reservoir-style shuffle
buffer decorrelates the yielded stream so DataLoader batches aren't
dominated by one perspective's consecutive frames.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset as TorchIterableDataset

from bc import actions, bfs
from bc.constants import H_PADDED, W_PADDED
from bc.mask import build_mask
from bc.obs import (
    MemoryState,
    build_obs,
    canonical_slot_order,
    init_memory,
    step_memory,
)
from bc.obs_config import ObsConfig
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


def _shuffle_buffered[T](
    upstream: Iterable[T],
    buffer_size: int,
    rng: random.Random,
) -> Iterator[T]:
    """
    Reservoir-style shuffle buffer over an iterable.

    Fill phase pulls items until the buffer holds `buffer_size`. Steady state:
    each step yields a random buffered item and replaces its slot with the
    next upstream item. Drain phase shuffles the remaining buffer and yields
    it. Every input item is yielded exactly once.

    Decorrelates a path-dependent producer (here: per-perspective frame walk)
    from the consumer's batch slicing, without requiring random access to
    the producer. Caller owns the RNG so determinism is seed-controlled.
    """
    # Call iter() so the two phases share a cursor — passing a list would
    # otherwise restart from the front after the fill loop breaks.
    it = iter(upstream)

    buffer: list[T] = []
    for item in it:
        buffer.append(item)
        if len(buffer) >= buffer_size:
            break

    for item in it:
        i = rng.randrange(len(buffer))
        out = buffer[i]
        buffer[i] = item
        yield out

    rng.shuffle(buffer)
    yield from buffer


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

    Orchestrates `build_obs` (96 channels, requires state + vis + bfs cache),
    `build_mask` (legality, stateless), `actions.encode` (action target), and
    the value-target extraction. DataLoader's default collate stacks the
    result keywise into batched tensors.

    Pure-read of `state` + `bfs_cache`. `step_memory` must already have been
    called for this `(t, vis)` — `__iter__` enforces this ordering.
    """
    obs_np = build_obs(sim, t, perspective_slot, opp_slots, vis, state, bfs_cache, H, W)
    mask_np = build_mask(sim, t, perspective_slot, H, W)

    # Per-sample [1, H_PADDED, W_PADDED] bool, True over the unpadded board
    # region. Consumed by PassHead (masked global pool) and ValueHead (zero
    # padded contributions before flatten) so per-game board-size variance
    # doesn't leak into the head outputs as a magnitude effect.
    valid_mask_np = np.zeros((1, H_PADDED, W_PADDED), dtype=np.bool_)
    valid_mask_np[0, :H, :W] = True

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
        "valid_mask": torch.from_numpy(valid_mask_np),
        "action_target": torch.tensor(flat_idx, dtype=torch.int64),
        "is_pass": torch.tensor(is_pass, dtype=torch.bool),
        "value_target": torch.tensor(value_target, dtype=torch.int64),
    }


def assert_safe_loader(loader: DataLoader) -> None:
    """Fail fast if a DataLoader over our `IterableDataset` is misconfigured.

    Currently guards one thing: `persistent_workers=True` is incompatible
    with `IterableDataset.set_epoch` — persistent workers fork the dataset
    once and never see subsequent epoch mutations, so they keep shuffling
    with the construction-time seed silently. See the WARNING in
    `IterableDataset.set_epoch` for the full picture + fix shape.

    Call this once, right after constructing the DataLoader, so the
    failure surfaces at startup instead of in the middle of an epoch.
    """
    if loader.persistent_workers:
        raise RuntimeError(
            "DataLoader was constructed with persistent_workers=True, "
            "which is NOT SAFE with our IterableDataset.set_epoch — "
            "persistent workers fork the dataset once and never see "
            "subsequent epoch mutations. See IterableDataset.set_epoch "
            "for the full picture and the fix shape."
        )


class IterableDataset(TorchIterableDataset):
    """
    Iterable over a manifest of `(sim_path, perspective_k)` pairs.

    `shuffle_buffer_size` enables a reservoir shuffle over yielded samples;
    values ≤ 1 disable it (raw per-perspective walk for diagnostics).

    Multi-worker safe: when consumed by `DataLoader(num_workers > 1)`,
    each worker takes a disjoint shard of groups via `get_worker_info()`,
    and worker_id is mixed into the per-epoch shuffle seed so workers
    shuffle independently. Caller advances epochs via `set_epoch(epoch)`
    before each iteration — modeled on `DistributedSampler.set_epoch`.
    Without it, every iteration reproduces the same shuffle (epoch 0).
    """

    def __init__(
        self,
        samples: list[tuple[Path, int]],
        seed: int,
        obs_cfg: ObsConfig,
        shuffle_buffer_size: int = 0,
    ) -> None:
        """
        `samples` is a list of `(sim_path, perspective_k)` pairs. Caller is
        responsible for filtering — this class trusts every pair is trainable.
        See `bc.splits.samples_for_split` for the production producer.

        `obs_cfg` is the obs-encoder config (sizes the obs tensor); it must
        match the model's `in_ch`. Required — pass `config.arch.obs` from the
        training config (or `OBS_CONFIG_DEFAULTS` for default-shape diagnostics).
        """
        self._groups = _group_by_path(samples)
        self._seed = seed
        self._obs_cfg = obs_cfg
        self._shuffle_buffer_size = shuffle_buffer_size
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch index used to seed the per-epoch shuffle. Call
        before each iteration of the dataset; otherwise every iteration
        reproduces the epoch-0 shuffle. Mirrors `DistributedSampler.set_epoch`.

        Necessary because `__iter__` runs inside the DataLoader worker
        subprocess when `num_workers > 0`; mutating an internal counter
        from inside `__iter__` wouldn't survive the worker fork.

        =====================================================================
        WARNING — NOT SAFE WITH `persistent_workers=True`.
        DO NOT ENABLE `persistent_workers` WITHOUT REWORKING EPOCH PROPAGATION.
        =====================================================================

        Persistent workers fork the dataset object once at DataLoader
        construction and reuse those copies across epochs. Subsequent
        `set_epoch` calls mutate `self._epoch` only in the MAIN process —
        the worker copies never see the update and keep shuffling with the
        construction-time seed for the entire run. The failure is silent:
        no error, val loss just stops behaving like the data is being
        re-shuffled each epoch.

        Fixing this requires an out-of-band epoch channel: e.g. a shared
        `multiprocessing.Value` the workers re-read at the top of every
        `__iter__`, or a `worker_init_fn` that re-seeds. `DistributedSampler`
        sidesteps this because it's a map-style sampler; `IterableDataset`
        has no built-in equivalent. See the tracker's `persistent_workers`
        backlog entry.
        """
        self._epoch = epoch

    def _iter_groups(
        self,
        rng: random.Random,
        worker_id: int = 0,
        num_workers: int = 1,
    ) -> list[tuple[Path, tuple[int, ...]]]:
        """Per-worker ordered list of groups to walk.

        Shards `self._groups` by `i % num_workers == worker_id` so each
        worker gets a disjoint subset, then shuffles with the caller-owned
        `rng`. Caller controls the seed (and thereby cross-worker /
        cross-epoch independence). The `worker_id * 1009` multiplier in
        the caller's seed formula is a prime — chosen so the per-worker
        offsets don't alias against the additive epoch index and produce
        identical shuffles modulo shard membership.
        """
        groups = [
            g for i, g in enumerate(self._groups) if i % num_workers == worker_id
        ]
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

        Multi-worker behavior is driven by `torch.utils.data.get_worker_info()`;
        single-process iteration falls back to `worker_id=0, num_workers=1`.
        The per-epoch RNG mixes `worker_id * 1009` into the seed so each
        worker shuffles its shard independently — preserves the original
        single-RNG-drives-both-shuffle-and-buffer behavior on `num_workers=0`.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = worker_info.id, worker_info.num_workers

        rng = random.Random(self._seed + self._epoch + worker_id * 1009)
        groups = self._iter_groups(rng, worker_id=worker_id, num_workers=num_workers)

        walk = self._walk(groups)
        if self._shuffle_buffer_size > 1:
            yield from _shuffle_buffered(walk, self._shuffle_buffer_size, rng)
        else:
            yield from walk

    def _walk(
        self,
        groups: list[tuple[Path, tuple[int, ...]]],
    ) -> Iterator[dict[str, torch.Tensor]]:
        for sim_path, ks in groups:
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

                state = init_memory(sim, perspective_slot, H, W, self._obs_cfg)
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
