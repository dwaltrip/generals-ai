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
from dataclasses import dataclass
from pathlib import Path
import random
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate
from torch.utils.data import IterableDataset as TorchIterableDataset

from training.bc import bfs
from training.bc.emit_spec import EmitSpec
from training.bc.encode_frame import encode_frame
from training.bc.obs import init_memory, step_memory
from training.bc.precompute import precompute_for
from training.bc.sample import FrameMeta
from training.bc.sim_types import GameMeta
from training.bc.visibility import compute_visibility
from training.shared.timing import timer


if TYPE_CHECKING:
    from training.shared.timing_run import FileSink


type SampleRef = tuple[Path, int]   # (sim_path, perspective_k)


@dataclass(frozen=True)
class PerspectivesByGame:
    """All selected perspectives from a single game"""
    # The sim npz path for the game
    sim_path: Path
    perspective_ks: tuple[int, ...]


def _group_by_path(samples: list[SampleRef]) -> list[PerspectivesByGame]:
    """Transform a flat `(path, k)` list into a PerspectivesByGame for each path.
    Preserves first-seen order for paths (dicts are insertion-ordered).
    Also preserves within-path order for each perspective `k`. This is essential
    for keeping dataset walks deterministic.
    """
    by_path: dict[Path, list[int]] = {}
    for path, k in samples:
        by_path.setdefault(path, []).append(k)
    return [PerspectivesByGame(p, tuple(ks)) for p, ks in by_path.items()]


def timed_collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """`default_collate` under a `collate` seam (runs in the worker, so its timer
    records it). When profiling, also pre-share the batch under `shm_copy` to
    surface the worker→main IPC copy the queue put would otherwise hide; the put
    reuses the shared storage, so total work is unchanged."""
    with timer.section("collate"):
        out = default_collate(batch)
    if timer.enabled:
        with timer.section("shm_copy"):
            for v in out.values():
                if isinstance(v, torch.Tensor):
                    v.share_memory_()
    return out


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
        spec: EmitSpec,
        shuffle_buffer_size: int = 0,
        prof_sink: FileSink | None = None,
    ) -> None:
        """
        `samples` is a list of `(sim_path, perspective_k)` pairs. Caller is
        responsible for filtering — this class trusts every pair is trainable.
        See `bc.splits.samples_for_split` for the production producer.

        `spec` is the emission contract: what each yielded sample carries.

        `prof_sink`, when set, switches on the per-worker timing profiler: each
        worker enables `timer` for its walk and flushes a snapshot through the
        sink on teardown. `None` (the default) keeps every timing seam inert.
        Wired from `train.build_dataloader` via `timing_run.active_sink()`.
        """
        self._groups = _group_by_path(samples)
        self._seed = seed
        self._spec = spec
        self._shuffle_buffer_size = shuffle_buffer_size
        self._epoch = 0
        self._prof_sink = prof_sink
        # Index of each (path, k) pair in the caller's `samples` order, so
        # `sample_idx` survives the group/epoch shuffles and lets offline
        # consumers join frames back to the manifest entry they came from.
        self._sample_index = (
            {pair: i for i, pair in enumerate(samples)} if spec.emit_frame_info else {}
        )

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
    ) -> list[PerspectivesByGame]:
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

        # Profiling is per-process: enable + reset this worker's own timer so
        # its flushed file holds just this epoch's walk. Inert when no sink.
        if self._prof_sink is not None:
            timer.enabled = True
            timer.reset()

        rng = random.Random(self._seed + self._epoch + worker_id * 1009)
        groups = self._iter_groups(rng, worker_id=worker_id, num_workers=num_workers)

        walk = self._walk(groups)
        try:
            if self._shuffle_buffer_size > 1:
                yield from _shuffle_buffered(walk, self._shuffle_buffer_size, rng)
            else:
                yield from walk
        finally:
            # Runs on normal exhaustion AND early break (generator close on
            # worker shutdown), so the snapshot always flushes.
            if self._prof_sink is not None:
                self._prof_sink.flush(self._epoch, worker_id, timer.snapshot())

    def _walk(self, groups: list[PerspectivesByGame]) -> Iterator[dict[str, torch.Tensor]]:
        for g in groups:
            meta_path = g.sim_path.with_name(g.sim_path.stem + ".meta.npz")

            # Timer: per-game volume read + npz DEFLATE decompression.
            # `np.load` is lazy and the dict comprehensions cause the actual read+inflate.
            # So the timer call needs to include the comprehensions (not just `load`).
            with timer.section("data_load"):
                with np.load(g.sim_path) as sim_npz:
                    sim = {key: sim_npz[key] for key in sim_npz.files}
                with np.load(meta_path) as meta_npz:
                    meta = {key: meta_npz[key] for key in meta_npz.files}

            game_meta = GameMeta.from_npz(sim, meta)

            pre = precompute_for(self._spec, sim)

            for k in g.perspective_ks:
                perspective = game_meta.perspectives[k]

                with timer.section("perspective_setup"):
                    state = init_memory(
                        sim,
                        perspective.slot,
                        game_meta.H,
                        game_meta.W,
                        self._spec.obs,
                    )
                    bfs_cache = bfs.init_bfs_cache()

                for t in range(perspective.end_t):
                    vis = compute_visibility(
                        sim["ownership"][t],
                        perspective.slot,
                        game_meta.H,
                        game_meta.W,
                    )
                    step_memory(state, sim, t, vis, perspective.slot, game_meta.H, game_meta.W)

                    frame_meta = None
                    if self._spec.emit_frame_info:
                        frame_meta = FrameMeta(
                            frame_t=torch.tensor(t, dtype=torch.int64),
                            players_alive=torch.tensor(
                                game_meta.count_players_alive_at(t),
                                dtype=torch.int64,
                            ),
                            p_start=torch.tensor(game_meta.p_start, dtype=torch.int64),
                            sample_idx=torch.tensor(
                                self._sample_index[(g.sim_path, k)], dtype=torch.int64
                            ),
                        )

                    # Timer: reference span over the build_obs/mask/tail child seams.
                    with timer.section("encode_frame", grouped=False):
                        sample = encode_frame(
                            sim,
                            t,
                            perspective,
                            frame_meta,
                            vis,
                            state,
                            bfs_cache,
                            self._spec,
                            pre,
                        )

                    # Timer: measures per-sample overhead plus (per batch boundary) collate,
                    # shm_copy, and the queue put/block. Then `handoff − collate − shm_copy`
                    # should give us the worker→main IPC/blocking blind spot.
                    with timer.section("handoff", grouped=False):
                        yield sample.to_dict()
