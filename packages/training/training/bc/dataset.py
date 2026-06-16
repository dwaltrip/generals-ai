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

from training.bc import actions, bfs
from training.bc.constants import H_PADDED, W_PADDED
from training.bc.mask import build_mask
from training.bc.obs import (
    MemoryState,
    build_obs,
    canonical_slot_order,
    init_memory,
    step_memory,
)
from training.bc.obs_config import ObsConfig
from training.bc.visibility import compute_visibility
from training.shared.timing import timer


if TYPE_CHECKING:
    from training.shared.timing_run import FileSink


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


@dataclass(frozen=True)
class _ElimCtx:
    """Per-game precompute backing the elimination heads' targets.

    `edges` is the dataset-constant bin-edge array (time_bin head only);
    `death_by_slot` and `is_real` are per-game [8]-arrays indexed by *raw* slot
    id; `sentinel` is the winner's stand-in death tick. Built once per game in
    `_walk` (when an elim head is enabled) and threaded into every `encode_frame`
    call for that game. Both elim variants read off this one precompute.
    """

    edges: np.ndarray | None   # [n_bins - 1] strictly-increasing bin edges; None for next_death
    death_by_slot: np.ndarray  # [8] int64 — elim timestep; winner sentinel; -1 = phantom
    is_real: np.ndarray        # [8] bool — slot actually played this game
    sentinel: int              # winner's stand-in death tick (> any real death)


def _precompute_elim(
    sim: dict[str, np.ndarray], edges: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, int]:
    """Per-game `(death_by_slot, is_real, sentinel)` for the elim targets.

    `death_by_slot[s]` is slot `s`'s elimination timestep, the large finite
    winner `sentinel` (integer-typed, not `np.inf`) for the one real slot absent
    from `death_events`, or `-1` for a phantom slot that never played.
    `is_real[s]` marks the slots present at t=0. The winner-vs-phantom split is
    what `real_slots` resolves: the winner is real-and-absent-from-deaths, a
    phantom is not-real — and only the latter must be masked out.

    The sentinel only has to exceed every real death tick (real deaths are ≤
    T−1) so the winner never wins the soonest-death argmin unless it is the lone
    survivor. For the time_bin head it is additionally sized as `T + max_edge`
    so the winner's `Δ = sentinel − t` lands in the top "never" bin; next_death
    passes no edges and uses `T + 1`.

    The obs encoder always runs at P=8 and `opp_slots` always yields 7 ids from
    `range(8)`, but FFA games can start with <8 players (~7% of the corpus, see
    `6.13-6`). Without `is_real`, those phantom channels would inherit the winner
    sentinel and train as "alive, never dies" on every frame.
    """
    own0 = sim["ownership"][0]
    real = np.unique(own0)
    real = real[real >= 0].astype(np.intp)
    T = sim["ownership"].shape[0]
    sentinel = T + (int(edges[-1]) if edges is not None else 1)

    death_by_slot = np.full(8, -1, dtype=np.int64)
    death_by_slot[real] = sentinel  # winner default; dead slots overwritten below
    deaths = sim["death_events"]
    if deaths.size:
        death_by_slot[deaths[:, 1].astype(np.intp)] = deaths[:, 0]

    is_real = np.zeros(8, dtype=np.bool_)
    is_real[real] = True
    return death_by_slot, is_real, sentinel


def _alive_mask(elim: _ElimCtx, raw_order: list[int], t: int) -> np.ndarray:
    """Per-channel alive mask `[8]`: a channel is alive iff its slot is real and
    not yet eliminated (`death > t` — a player eliminated at `t` counts dead at
    frame `t`). The cross-player softmax / regression domain; both elim target
    builders and the alive-only path derive `alive` here.
    """
    raw = np.asarray(raw_order, dtype=np.intp)
    return elim.is_real[raw] & (elim.death_by_slot[raw] > t)


def _elim_targets(
    elim: _ElimCtx, raw_order: list[int], t: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel `(elim_bin_target[8], alive_mask[8])` at frame `t`.

    `raw_order` is the canonical channel→raw-slot map (`[perspective_slot,
    *opp_slots]`). A channel is alive iff its slot is real AND not yet
    eliminated (`death > t` matches the existing aliveness convention — a player
    eliminated at `t` counts dead at frame `t`). Live `Δ = death − t` digitizes
    into a bin; the winner's large sentinel Δ lands in the top bin (merged with
    "never"). Dead and phantom channels get bin 0, masked out by `alive`.
    """
    assert elim.edges is not None, "time_bin targets require bin edges"
    raw = np.asarray(raw_order, dtype=np.intp)
    death_ch = elim.death_by_slot[raw]
    alive = _alive_mask(elim, raw_order, t)
    delta = death_ch - t
    bins = np.where(alive, np.digitize(delta, elim.edges, right=False), 0)
    return bins.astype(np.int64), alive


def _next_death_target(
    elim: _ElimCtx, raw_order: list[int], t: int
) -> tuple[int, np.ndarray, int]:
    """Per-frame `(next_victim_channel, alive_mask[8], dt)` for who-dies-next.

    `raw_order` is the canonical channel→raw-slot map (`[perspective_slot,
    *opp_slots]`). A channel is alive iff its slot is real AND not yet eliminated
    (`death > t`) — the same convention as the time_bin head; this is the domain
    of the cross-player softmax. The next victim is the alive channel with the
    soonest death tick; the winner's `sentinel` death is larger than any real
    death, so it only wins this argmin when it is the lone survivor.

    Returns the next-victim channel index, the alive mask, and `dt = death − t`
    (ticks until that death — the horizon, dumped for offline confidence-ramp /
    horizon-stratified reads). Frames with no real future death — the winner's
    tail, where only the winner remains alive — get `target = -1` (a CE
    `ignore_index` sentinel, masked from the loss) and `dt = -1`. Ties (two
    deaths at the same tick) resolve to the lowest canonical channel index via
    `argmin`'s first-min rule — a fixed, deterministic convention.
    """
    raw = np.asarray(raw_order, dtype=np.intp)
    death_ch = elim.death_by_slot[raw]                       # [8]
    alive = _alive_mask(elim, raw_order, t)                  # [8]
    # Soonest death among alive channels; non-alive channels can't be the victim.
    cand = np.where(alive, death_ch, np.iinfo(np.int64).max)
    nxt = int(cand.argmin())
    if cand[nxt] >= elim.sentinel:
        # The soonest "death" is the winner sentinel (or no alive channel) → no
        # real next elimination from here. Mask the frame out.
        return -1, alive, -1
    return nxt, alive, int(cand[nxt] - t)


@dataclass(frozen=True)
class FrameView:
    """Raw per-frame context, attached to each yielded sample when the dataset is
    iterated with `frame_view=True` (an analysis-only seam, default off).

    Holds the per-game `sim`/`meta` dicts (shared by reference across all of that
    game's frames — treat read-only) plus this frame's indexing, so offline
    analysis can read raw sim ground truth alongside the encoded obs in the same
    pass. Non-collatable (carries numpy dicts), so it is for direct iteration
    only — never route a `frame_view=True` dataset through a DataLoader.
    """

    sim: dict[str, np.ndarray]
    meta: dict[str, np.ndarray]
    k: int
    t: int
    perspective_slot: int
    opp_slots: list[int]


# TODO: Create a more robust return type for this.
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
    elim: _ElimCtx | None = None,
    elim_variant: str | None = None,
) -> dict[str, torch.Tensor]:
    """
    One (game, perspective, timestep) → one training sample dict.

    Orchestrates `build_obs` (96 channels, requires state + vis + bfs cache),
    `build_mask` (legality, stateless), `actions.encode` (action target), and
    the value-target extraction. DataLoader's default collate stacks the
    result keywise into batched tensors.

    `elim`, when set (the precompute is built), adds the per-frame elim targets;
    `None` leaves them off so non-elim runs carry no extra collate/transfer
    weight. `elim_variant` selects which targets:
      - `"time_bin"`: per-player `elim_bin_target`.
      - `"next_death"`: scalar `next_elim_target` (the next-victim channel, or -1
        when no future death) and `next_elim_dt` (ticks-to-next-death horizon).
      - `None` (with `elim` set): no variant targets — the alive-only path.

    Whenever `elim` is set, the frame carries the per-player [8] `alive_mask`
    (the softmax/eval domain), alongside any variant targets — and as the sole
    output of the `None`-variant path.

    Pure-read of `state` + `bfs_cache`. `step_memory` must already have been
    called for this `(t, vis)` — `__iter__` enforces this ordering.
    """
    obs_np = build_obs(sim, t, perspective_slot, opp_slots, vis, state, bfs_cache, H, W)
    mask_np = build_mask(sim, t, perspective_slot, H, W)

    # Mask + action target + value target + the numpy→tensor conversion. Timed
    # as one region: it's the per-sample tail after the obs build, dominated by
    # the `torch.from_numpy` / `torch.tensor` conversions feeding collation.
    with timer.section("encode_tail"):
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

        sample = {
            "obs": torch.from_numpy(obs_np),
            "mask": torch.from_numpy(mask_np),
            "valid_mask": torch.from_numpy(valid_mask_np),
            "action_target": torch.tensor(flat_idx, dtype=torch.int64),
            "is_pass": torch.tensor(is_pass, dtype=torch.bool),
            "value_target": torch.tensor(value_target, dtype=torch.int64),
        }
        if elim is not None:
            raw_order = [perspective_slot, *opp_slots]
            # Aliveness is variant-independent and is the one key every elim
            # consumer (and the alive-only probe path) needs — emit it once here,
            # then add whatever variant-specific targets were asked for.
            sample["alive_mask"] = torch.from_numpy(_alive_mask(elim, raw_order, t))
            if elim_variant == "next_death":
                nxt, _, dt = _next_death_target(elim, raw_order, t)
                sample["next_elim_target"] = torch.tensor(nxt, dtype=torch.int64)
                sample["next_elim_dt"] = torch.tensor(dt, dtype=torch.int64)
            elif elim_variant == "time_bin":
                bins, _ = _elim_targets(elim, raw_order, t)
                sample["elim_bin_target"] = torch.from_numpy(bins)
            # elim_variant is None → alive-only emission (emit_alive_mask path).
        return sample


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
        prof_sink: FileSink | None = None,
        include_frame_info: bool = False,
        elim_bin_edges: tuple[int, ...] | None = None,
        elim_head_variant: str | None = None,
        emit_alive_mask: bool = False,
        frame_view: bool = False,
    ) -> None:
        """
        `samples` is a list of `(sim_path, perspective_k)` pairs. Caller is
        responsible for filtering — this class trusts every pair is trainable.
        See `bc.splits.samples_for_split` for the production producer.

        `obs_cfg` is the obs-encoder config (sizes the obs tensor); it must
        match the model's `in_ch`. Required — pass `config.arch.obs` from the
        training config (or `OBS_CONFIG_DEFAULTS` for default-shape diagnostics).

        `prof_sink`, when set, switches on the per-worker timing profiler: each
        worker enables `timer` for its walk and flushes a snapshot through the
        sink on teardown. `None` (the default) keeps every timing seam inert.
        Wired from `train.build_dataloader` via `timing_run.active_sink()`.

        `include_frame_info` adds per-frame provenance scalars to each sample
        dict (`frame_t`, `players_alive`, `p_start`, `sample_idx`) for offline
        stratified analysis. Off by default:
        the training loop doesn't consume them, and the extra keys would ride
        through collate + device transfer for nothing.

        `elim_head_variant`, when set, switches on an elimination head's
        per-frame targets and selects which: `"time_bin"` →
        `elim_bin_target` (requires `elim_bin_edges` — they size the bins and the
        winner sentinel); `"next_death"` → `next_elim_target`/`next_elim_dt` (no
        edges — the next-victim target is invariant to them, so the sentinel is
        just "beyond the last tick"). Both also emit the per-player `alive_mask`.
        Pass the model's `arch.elim_head_variant` (+
        `arch.elim_bin_edges` for time_bin) iff the elim head is enabled. A
        `None` variant leaves the targets off, so non-elim runs are unaffected.

        `emit_alive_mask` requests the per-player `alive_mask` *without* any elim
        head variant — for consumers (e.g. an army-regression probe) that need
        the alive field but no elimination targets. It builds the same per-game
        elim precompute the variants use and emits only `alive_mask`. Redundant
        when a variant is set (that already emits it).
        """
        if elim_head_variant == "time_bin" and elim_bin_edges is None:
            raise ValueError("time_bin elim head requires elim_bin_edges")
        self._groups = _group_by_path(samples)
        self._seed = seed
        self._obs_cfg = obs_cfg
        self._shuffle_buffer_size = shuffle_buffer_size
        self._epoch = 0
        self._prof_sink = prof_sink
        self._frame_info = include_frame_info
        # Analysis-only seam: attach the raw sim/meta + frame indexing to each
        # yielded sample (direct iteration only — non-collatable). Off by default
        # so the training path is untouched.
        self._frame_view = frame_view
        self._elim_edges = (
            np.asarray(elim_bin_edges, dtype=np.int64)
            if elim_bin_edges is not None
            else None
        )
        self._elim_variant = elim_head_variant
        # The per-game elim precompute backs both the variant targets and the
        # standalone alive mask, so either request triggers it.
        self._needs_elim_ctx = elim_head_variant is not None or emit_alive_mask
        # Index of each (path, k) pair in the caller's `samples` order, so
        # `sample_idx` survives the group/epoch shuffles and lets offline
        # consumers join frames back to the manifest entry they came from.
        self._sample_index = (
            {pair: i for i, pair in enumerate(samples)} if include_frame_info else {}
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

        # Profiling is per-process: enable + reset this worker's own `timer` so
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

    def _walk(
        self,
        groups: list[tuple[Path, tuple[int, ...]]],
    ) -> Iterator[dict[str, torch.Tensor]]:
        for sim_path, ks in groups:
            meta_path = sim_path.with_name(sim_path.stem + ".meta.npz")

            # Per-game volume read + npz DEFLATE decompression. `np.load` is
            # lazy; the dict comprehensions force the read+inflate, so the seam
            # has to wrap them, not just the `np.load` calls.
            with timer.section("data_load"):
                with np.load(sim_path) as sim_npz:
                    sim = {key: sim_npz[key] for key in sim_npz.files}
                with np.load(meta_path) as meta_npz:
                    meta = {key: meta_npz[key] for key in meta_npz.files}

            T = sim["ownership"].shape[0]
            H = int(sim["map_height"])
            W = int(sim["map_width"])

            if self._frame_info:
                # Aliveness comes from the sim, not meta: `meta.elim_timestep`
                # covers only the *recorded* perspectives, while `death_events`
                # is (timestep, slot) for every eliminated player (winner
                # absent). players_alive(t) = P_start − deaths with ts ≤ t,
                # i.e. a player eliminated at t counts as dead in frame t.
                deaths = (
                    np.sort(sim["death_events"][:, 0])
                    if sim["death_events"].size
                    else np.zeros(0, dtype=np.int64)
                )
                # Owner ids are ≥ 0; -1 is neutral, -2 mountains. At t=0 each
                # player owns exactly their general, so the distinct owner
                # count is the starting player count.
                p_start = int((np.unique(sim["ownership"][0]) >= 0).sum())

            # Per-game elim precompute (winner-vs-phantom resolution lives here,
            # once per game, not per perspective). `None` unless a variant or the
            # standalone alive mask was requested.
            elim_ctx = (
                _ElimCtx(self._elim_edges, *_precompute_elim(sim, self._elim_edges))
                if self._needs_elim_ctx
                else None
            )

            for k in ks:
                perspective_slot = int(meta["perspective_player_ids"][k])
                opp_slots = canonical_slot_order(perspective_slot)[1:]

                with timer.section("perspective_setup"):
                    state = init_memory(sim, perspective_slot, H, W, self._obs_cfg)
                    bfs_cache = bfs.init_bfs_cache()

                elim_t = int(meta["elim_timestep"][k])
                end_t = T - 1 if elim_t == -1 else min(T - 1, elim_t)

                for t in range(end_t):
                    vis = compute_visibility(sim["ownership"][t], perspective_slot, H, W)
                    step_memory(state, sim, t, vis, perspective_slot, H, W)

                    # Reference span over the build_obs/mask/tail child seams.
                    with timer.section("encode_frame", grouped=False):
                        sample = encode_frame(
                            sim, meta, k, t,
                            perspective_slot, opp_slots, vis,
                            state, bfs_cache, H, W,
                            elim=elim_ctx,
                            elim_variant=self._elim_variant,
                        )
                    if self._frame_info:
                        sample["frame_t"] = torch.tensor(t, dtype=torch.int64)
                        sample["players_alive"] = torch.tensor(
                            p_start - int(np.searchsorted(deaths, t, side="right")),
                            dtype=torch.int64,
                        )
                        sample["p_start"] = torch.tensor(p_start, dtype=torch.int64)
                        sample["sample_idx"] = torch.tensor(
                            self._sample_index[(sim_path, k)], dtype=torch.int64
                        )
                    if self._frame_view:
                        sample["frame_view"] = FrameView(
                            sim=sim, meta=meta, k=k, t=t,
                            perspective_slot=perspective_slot, opp_slots=opp_slots,
                        )
                    # Measures per-sample overhead plus (per batch boundary) collate, shm_copy,
                    # and the queue put/block. Then `handoff − collate − shm_copy` should give
                    # us the worker→main IPC/blocking blind spot.
                    with timer.section("handoff", grouped=False):
                        yield sample
