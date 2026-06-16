"""Tests for `bc.dataset` (shuffle-buffer helper, IterableDataset worker
semantics, `_walk` elimination cutoff) and `bc.filters.is_eligible`."""

from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pytest

from training.bc.constants import ELIGIBLE_PLAYER_COUNT, MAX_BOARD_SIDE
from training.bc.dataset import (
    IterableDataset,
    _shuffle_buffered,
)
from training.bc.filters import is_eligible
from training.bc.obs_config import OBS_CONFIG_DEFAULTS
from training.bc.targets.elim_targets import (
    ElimCtx,
    next_death_target,
    precompute_elim,
    time_bin_targets,
)


class _IdentityDataset(IterableDataset):
    """IterableDataset variant that yields identifier tuples instead of
    encoded frames. Lets tests iterate the dataset directly without
    depending on .npz fixtures or the encode_frame path."""

    def _walk(self, groups):
        for sim_path, ks in groups:
            for k in ks:
                for t in range(3):
                    yield (str(sim_path), k, t)


def test_shuffle_buffered_yields_all_inputs_once_in_new_order() -> None:
    """
    Reservoir buffer must preserve the multiset of inputs (every item yielded
    exactly once) while producing an order that differs from the input.
    """
    inputs = list(range(500))
    out = list(_shuffle_buffered(inputs, buffer_size=64, rng=random.Random(0)))

    assert sorted(out) == inputs, "buffer dropped or duplicated items"
    assert out != inputs, "buffer left order unchanged"


def test_shuffle_buffered_handles_undersized_upstream() -> None:
    """
    When upstream has fewer items than `buffer_size`, every item still gets
    yielded — the drain phase shuffles a partially-filled buffer.
    """
    inputs = list(range(10))
    out = list(_shuffle_buffered(inputs, buffer_size=64, rng=random.Random(0)))

    assert sorted(out) == inputs


def test_shuffle_buffered_is_deterministic() -> None:
    """Same seed + same inputs → same yield order."""
    inputs = list(range(500))
    a = list(_shuffle_buffered(inputs, buffer_size=64, rng=random.Random(42)))
    b = list(_shuffle_buffered(inputs, buffer_size=64, rng=random.Random(42)))
    assert a == b


@pytest.mark.parametrize("num_workers", [1, 2, 4, 8])
def test_iter_groups_partitions_disjointly_and_completely(num_workers: int) -> None:
    """Across all `worker_id`s, `_iter_groups` shards are pairwise disjoint
    and together cover every group exactly once, with near-equal sizes.
    Direct test of the sharding math — DataLoader's role is just wiring
    `get_worker_info()` into our adapter, which is PyTorch's responsibility.

    Not parametrized over `num_workers > num_groups` — empty shards yield
    nothing by construction, so the case has no bug to catch.
    """
    samples = [(Path(f"/synth/g{i:03d}.npz"), 0) for i in range(40)]
    ds = IterableDataset(samples=samples, seed=0, obs_cfg=OBS_CONFIG_DEFAULTS)

    seen_paths: set[Path] = set()
    sizes: list[int] = []
    for wid in range(num_workers):
        shard = ds._iter_groups(
            rng=random.Random(0),
            worker_id=wid,
            num_workers=num_workers,
        )
        shard_paths = {p for p, _ks in shard}
        assert shard_paths.isdisjoint(seen_paths), (
            f"worker {wid} shard overlaps prior workers"
        )
        seen_paths.update(shard_paths)
        sizes.append(len(shard))

    expected_paths = {p for p, _ in samples}
    assert seen_paths == expected_paths, "shards don't cover every group"
    assert max(sizes) - min(sizes) <= 1, f"shards unbalanced: {sizes}"


def test_cross_epoch_shuffle_advances() -> None:
    """`set_epoch(n)` controls the per-epoch shuffle. Same multiset both
    epochs, different order. Iterates the dataset directly — no DataLoader,
    no worker subprocess. The worker-fork failure mode the old
    `_epoch_counter += 1` had is now structurally impossible: `__iter__`
    no longer mutates `self`, so there's nothing for a fork to lose."""
    samples = [(Path(f"/synth/g{i:03d}.npz"), 0) for i in range(40)]
    ds = _IdentityDataset(samples=samples, seed=0, obs_cfg=OBS_CONFIG_DEFAULTS)

    ds.set_epoch(0)
    epoch0 = list(ds)
    ds.set_epoch(1)
    epoch1 = list(ds)

    assert sorted(epoch0) == sorted(epoch1), "same multiset both epochs"
    assert epoch0 != epoch1, "different epochs must yield different orders"


def test_walk_stops_at_elim_timestep(
    samples: list[tuple[Path, int]],
) -> None:
    """For an eliminated perspective, `_walk` yields exactly `elim_timestep`
    frames — once a player is out, post-elim frames carry no training
    signal and must not be yielded. Pins the elimination cutoff directly
    against the raw `elim_timestep` value, not the production formula
    (which the previous version of this test mirrored verbatim)."""
    for sim_path, k in samples:
        meta_path = sim_path.with_name(sim_path.stem + ".meta.npz")
        with np.load(meta_path) as meta_npz:
            elim_t = int(meta_npz["elim_timestep"][k])
        if elim_t > 0:
            ds = IterableDataset(samples=[(sim_path, k)], seed=0, obs_cfg=OBS_CONFIG_DEFAULTS)
            walked = sum(1 for _ in ds._walk(ds._groups))
            assert walked == elim_t, (
                f"perspective k={k} in {sim_path.name}: "
                f"expected {elim_t} frames (elim_timestep), got {walked}"
            )
            return
    pytest.skip("no eliminated perspectives in fixture")


_ELIM_EDGES = np.array([10, 20, 40, 80, 160, 320, 640], dtype=np.int64)


def _synthetic_sim(real_slots: list[int], deaths: list[tuple[int, int]], T: int):
    """Minimal sim dict for the elim precompute: only `ownership[0]` (to recover
    the starting slots) and `death_events` (timestep, slot) are read."""
    own = np.full((T, 1, 8), -1, dtype=np.int64)
    own[0, 0, : len(real_slots)] = real_slots
    de = np.array(deaths, dtype=np.int32).reshape(-1, 2)
    return {"ownership": own, "death_events": de}


def test_elim_precompute_winner_vs_phantom() -> None:
    """`precompute_elim` distinguishes the three slot kinds: dead-real slots get
    their elim timestep, the winner (real, absent from deaths) the large finite
    sentinel, phantom slots (never played) the -1 marker."""
    # 4-player game: slots 0..3 real, 4..7 phantom. Slot 0 wins; 1/2/3 die.
    sim = _synthetic_sim([0, 1, 2, 3], [(10, 1), (20, 2), (30, 3)], T=40)
    death_by_slot, is_real, sentinel = precompute_elim(sim, _ELIM_EDGES)

    assert sentinel == 40 + 640
    assert list(death_by_slot) == [sentinel, 10, 20, 30, -1, -1, -1, -1]
    assert list(is_real) == [True, True, True, True, False, False, False, False]


def test_elim_targets_masks_phantom_and_dead_channels() -> None:
    """The load-bearing case: a sub-8-player game's non-existent channels are
    masked out (never trained as 'alive, never dies'), and a real-but-already-
    eliminated channel is masked too — only currently-alive real players carry a
    target."""
    sim = _synthetic_sim([0, 1, 2, 3], [(10, 1), (20, 2), (30, 3)], T=40)
    elim = ElimCtx(_ELIM_EDGES, *precompute_elim(sim, _ELIM_EDGES))
    # Perspective = slot 0 → canonical channel order [0,1,2,3,4,5,6,7].
    raw_order = [0, 1, 2, 3, 4, 5, 6, 7]

    bins, alive = time_bin_targets(elim, raw_order, t=5)
    # Phantom channels 4..7 masked out — the whole point of the real_slots mask.
    assert list(alive) == [True, True, True, True, False, False, False, False]
    # ch0 winner → top bin (Δ=675 ≥ 640); ch1/2/3 → bins by Δ = 5/15/25.
    assert list(bins) == [7, 0, 1, 2, 0, 0, 0, 0]

    # After slot 1's elimination at t=10: its channel is real but dead → masked.
    bins2, alive2 = time_bin_targets(elim, raw_order, t=15)
    assert not alive2[1]            # death (10) > 15 is False → dead
    assert alive2[0] and alive2[2] and alive2[3]
    assert not alive2[4:].any()    # phantoms still masked


def test_next_death_target_picks_soonest_and_masks_winner_tail() -> None:
    """who-dies-next target: the next victim is the alive channel with the
    soonest death; the alive mask is the softmax domain; winner-tail frames (only
    the winner left) return target/-dt = -1 so the loss ignores them."""
    # Slots 0..3 real; 0 wins; deaths at 10/20/30 for slots 1/2/3.
    sim = _synthetic_sim([0, 1, 2, 3], [(10, 1), (20, 2), (30, 3)], T=40)
    elim = ElimCtx(_ELIM_EDGES, *precompute_elim(sim, _ELIM_EDGES))
    raw_order = [0, 1, 2, 3, 4, 5, 6, 7]

    # At t=5: all four real players alive; soonest death is slot 1 at t=10.
    nxt, alive, dt = next_death_target(elim, raw_order, t=5)
    assert nxt == 1 and dt == 5
    assert list(alive) == [True, True, True, True, False, False, False, False]

    # At t=15 (slot 1 dead): next victim is slot 2 (death 20), dt=5.
    nxt2, alive2, dt2 = next_death_target(elim, raw_order, t=15)
    assert nxt2 == 2 and dt2 == 5
    assert not alive2[1]

    # At t=35 (all opponents dead, only the winner left): no real next death →
    # masked frame.
    nxt3, alive3, dt3 = next_death_target(elim, raw_order, t=35)
    assert nxt3 == -1 and dt3 == -1
    assert list(alive3) == [True, False, False, False, False, False, False, False]


def test_next_death_target_ties_resolve_to_lowest_channel() -> None:
    """Two deaths at the same tick resolve to the lowest canonical channel index
    (argmin's first-min rule) — the fixed tie convention."""
    # Slots 2 and 3 both die at t=20.
    sim = _synthetic_sim([0, 1, 2, 3], [(10, 1), (20, 2), (20, 3)], T=40)
    elim = ElimCtx(_ELIM_EDGES, *precompute_elim(sim, _ELIM_EDGES))
    raw_order = [0, 1, 2, 3, 4, 5, 6, 7]
    nxt, _alive, dt = next_death_target(elim, raw_order, t=15)
    assert nxt == 2 and dt == 5   # channel 2 over channel 3


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
