"""Tests for `bc.dataset` (shuffle-buffer helper, IterableDataset worker
semantics, `_walk` elimination cutoff) and `bc.filters.is_eligible`."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from bc.constants import ELIGIBLE_PLAYER_COUNT, MAX_BOARD_SIDE
from bc.dataset import IterableDataset, _shuffle_buffered
from bc.filters import is_eligible


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
    `get_worker_info()` into our adapter, which is PyTorch's responsibility."""
    samples = [(Path(f"/synth/g{i:03d}.npz"), 0) for i in range(40)]
    ds = IterableDataset(samples=samples, seed=0)

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
    ds = _IdentityDataset(samples=samples, seed=0)

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
            ds = IterableDataset(samples=[(sim_path, k)], seed=0)
            walked = sum(1 for _ in ds._walk(ds._groups))
            assert walked == elim_t, (
                f"perspective k={k} in {sim_path.name}: "
                f"expected {elim_t} frames (elim_timestep), got {walked}"
            )
            return
    pytest.skip("no eliminated perspectives in fixture")


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
