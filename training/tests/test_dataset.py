"""Tests for `bc.dataset` (shuffle-buffer helper) and `bc.filters.is_eligible`."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from bc.constants import ELIGIBLE_PLAYER_COUNT, MAX_BOARD_SIDE
from bc.dataset import _shuffle_buffered
from bc.filters import is_eligible


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
