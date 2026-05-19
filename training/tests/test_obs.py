"""Targeted tests for `bc.obs`. Slot canonicalization is the high-silent-bug-risk surface."""

from __future__ import annotations

import numpy as np
import pytest

from bc.obs import build_obs, canonical_slot_order


@pytest.mark.parametrize(
    "perspective_slot,expected",
    [
        (0, [0, 1, 2, 3, 4, 5, 6, 7]),
        (1, [1, 0, 2, 3, 4, 5, 6, 7]),
        (3, [3, 0, 1, 2, 4, 5, 6, 7]),
        (5, [5, 0, 1, 2, 3, 4, 6, 7]),
        (7, [7, 0, 1, 2, 3, 4, 5, 6]),
    ],
)
def test_canonical_slot_order(perspective_slot, expected):
    """Channel 0 = perspective; channels 1..7 = ascending raw slot, perspective skipped."""
    assert canonical_slot_order(perspective_slot) == expected


def test_build_obs_channel_zero_holds_perspective_cells():
    """
    End-to-end check that `build_obs` actually applies canonicalization.

    Synthetic 4×4 board where slot-3 owns the top-left cell and slot-0 owns the
    bottom-right. When we build obs from slot-3's perspective, channel 0 must
    light up at the top-left (not the bottom-right). Catches "we ignored the
    permutation" bugs that the pure-function test above can't see.
    """
    H, W = 4, 4
    HW = H * W
    ownership = np.full((1, HW), -1, dtype=np.int8)
    ownership[0, 0] = 3   # slot 3 owns top-left
    ownership[0, 15] = 0  # slot 0 owns bottom-right
    sim = {
        "ownership": ownership,
        "armies": np.zeros((1, HW), dtype=np.int16),
        "mountains": np.array([], dtype=np.int32),
        "cities": np.array([], dtype=np.int32),
        "cities_present_at": np.array([], dtype=np.int32),
    }

    obs = build_obs(sim, t=0, perspective_slot=3, H=H, W=W)

    assert obs[0, 0, 0] == 1.0, "channel 0 should light up where perspective (slot 3) owns"
    assert obs[0, 3, 3] == 0.0, "channel 0 should NOT light up where slot 0 owns"

    # Slot 0 lands in channel 1 (first opponent in ascending-skip order: [3, 0, 1, 2, 4, 5, 6, 7]).
    assert obs[1, 3, 3] == 1.0
    assert obs[1, 0, 0] == 0.0
