"""
Targeted tests for `bc.obs`. Slot canonicalization is the high-silent-bug-risk
surface; the rest of `build_obs` (96-channel full obs) is exercised end-to-end
by `test_dataloader_smoke.py` against real-corpus data.
"""

from __future__ import annotations

import numpy as np
import pytest

from bc.obs import _encode_ownership_transition, canonical_slot_order


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


def test_encode_ownership_transition_all_categories():
    """Per-cell lookup table from 5.05-1 §G. Uses perspective=3 so the raw
    slot → canonical channel index mapping is non-trivial (raw 0→ch 1,
    raw 4→ch 4, etc.). Each cell exercises one category."""
    perspective = 3
    opp_slots = canonical_slot_order(perspective)[1:]  # [0, 1, 2, 4, 5, 6, 7]

    # (older, newer, expected) — one cell per category. Gainer's identity
    # doesn't affect the encoding; only the older owner matters.
    cases = [
        (3, 3, 0.0),     # self, no change
        (3, -1, -1.0),   # self lost (to neutral)
        (-1, 3, 0.5),    # neutral lost (to self)
        (-1, 0, 0.5),    # neutral lost (to opp) — gainer doesn't matter
        (0, 3, 1.125),   # opp at canonical channel 1 lost
        (1, -1, 1.25),   # opp at canonical channel 2 lost
        (2, 4, 1.375),   # opp at canonical channel 3 lost
        (4, 5, 1.5),     # opp at canonical channel 4 lost
        (5, 6, 1.625),   # opp at canonical channel 5 lost
        (6, 7, 1.75),    # opp at canonical channel 6 lost
        (7, 0, 1.875),   # opp at canonical channel 7 lost
        (-2, 3, 0.0),    # mountain → owned: defensive, no special category
    ]

    own_older = np.array([[c[0] for c in cases]], dtype=np.int8)
    own_newer = np.array([[c[1] for c in cases]], dtype=np.int8)
    expected = np.array([[c[2] for c in cases]], dtype=np.float32)

    result = _encode_ownership_transition(
        own_newer, own_older, perspective, opp_slots,
    )
    np.testing.assert_array_equal(result, expected)
