"""
Targeted tests for `bc.obs`. Slot canonicalization is the high-silent-bug-risk
surface; the rest of `build_obs` (89-channel full obs) is exercised end-to-end
by `test_dataloader_smoke.py` against real-corpus data.
"""

from __future__ import annotations

import pytest

from bc.obs import canonical_slot_order


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
