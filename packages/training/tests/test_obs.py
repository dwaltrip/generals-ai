"""
Targeted unit tests for slot canonicalization (`bc.slots`) and the
dense-history encoders (`_encode_ownership_transition`, `_encode_army_delta`).
Full `build_obs` is exercised end-to-end by `test_dataloader_smoke.py` against
real-corpus data.
"""

from __future__ import annotations

import numpy as np
import pytest

from training.bc.slots import SlotOrder
from training.bc.obs.memory import _encode_army_delta, _encode_ownership_transition


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
    slot_order = SlotOrder.for_perspective(perspective_slot)
    assert list(slot_order.order) == expected
    assert slot_order.perspective == expected[0]
    assert slot_order.opp_slots == expected[1:]


def test_encode_ownership_transition_all_categories():
    """Per-cell lookup table from 5.05-1 §G. Uses perspective=3 so the raw
    slot → canonical channel index mapping is non-trivial (raw 0→ch 1,
    raw 4→ch 4, etc.). Each cell exercises one category."""
    perspective = 3
    opp_slots = SlotOrder.for_perspective(perspective).opp_slots  # [0, 1, 2, 4, 5, 6, 7]

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


def test_encode_army_delta_production_subtraction():
    """Production subtraction at per-2-step + land-tick boundaries (5.05-1 §G).

    `_encode_army_delta` returns the raw adjusted delta (`raw - prod`); the
    log scaling lives in `_signed_log` at the call site. Each scenario is
    crafted so adjusted is either 0 (production fully subtracted) or +1
    (not), checked directly without re-deriving the log.
    """
    # --- Scenario A: t_newer=50 (per-2-step AND land-tick both fire) ---
    # cell 0: city  owned → prod=2 (city + land-tick), raw=2 → adj=0
    # cell 1: general owned → prod=2, raw=2 → adj=0
    # cell 2: plain owned → prod=1 (land-tick only), raw=1 → adj=0
    # cell 3: neutral city (unowned) → prod=0, raw=1 → adj=1
    # cell 4: mountain → prod=0, raw=0 → adj=0
    # cell 5: plain owned with raw=2 → prod=1, adj=1 (catches "subtract too much")
    own_newer = np.array([[3, 3, 3, -1, -2, 3]], dtype=np.int8)
    arm_older = np.array([[10, 10, 10, 10, 0, 10]], dtype=np.int16)
    arm_newer = np.array([[12, 12, 11, 11, 0, 12]], dtype=np.int16)
    city_mask = np.array([[True, False, False, True, False, False]])
    general_mask = np.array([[False, True, False, False, False, False]])
    expected = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 1.0]], dtype=np.float32)
    np.testing.assert_array_equal(
        _encode_army_delta(
            arm_newer, arm_older, own_newer, 50, city_mask, general_mask,
        ),
        expected,
    )

    # --- Scenario B: t_newer=2 (per-2-step only; not land-tick) ---
    # cell 0: city owned → prod=1, raw=1 → adj=0
    # cell 1: plain owned → prod=0 (no land-tick), raw=1 → adj=1
    # cell 2: general owned → prod=1, raw=1 → adj=0
    own_newer = np.array([[3, 3, 3]], dtype=np.int8)
    arm_older = np.array([[10, 10, 10]], dtype=np.int16)
    arm_newer = np.array([[11, 11, 11]], dtype=np.int16)
    city_mask = np.array([[True, False, False]])
    general_mask = np.array([[False, False, True]])
    expected = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    np.testing.assert_array_equal(
        _encode_army_delta(
            arm_newer, arm_older, own_newer, 2, city_mask, general_mask,
        ),
        expected,
    )

    # --- Scenario C: t_newer=3 (odd → neither rule fires) ---
    # cell 0: city owned → prod=0, raw=1 → adj=1 (catches "fired on odd t")
    own_newer = np.array([[3]], dtype=np.int8)
    arm_older = np.array([[10]], dtype=np.int16)
    arm_newer = np.array([[11]], dtype=np.int16)
    city_mask = np.array([[True]])
    general_mask = np.array([[False]])
    expected = np.array([[1.0]], dtype=np.float32)
    np.testing.assert_array_equal(
        _encode_army_delta(
            arm_newer, arm_older, own_newer, 3, city_mask, general_mask,
        ),
        expected,
    )
