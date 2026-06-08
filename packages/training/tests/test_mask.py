"""
Targeted test for `bc.mask.build_mask` legality logic.

One synthetic 4×4 board exercises all four legality conditions plus the
split-independence invariant. Real-corpus coverage lives in
`test_dataloader_smoke.py`.
"""

from __future__ import annotations

import numpy as np

from training.bc import actions
from training.bc.mask import build_mask


def test_build_mask_legality_conditions():
    """
    Synthetic board, perspective = slot 0:

      (0,0) slot 0, armies 2   (0,1) slot 1, armies 10
      (1,0) MOUNTAIN            (1,1) slot 0, armies 1
      ...                       ...
      (3,3) slot 0, armies 5

    Expected outcomes per legality condition:
      (1) own slot:       (0,1) cannot source — slot 1 ≠ perspective
      (2) armies ≥ 2:     (1,1) cannot source — armies = 1
      (3) dest in bounds: (0,0) N/W blocked, (3,3) E/S blocked
      (4) dest passable:  (0,0) S blocked — dest (1,0) is a mountain
    Positive cases: (0,0) E and (3,3) N + W stay True.
    """
    H, W = 4, 4
    HW = H * W

    ownership = np.full((1, HW), -1, dtype=np.int8)
    ownership[0, 0] = 0    # (0,0) perspective
    ownership[0, 1] = 1    # (0,1) other slot
    ownership[0, 5] = 0    # (1,1) perspective
    ownership[0, 15] = 0   # (3,3) perspective
    ownership[0, 4] = -2   # (1,0) mountain (sentinel value used by parser)

    armies = np.zeros((1, HW), dtype=np.int16)
    armies[0, 0] = 2
    armies[0, 1] = 10
    armies[0, 5] = 1       # too few — condition (2)
    armies[0, 15] = 5

    sim = {
        "ownership": ownership,
        "armies": armies,
        "mountains": np.array([4], dtype=np.int32),  # (1,0) — used by condition (4)
    }

    mask = build_mask(sim, t=0, perspective_slot=0, H=H, W=W)

    # (0,0) — perspective, armies 2 (eligible source)
    assert not mask[0, 0, actions.N * 2 + 0], "N from (0,0): dest out of bounds (cond 3)"
    assert     mask[0, 0, actions.E * 2 + 0], "E from (0,0): dest (0,1) — positive case"
    assert not mask[0, 0, actions.S * 2 + 0], "S from (0,0): dest (1,0) is mountain (cond 4)"
    assert not mask[0, 0, actions.W * 2 + 0], "W from (0,0): dest out of bounds (cond 3)"

    # (0,1) — slot 1, not perspective: condition (1) fails for all directions
    assert not mask[0, 1].any(), "(0,1) not owned by perspective — no legal moves (cond 1)"

    # (1,1) — perspective but armies < 2: condition (2) fails for all directions
    assert not mask[1, 1].any(), "(1,1) armies < 2 — no legal moves (cond 2)"

    # (3,3) — bottom-right corner; N and W are positive, E and S out of bounds
    assert     mask[3, 3, actions.N * 2 + 0], "N from (3,3): dest (2,3) — positive case"
    assert not mask[3, 3, actions.E * 2 + 0], "E from (3,3): dest out of bounds (cond 3)"
    assert not mask[3, 3, actions.S * 2 + 0], "S from (3,3): dest out of bounds (cond 3)"
    assert     mask[3, 3, actions.W * 2 + 0], "W from (3,3): dest (3,2) — positive case"

    # Split-independence — the full-move and half-move sub-channels are constructed
    # from the same per-direction legality, so they must agree everywhere.
    assert np.array_equal(mask[..., 0::2], mask[..., 1::2])

    # Padded region is all False (the unpadded board is 4×4 inside a 32×32 mask)
    assert not mask[H:, :, :].any()
    assert not mask[:, W:, :].any()
