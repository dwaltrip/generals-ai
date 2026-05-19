"""
Per-cell legality mask for the policy head.

Encodes generals.io movement rules into a `bool [H_PADDED, W_PADDED, 8]`
tensor: which `(source_cell, direction, split)` triples are legal at a
given snapshot for the perspective player. Used at training time to mask
the policy softmax and at inference time to constrain sampled moves.
"""

from __future__ import annotations

import numpy as np

from bc import actions
from bc.constants import H_PADDED, W_PADDED


def build_mask(
    sim: dict[str, np.ndarray],
    t: int,
    perspective_slot: int,
    H: int,
    W: int,
) -> np.ndarray:
    """
    Per-cell legality mask for the policy head.

    Returns `bool [H_PADDED, W_PADDED, 8]`. Index layout matches `bc.actions`:
    `sub = dir * 2 + split` with directions in NESW order (N=0, E=1, S=2, W=3).
    A move `(source_cell, direction, split)` is legal at time `t` for the
    perspective player iff all four conditions hold:

      (1) `ownership[t, source_cell] == perspective_slot` — source is owned by
          perspective (raw slot id, NOT canonicalized channel 0; canonicalization
          only permutes the obs tensor, raw `ownership` values stay in their
          original slot space)
      (2) `armies[t, source_cell] >= 2` — need >1 army so one can leave (the
          generals.io rule). Same threshold for full and half moves.
      (3) destination cell is in-bounds — edge cells lose 1–3 directions
      (4) destination is not a mountain — cities + generals are passable

    Split-independence: conditions (1)–(4) are all split-invariant, so
    `mask[r, c, dir*2+0] == mask[r, c, dir*2+1]` by construction. The mask
    repeats each per-direction legality into both sub-channels.

    Padded cells (`r >= H` or `c >= W`) are all False — initial zero allocation
    handles this; the assignment only writes the unpadded region.
    """
    HW = H * W

    # Source-cell legality — (1) own + (2) armies — combined as a [H, W] mask.
    own_2d = sim["ownership"][t].reshape(H, W)
    armies_2d = sim["armies"][t].reshape(H, W)
    source_ok = (own_2d == perspective_slot) & (armies_2d >= 2)

    # Densify mountains for (4). Static within a game; per-frame cost is
    # microseconds (could lift to a per-game precompute if profiling shows it).
    mountains = np.zeros(HW, dtype=bool)
    mountains[sim["mountains"]] = True
    not_mountain_2d = ~mountains.reshape(H, W)

    # Per-direction legality via shifted-slice arithmetic. For each direction,
    # we restrict the source region to cells whose destination falls in-bounds,
    # then AND with the (shifted) destination passability. The "missing" rows/
    # cols (where the destination would be out of bounds) stay False from the
    # zero allocation — that's how condition (3) gets baked in.
    legal_per_dir = np.zeros((H, W, 4), dtype=bool)
    # N (dest_row = r - 1): valid sources are rows 1..H-1; dest is rows 0..H-2.
    legal_per_dir[1:, :, actions.N] = source_ok[1:, :] & not_mountain_2d[:-1, :]
    # E (dest_col = c + 1): valid sources are cols 0..W-2; dest is cols 1..W-1.
    legal_per_dir[:, :-1, actions.E] = source_ok[:, :-1] & not_mountain_2d[:, 1:]
    # S (dest_row = r + 1): valid sources are rows 0..H-2; dest is rows 1..H-1.
    legal_per_dir[:-1, :, actions.S] = source_ok[:-1, :] & not_mountain_2d[1:, :]
    # W (dest_col = c - 1): valid sources are cols 1..W-1; dest is cols 0..W-2.
    legal_per_dir[:, 1:, actions.W] = source_ok[:, 1:] & not_mountain_2d[:, :-1]

    # Expand 4 directions → 8 sub-channels by repeating each direction twice
    # (split-independence). `[a, b, c, d]` becomes `[a, a, b, b, c, c, d, d]`.
    mask_unpadded = np.repeat(legal_per_dir, 2, axis=-1)

    mask = np.zeros((H_PADDED, W_PADDED, 8), dtype=bool)
    mask[:H, :W, :] = mask_unpadded
    return mask
