"""
Per-perspective Moore-neighborhood visibility computation.

A cell is visible to a player iff it's within the 3x3 Moore neighborhood of
any cell that player currently owns (including the cell itself). Pure
function of one ownership snapshot — no history, no fog memory. The fog
*memory* channels (last_seen_owner, turns_since_seen, etc.) live in
`obs.MemoryState` and are advanced by `step_memory`; visibility is just
the first ingredient that step_memory consumes each tick.

Reference: `generals-io-game-mechanics.md` line 81 (Moore-neighborhood
vision rule). Confirmed by parser-side test fixtures + sim-parity
validator behavior.
"""

from __future__ import annotations

import numpy as np


def compute_visibility(
    ownership_t: np.ndarray,
    perspective_slot: int,
    H: int,
    W: int,
) -> np.ndarray:
    """
    Compute the perspective player's visibility mask at one snapshot.

    Args:
        ownership_t: int8 [H*W] ownership at this tick (-2 mountain, -1 neutral,
            0..7 player slots).
        perspective_slot: raw slot id of the perspective (NOT canonicalized).
        H, W: unpadded board dims.

    Returns:
        bool [H, W] mask. True where this cell is visible to perspective.
    """
    owned = (ownership_t.reshape(H, W) == perspective_slot)
    vis = owned.copy()

    # OR in each of the 8 neighbor-shifts. Equivalent to a 3x3 binary dilation
    # but with no scipy dependency. Each line: "if a cell is owned, the cell
    # offset in this direction also gets vision."
    # N (owned at (r, c) → visible at (r-1, c))
    vis[:-1, :] |= owned[1:, :]
    # S
    vis[1:, :] |= owned[:-1, :]
    # W
    vis[:, :-1] |= owned[:, 1:]
    # E
    vis[:, 1:] |= owned[:, :-1]
    # NW
    vis[:-1, :-1] |= owned[1:, 1:]
    # NE
    vis[:-1, 1:] |= owned[1:, :-1]
    # SW
    vis[1:, :-1] |= owned[:-1, 1:]
    # SE
    vis[1:, 1:] |= owned[:-1, :-1]

    return vis
