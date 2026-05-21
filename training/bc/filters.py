"""
Game- and perspective-level eligibility filters for the BC training corpus.

`is_eligible` is the game-level filter (cheap — peeks at sim scalars only).
`eligible_perspectives` is the full pipeline: game-level gate + future
per-perspective gates (e.g., short-eliminated perspectives, low-skill
opponents). v1 is all-or-nothing — if the game is eligible, every
perspective is eligible.

`FILTER_VERSION` is a manually-bumped string stamped into the split manifest.
Bump it whenever filter logic changes so manifests built under different
filter sets are distinguishable from their provenance fields.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bc.constants import ELIGIBLE_PLAYER_COUNT, MAX_BOARD_SIDE


FILTER_VERSION = "v1"


def is_eligible(sim_path: Path) -> bool:
    """
    True iff the per-game sim file passes the game-level drop filter.

    Two conditions: `max(map_width, map_height) <= MAX_BOARD_SIDE` and
    player count == ELIGIBLE_PLAYER_COUNT. Map dims are checked first —
    they're 0-d scalars (~8 bytes), while player count requires reading
    `actions_source.shape` which materializes a ~32 KB array. The ~7% of
    games dropped on map dims short-circuit before paying the larger read.
    """
    with np.load(sim_path) as sim:
        w = int(sim["map_width"])
        h = int(sim["map_height"])
        if max(w, h) > MAX_BOARD_SIDE:
            return False
        p = sim["actions_source"].shape[0]
    return p == ELIGIBLE_PLAYER_COUNT


def eligible_perspectives(sim_path: Path, meta_path: Path) -> list[int]:
    """
    List of perspective indices for this game that pass all training filters.

    Empty list means "this game contributes no training samples." v1 is
    all-or-nothing: either the game-level filter passes and every perspective
    is in, or nothing is. Per-perspective gates land here as they're added.
    """
    if not is_eligible(sim_path):
        return []
    with np.load(meta_path) as meta:
        n_perspectives = int(meta["perspective_player_ids"].shape[0])
    return list(range(n_perspectives))
