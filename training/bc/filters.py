"""
Game- and perspective-level eligibility filters for the BC training corpus.

`is_eligible` is the game-level filter (cheap — peeks at sim scalars only).
`eligible_perspectives` is the full pipeline: game-level gate plus a
per-perspective curated-name gate (a perspective enters training only if
its username appears in the caller-supplied curated set).

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


def eligible_perspectives(
    sim_path: Path,
    meta_path: Path,
    curated_names: set[str],
) -> list[int]:
    """
    List of perspective indices for this game that pass all training filters.

    Empty list means "this game contributes no training samples." Game-level
    filter via `is_eligible` plus the per-perspective curated-name gate: a
    perspective is kept iff its username is in `curated_names`.

    NOTE: `curated_names` was added during a session where it wasn't yet
    clear whether the parser-produced `meta["perspective_usernames"]` had
    already been intersected with the curated-player list. It turns out
    the parser *does* this intersection at parse time
    (`replay_parser/driver.py` drops non-curated perspectives before
    writing the meta file), so against the live corpus today this filter
    is a no-op — every name we see here is already in the curated set.

    Kept anyway for two reasons:
      1. Defense-in-depth: a future parser change that silently introduces
         non-curated perspectives would otherwise leak into training.
      2. Tighter-filtering hook: a training run can pass a *subset* of the
         parser's curated list to train on a narrower group (e.g., the
         top-N of the 205 curated players). The parser fixes the outer
         set; this gate lets training pick a finer slice without
         re-parsing the corpus.
    
    Cost is one set-membership check per perspective.

    TODO: Revisit in the future and decide whether it's worth keeping,
    or if it's confusing having a similar filter here and in the parser.
    """
    if not is_eligible(sim_path):
        return []
    with np.load(meta_path) as meta:
        usernames = meta["perspective_usernames"]
    return [k for k, name in enumerate(usernames) if str(name) in curated_names]
