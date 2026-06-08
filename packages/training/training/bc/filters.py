"""
Game- and perspective-level eligibility filters for the BC training corpus.

`is_eligible` is the game-level filter (cheap — peeks at sim scalars only).
`eligible_perspectives` is the full pipeline: game-level gate plus per-perspective
gates for curated-name membership and signal quality (elim time, rolling rates).

`FILTER_VERSION` is a manually-bumped string stamped into the split manifest.
Bump it whenever filter logic changes so manifests built under different
filter sets are distinguishable from their provenance fields.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from training.bc.constants import ELIGIBLE_PLAYER_COUNT, MAX_BOARD_SIDE


FILTER_VERSION = "v2"

# Game-level threshold. Very long games are usually sub-optimal play.
MAX_GAME_LENGTH = 2000

# Perspective-level thresholds.
# TODO: Should be configurable? It would change meaning of FILTER_VERSION.

# Short-lived perspectives are more likely to contain poor early-game play.
MIN_ELIM_TIMESTEP = 120

# Rolling-rate cutoffs sit above the parser's noise floor (0.125 / 0.375 in
# `replay_parser/driver.py`). The noise floor is a mild skill floor.
# These cutoffs curate sharply - isolating strong play more robustly.
# It also protects against pulling older games for a currently strong player
# who didn't always play this well.
MIN_ROLLING_1ST_RATE = 0.225
MIN_ROLLING_TOP3_RATE = 0.475

# Drop-reason keys reported by `eligible_perspectives` when the caller passes
# in a `drop_counts` dict. Listed here so callers can initialize all keys to
# zero before the scan — keeps zero-count filters visible in the summary and
# avoids stray-key bugs.
DROP_REASONS = ("non_curated", "early_elim", "low_1st_rate", "low_top3_rate")


def is_eligible(sim_path: Path) -> bool:
    """
    True iff the per-game sim file passes the game-level drop filter.

    Three conditions: `max(map_width, map_height) <= MAX_BOARD_SIDE`,
    player count == ELIGIBLE_PLAYER_COUNT, and `T <= MAX_GAME_LENGTH`.
    Map dims are checked first — they're 0-d scalars (~8 bytes). Player count
    and game length both come from `actions_source.shape` (shape `[P, T-1]`)
    in one load (~32 KB), so adding the game-length check is free once we're
    paying for actions_source. The ~7% of games dropped on map dims
    short-circuit before that larger read.
    """
    with np.load(sim_path) as sim:
        w = int(sim["map_width"])
        h = int(sim["map_height"])
        if max(w, h) > MAX_BOARD_SIDE:
            return False
        actions_shape = sim["actions_source"].shape
    p, t_minus_1 = actions_shape
    if p != ELIGIBLE_PLAYER_COUNT:
        return False
    return (t_minus_1 + 1) <= MAX_GAME_LENGTH


def eligible_perspectives(
    sim_path: Path,
    meta_path: Path,
    curated_names: set[str],
    drop_counts: dict[str, int] | None = None,
) -> list[int]:
    """
    List of perspective indices for this game that pass all training filters.

    Empty list means "this game contributes no training samples." Pipeline:
    game-level gate via `is_eligible`, then per-perspective gates for
    curated-name membership and signal quality (elim timestep, rolling 1st
    and top-3 rates).

    If `drop_counts` is supplied, the function bumps it once per rejected
    perspective, keyed by reason (`DROP_REASONS`). The caller owns the dict
    and is expected to pre-initialize all `DROP_REASONS` keys to zero.
    Game-level drops (from `is_eligible`) are not broken out here — those
    short-circuit before the per-perspective loop runs.

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

    TODO: Revisit in the future and decide whether it's worth keeping
    `curated_names` here, or if it's confusing having a similar filter
    here and in the parser.
    """
    if not is_eligible(sim_path):
        return []
    with np.load(meta_path) as meta_npz:
        usernames = meta_npz["perspective_usernames"]
        elim = meta_npz["elim_timestep"]
        r1st = meta_npz["rolling_1st_rate"]
        rtop3 = meta_npz["rolling_top3_rate"]
    kept: list[int] = []
    for k, name in enumerate(usernames):
        if str(name) not in curated_names:
            if drop_counts is not None:
                drop_counts["non_curated"] += 1
            continue
        # elim_timestep == -1 means "survived to game end" — keep.
        e = int(elim[k])
        if e != -1 and e < MIN_ELIM_TIMESTEP:
            if drop_counts is not None:
                drop_counts["early_elim"] += 1
            continue
        if float(r1st[k]) <= MIN_ROLLING_1ST_RATE:
            if drop_counts is not None:
                drop_counts["low_1st_rate"] += 1
            continue
        if float(rtop3[k]) <= MIN_ROLLING_TOP3_RATE:
            if drop_counts is not None:
                drop_counts["low_top3_rate"] += 1
            continue
        kept.append(k)
    return kept
