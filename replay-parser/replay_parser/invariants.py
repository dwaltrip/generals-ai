"""Per-snapshot invariant checks for the simulator.

Snapshot/event timing convention: see
`tools/timestep_viewer/extract.py` docstring.
"""
from dataclasses import dataclass

import numpy as np
import sim_core

from replay_parser.decode import ReplayData
from replay_parser.types import PlayerIndex, Timestep


@dataclass(frozen=True)
class Violation:
    kind: str
    t: Timestep | None
    detail: str = ""


def check_invariants(state: sim_core.State, replay: ReplayData) -> list[Violation]:
    v: list[Violation] = []
    s = replay.static
    initial_generals = list(s.initial_generals)
    map_size = s.map_width * s.map_height

    mountain_mask = np.zeros(map_size, dtype=bool)
    mountain_mask[list(s.mountains)] = True

    # Per-player resolution timestep + new owner for each general tile.
    # Captures take precedence over neutralizes (a captured player's general
    # flips owner; a neutralize on an already-captured player is a no-op).
    flip: dict[PlayerIndex, tuple[Timestep, int]] = {}
    for ce in state.capture_events:
        flip[ce.captured] = (ce.timestep, ce.captor)
    for ne in state.neutralize_events:
        flip.setdefault(ne.player, (ne.timestep, -1))

    # Bind once: each getter access clones the full snapshot list, so
    # indexing through the getter inside the loop is O(T²).
    snaps_own = state.snapshots_ownership
    snaps_arm = state.snapshots_armies
    snaps_cm  = state.snapshots_cities_mask
    T = state.snapshots_len

    # Per-timestep structural checks.
    prev_cm = None
    for t in range(T):
        own = snaps_own[t]   # int8[map_size]
        arm = snaps_arm[t]   # int16[map_size]
        cm  = snaps_cm[t]    # uint8[map_size]

        if not (own[mountain_mask] == -2).all():
            v.append(Violation("mountain_owner_changed", t))
        if (own[~mountain_mask] == -2).any():
            v.append(Violation("non_mountain_became_mountain", t))
        if (arm[mountain_mask] != 0).any():
            v.append(Violation("mountain_has_army", t))

        if (arm < 0).any():
            v.append(Violation("negative_army", t))

        if prev_cm is not None and (cm < prev_cm).any():
            v.append(Violation("cities_mask_decreased", t))
        prev_cm = cm

    # General-tile boundary checks: every general starts owned by its player,
    # and each flip event must match a single-step ownership transition on
    # that tile. Interior timesteps aren't checked — chained captures (q
    # gets captured after capturing p) move p's general tile to a third
    # owner via the mass re-ownership in execute_player_capture, and that
    # chain isn't directly derivable from the events alone.
    own_initial = snaps_own[0]
    own_final = snaps_own[T - 1]
    for p, gt in enumerate(initial_generals):
        if gt < 0:
            continue
        if own_initial[gt] != p:
            v.append(Violation(
                "initial_general_not_owned",
                0,
                f"p={p} tile={gt} got={int(own_initial[gt])}",
            ))
        ft = flip.get(p)
        if ft is None:
            # Player kept their general the entire game.
            if own_final[gt] != p:
                v.append(Violation(
                    "no_flip_but_general_changed",
                    T - 1,
                    f"p={p} tile={gt} got={int(own_final[gt])}",
                ))
            continue
        t_flip, new_owner = ft
        if t_flip < T:
            own_pre = snaps_own[t_flip]
            if own_pre[gt] != p:
                v.append(Violation(
                    "general_pre_flip_wrong",
                    t_flip,
                    f"p={p} tile={gt} got={int(own_pre[gt])} want={p}",
                ))
        if t_flip + 1 < T:
            # The captured player must not still own their general tile after
            # the event resolves. Stronger checks (which exact owner is on the
            # tile) need sim re-execution: same-timestep chained captures and
            # normal-combat recaptures of the flipped tile both happen and
            # aren't derivable from the event list alone.
            own_post = snaps_own[t_flip + 1]
            if own_post[gt] == p:
                v.append(Violation(
                    "general_post_flip_still_p",
                    t_flip + 1,
                    f"p={p} tile={gt}",
                ))

    return v
