import numpy as np
import sim_core

from replay_parser.types import PlayerIndex


# Empirical bounds on the v30.9.2 deploy time, derived by bisecting our replay
# corpus against the two lbSort variants (old: no partition; new: kill/no-kill
# partition) — see scripts/find_lbsort_deploy_time.py.
#
# The changelog dates v30.9.2 as 2025-11-29; the deploy actually happened ~05:30
# UTC on 2025-11-30, which is ~21:30 on 2025-11-29 Pacific Time (matching the
# changelog, since the lead dev is in PT).
#
# Within the 32-minute window between PRE and POST, every discriminating game
# is BOTH-rule-agree, so the data can't pin the deploy further. Code that needs
# a definitive ranking for a replay inside the gap must pick one side or skip.
PRE_V30_9_2_CUTOFF_MS = 1764480441447   # 2025-11-30 05:27:21.447 UTC — last OLD-rule replay
POST_V30_9_2_CUTOFF_MS = 1764482381217  # 2025-11-30 05:59:41.217 UTC — first NEW-rule replay


def deduce_ranking_for_replay(
    state: sim_core.State,
    started_ms: int,
) -> list[PlayerIndex]:
    """Compute the ranking using the lbSort rule that was live when the game
    was played. Raises ValueError for replays inside the 32-minute deploy
    ambiguity window."""
    if started_ms <= PRE_V30_9_2_CUTOFF_MS:
        return deduce_ranking(state, partition_kill_no_kill=False)
    if started_ms >= POST_V30_9_2_CUTOFF_MS:
        return deduce_ranking(
            state,
            partition_kill_no_kill=True,
            has_kill=apply_surrender_bonus(state),
        )
    raise ValueError(
        f"started_ms={started_ms} falls inside the v30.9.2 deploy ambiguity "
        f"window ({PRE_V30_9_2_CUTOFF_MS}, {POST_V30_9_2_CUTOFF_MS})"
    )


def apply_surrender_bonus(state: State) -> list[bool]:
    """Return effective has_kill after applying the server's surrender-bonus
    rule: each surrender awards +1 has_kill to the top offense-only damager
    who is still alive when the surrenderer's tiles fully revert to neutral.

    Captured deaths don't trigger the bonus (the capture itself already credits
    the captor via execute_player_capture). The "alive at neutralization"
    qualifier — rather than "alive at surrender" — handles cases where the
    top damager gets killed during the AFK countdown: they forfeit the credit
    to the next-highest damager who survives long enough. If the surrenderer
    never fully neutralizes (game ends first), end-of-game is used as the
    cutoff.
    """
    captured = {ce.captured for ce in state.capture_events}
    death_t = {de.player: de.timestep for de in state.death_events}
    neutralize_t = {ne.player: ne.timestep for ne in state.neutralize_events}
    end_t = state.timestep
    has_kill = list(state.has_kill)
    n = state.num_players
    for de in state.death_events:
        if de.player in captured:
            continue
        cutoff = neutralize_t.get(de.player, end_t)
        best, best_val = None, 0
        for q in range(n):
            if q == de.player:
                continue
            if death_t.get(q, float("inf")) <= cutoff:
                continue   # damager died before surrenderer neutralized
            v = int(state.damage_off_all[q, de.player])
            if v > best_val:
                best, best_val = q, v
        if best is not None:
            has_kill[best] = True
    return has_kill


def deduce_ranking(
    state: sim_core.State,
    *,
    partition_kill_no_kill: bool = True,
    has_kill: list[bool] | None = None,
) -> list[PlayerIndex]:
    """Compute the final ranking from a parsed State per the bundle's lbSort.
    Returns slot indices in rank order (winner first).

    Ranking keys:
      1. has_kill desc (only if partition_kill_no_kill=True)
      2. alive desc
      3. Among dead: later death ranks higher
      4. army desc, tiles desc, player-index asc

    The kill/no-kill partition was added to lbSort in bundle v30.9.2
    (deployed ~2025-11-30 05:30 UTC). Pre-v30.9.2 bundles (e.g. v30.8.5
    line 23586) and the server's listings API prior to that time used no
    partition. For replays played before the patch, pass
    partition_kill_no_kill=False — or use deduce_ranking_for_replay() to
    pick the right rule from the replay's `started` timestamp.
    """
    owned_mask = state.ownership >= 0
    if owned_mask.any():
        owners = state.ownership[owned_mask]
        armies_per_p = np.bincount(
            owners,
            weights=state.armies[owned_mask].astype(np.int64),
            minlength=state.num_players,
        ).astype(np.int64)
        tiles_per_p = np.bincount(owners, minlength=state.num_players)
    else:
        armies_per_p = np.zeros(state.num_players, dtype=np.int64)
        tiles_per_p = np.zeros(state.num_players, dtype=np.int64)

    if has_kill is None:
        has_kill = state.has_kill
    death_order = {de.player: i for i, de in enumerate(state.death_events)}

    def sort_key(p: PlayerIndex) -> tuple:
        partition = (0 if has_kill[p] else 1,) if partition_kill_no_kill else ()
        return (
            *partition,
            0 if state.alive[p] else 1,
            -death_order[p] if not state.alive[p] else 0,
            -int(armies_per_p[p]),
            -int(tiles_per_p[p]),
            p,
        )

    return sorted(range(state.num_players), key=sort_key)
