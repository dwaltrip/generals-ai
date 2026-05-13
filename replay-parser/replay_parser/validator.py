import numpy as np

from replay_parser.state import State
from replay_parser.types import PlayerIndex


def deduce_ranking(state: State) -> list[PlayerIndex]:
    """Compute the final ranking from a parsed State per the bundle's lbSort
    (5.11-2 §3.6). Returns slot indices in rank order (winner first).

    Ranking keys:
      1. has_kill desc (players with kills outrank kill-less players)
      2. alive desc (alive outrank dead)
      3. Among dead: later death ranks higher (uses death_events index order)
      4. army desc, tiles desc, player-index asc
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

    death_order = {de.player: i for i, de in enumerate(state.death_events)}

    def sort_key(p: PlayerIndex) -> tuple[int, int, int, int, int, int]:
        return (
            0 if state.has_kill[p] else 1,
            0 if state.alive[p] else 1,
            -death_order[p] if not state.alive[p] else 0,
            -int(armies_per_p[p]),
            -int(tiles_per_p[p]),
            p,
        )

    return sorted(range(state.num_players), key=sort_key)
