"""Per-perspective metadata builder (sidecar to the sim output).

Schema: see `docs/2026-05/5.16-3-parser-output-design.md` §4.1.

Most fields come from the wire static block, DB, or sim event lists. The
rolling-rate / prior-games fields take optional lists; passing `None`
leaves the documented sentinels (`-1.0` / `-1`) in place.

`sim_core_version` is supplied by the caller (the corpus driver captures
it at run-time via `git rev-parse` + dirty-flag, so each meta sidecar is
stamped with the exact build that produced its sibling sim file).
"""

import numpy as np

from replay_parser.decode import ReplayData
import sim_core


def build_metadata(
    state: sim_core.State,
    replay: ReplayData,
    perspective_player_ids: list[int],
    placement: list[int],
    sim_core_version: str,
    rolling_1st_rate: list[float] | None = None,
    rolling_top3_rate: list[float] | None = None,
    prior_games_count: list[int] | None = None,
) -> dict[str, np.ndarray]:
    K = len(perspective_player_ids)
    if len(placement) != K:
        raise ValueError(f"placement length {len(placement)} != K={K}")

    stars_at_start = [replay.static.stars[p] for p in perspective_player_ids]
    perspective_usernames = [replay.static.usernames[p] for p in perspective_player_ids]

    death_by_player: dict[int, int] = {e.player: e.timestep for e in state.death_events}
    elim_timestep = [death_by_player.get(p, -1) for p in perspective_player_ids]

    rolling_1st = rolling_1st_rate if rolling_1st_rate is not None else [-1.0] * K
    rolling_top3 = rolling_top3_rate if rolling_top3_rate is not None else [-1.0] * K
    prior_games = prior_games_count if prior_games_count is not None else [-1] * K

    return {
        "replay_id": np.asarray(replay.static.id, dtype="<U16"),
        "sim_core_version": np.asarray(sim_core_version, dtype="<U24"),
        "perspective_player_ids": np.asarray(perspective_player_ids, dtype=np.int8),
        "perspective_usernames": np.asarray(perspective_usernames, dtype="<U32"),
        "placement": np.asarray(placement, dtype=np.int8),
        "stars_at_start": np.asarray(stars_at_start, dtype=np.float32),
        "elim_timestep": np.asarray(elim_timestep, dtype=np.int32),
        "rolling_1st_rate": np.asarray(rolling_1st, dtype=np.float32),
        "rolling_top3_rate": np.asarray(rolling_top3, dtype=np.float32),
        "prior_games_count": np.asarray(prior_games, dtype=np.int32),
    }
