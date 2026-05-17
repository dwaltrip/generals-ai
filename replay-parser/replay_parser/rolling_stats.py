"""Rolling-rate skill stats per (replay, player), computed from listing data.

A single chronological walk over `ladder_id='ffa'` games maintains a
200-game rolling deque per filtered player. For each (replay, player)
pair where `player` is in `player_names`, snapshot the rolling state
**strictly before** updating with this game's result, per the
"strictly prior" convention in `replay-parser-design.md` §5.2.

Cost: tens of thousands of records for a curated list of ~tens of players
across the full corpus. Seconds end-to-end. Memory bound is
`|player_names| × 200 × O(1)`.

Consumer: `replay_parser.metadata.build_metadata`. The corpus driver
loads once at startup, looks up per-replay during the per-game write
loop.
"""
import sqlite3
from collections import deque
from dataclasses import dataclass

# Top-3 uses raw `position < 3` regardless of player_count. The random
# baseline rate differs by P (75% for 4p, 37.5% for 8p); downstream
# noise-floor and training-time filters consume the rate as-is. See
# `replay-parser-design.md` §5.2 / §5.3.
TOP3_CUTOFF = 3
WINDOW_SIZE = 200


@dataclass(frozen=True, slots=True)
class RollingStat:
    rolling_1st_rate: float
    rolling_top3_rate: float
    prior_games_count: int  # always >= 1 when this object exists


@dataclass(frozen=True, slots=True)
class RollingStatsResult:
    # Primary index — corpus driver consumes this per replay.
    # by_replay[replay_id][player_name] is None iff the player had zero
    # prior FFA games at the time of this replay (i.e. their first-ever
    # FFA game). Players who weren't in this replay aren't keys here.
    by_replay: dict[str, dict[str, RollingStat | None]]
    # Secondary index — supports per-player inspection in notebooks.
    # by_player[player_name] is chronologically ordered (replays in
    # started ASC order). Same None semantics as by_replay.
    by_player: dict[str, list[tuple[str, RollingStat | None]]]


def compute_rolling_stats(
    conn: sqlite3.Connection,
    player_names: list[str],
) -> RollingStatsResult:
    """Walk all FFA games once, emit rolling stats for filtered players.

    Includes pre-v15 games — the skill signal exists whether or not the
    game survives the parser filter set.

    Note on representation: this pure-Python intermediate uses `None`
    to mark "no prior data" (clearer than embedding a sentinel). The
    serialization boundary in `build_metadata` converts None to the
    documented `-1.0` / `-1` sentinels for numpy's fixed-dtype storage.
    We may unify on the sentinel representation later for consistency.
    """
    if not player_names:
        raise ValueError("player_names must be non-empty")

    placeholders = ",".join("?" * len(player_names))
    rows = conn.execute(
        f"SELECT id, name FROM players WHERE name IN ({placeholders})",
        player_names,
    ).fetchall()
    id_to_name: dict[int, str] = {pid: name for pid, name in rows}
    missing = set(player_names) - set(id_to_name.values())
    if missing:
        raise ValueError(f"unknown player names: {sorted(missing)}")
    filter_ids = set(id_to_name)

    history: dict[int, deque[int]] = {pid: deque(maxlen=WINDOW_SIZE) for pid in filter_ids}
    counts: dict[int, int] = {pid: 0 for pid in filter_ids}
    by_replay: dict[str, dict[str, RollingStat | None]] = {}
    by_player: dict[str, list[tuple[str, RollingStat | None]]] = {n: [] for n in player_names}

    cursor = conn.execute(
        """
        SELECT r.id, rp.player_id, rp.position
        FROM replays r
        JOIN replay_players rp ON rp.replay_id = r.id
        WHERE r.ladder_id = 'ffa'
        ORDER BY r.started ASC, r.id ASC, rp.position ASC
        """
    )
    for replay_id, player_id, position in cursor:
        if player_id not in filter_ids:
            continue
        name = id_to_name[player_id]
        h = history[player_id]
        n = len(h)
        if n == 0:
            stat: RollingStat | None = None
        else:
            stat = RollingStat(
                rolling_1st_rate=sum(1 for p in h if p == 0) / n,
                rolling_top3_rate=sum(1 for p in h if p < TOP3_CUTOFF) / n,
                prior_games_count=counts[player_id],
            )
        by_replay.setdefault(replay_id, {})[name] = stat
        by_player[name].append((replay_id, stat))
        h.append(position)
        counts[player_id] += 1

    return RollingStatsResult(by_replay=by_replay, by_player=by_player)
