"""Reusable SQL fragments for ad-hoc queries.

Each helper takes alias parameters and returns a SQL string fragment to
splice into a query body via f-string interpolation.
"""


def ffa_match_filter(replays_alias: str) -> str:
    a = replays_alias
    return f"({a}.ladder_id = 'ffa' AND {a}.player_count BETWEEN 4 AND 8)"


def wire_data_filter(replays_alias: str) -> str:
    return f"{replays_alias}.wire_data IS NOT NULL"


def from_player_games(rp_alias='rp', p_alias='p', r_alias='r'):
    """Canonical FROM/JOIN block traversing replay_players + players + replays.
    Aliases default to `rp`/`p`/`r` to match every existing call site; override
    only for self-joins or alias collisions."""
    rp = rp_alias
    p = p_alias
    r = r_alias
    return (
        f"""FROM replay_players {rp}
        JOIN players {p} ON {p}.id = {rp}.player_id
        JOIN replays {r} ON {r}.id = {rp}.replay_id"""
    )
