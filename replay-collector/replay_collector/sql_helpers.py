"""Reusable WHERE-clause fragments for ad-hoc queries.

Each helper takes the alias of the table it filters and returns a single SQL
condition string (no leading `WHERE`/`AND`). Multi-condition fragments are
wrapped in parens so they compose safely under `OR`.
"""


def ffa_match_filter(replays_alias: str) -> str:
    a = replays_alias
    return f"({a}.ladder_id = 'ffa' AND {a}.player_count BETWEEN 4 AND 8)"


def wire_data_filter(replays_alias: str) -> str:
    return f"{replays_alias}.wire_data IS NOT NULL"
