"""Parse an ASCII board fixture into a `StaticMap`.

Single source of truth for small hand-authored test maps: the `.txt` file is
*read*, not mirrored in Python constants (which drift). Tokens are
whitespace-delimited, one board row per non-blank line:

    ·  (or .)   neutral plain
    #           mountain
    c           neutral city (army = `city_army`)
    g<d>        general for player slot d (0-indexed)

Player count is inferred as `max(slot) + 1`; any slot without a `g<d>` token
gets a `-1` sentinel in `initial_generals`. Whitespace alignment in the file is
cosmetic — tokenization is `str.split`, so a token's width is free to grow
later (e.g. a future `c40` for a per-city army) without touching the grid parser.
"""

from __future__ import annotations

from pathlib import Path

from game_types import StaticMap


DEFAULT_CITY_ARMY = 40
_NEUTRAL = {"·", "."}


def parse_ascii_map(text: str, *, city_army: int = DEFAULT_CITY_ARMY) -> StaticMap:
    """Parse ASCII board `text` into a `StaticMap` (see module docstring)."""
    rows = [line.split() for line in text.splitlines() if line.strip()]
    if not rows:
        raise ValueError("ASCII map has no board rows")
    height = len(rows)
    width = len(rows[0])
    for r, row in enumerate(rows):
        if len(row) != width:
            raise ValueError(
                f"row {r} has {len(row)} tokens, expected {width} (ragged board)"
            )

    mountains: list[int] = []
    cities: list[int] = []
    generals: dict[int, int] = {}
    for r, row in enumerate(rows):
        for c, tok in enumerate(row):
            cell = r * width + c
            if tok in _NEUTRAL:
                continue
            if tok == "#":
                mountains.append(cell)
            elif tok == "c":
                cities.append(cell)
            elif tok[0] == "g" and tok[1:].isdigit():
                slot = int(tok[1:])
                if slot in generals:
                    raise ValueError(
                        f"duplicate general for slot {slot} (cells "
                        f"{generals[slot]} and {cell})"
                    )
                generals[slot] = cell
            else:
                raise ValueError(f"unknown tile token {tok!r} at row {r}, col {c}")

    if not generals:
        raise ValueError("ASCII map defines no generals")
    num_players = max(generals) + 1
    initial_generals = [generals.get(s, -1) for s in range(num_players)]

    return StaticMap(
        map_width=width,
        map_height=height,
        usernames=[f"p{s}" for s in range(num_players)],
        mountains=mountains,
        initial_cities=cities,
        initial_city_armies=[city_army] * len(cities),
        initial_generals=initial_generals,
        initial_neutrals=[],
        initial_neutral_armies=[],
    )


def load_ascii_map(path: str | Path, *, city_army: int = DEFAULT_CITY_ARMY) -> StaticMap:
    """Read and parse an ASCII map fixture file into a `StaticMap`."""
    return parse_ascii_map(Path(path).read_text(encoding="utf-8"), city_army=city_army)
