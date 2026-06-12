"""Loader for frozen eval map sets (the manifest + pack dirs that
`scripts/build_map_set.py` produces).

A set directory holds one `maps-<N>p.jsonl.gz` pack per player-count bucket —
one StaticMap record per line, in the set's canonical (frozen) order — plus a
`manifest.json` copy describing provenance. Loading is stdlib gzip+json with no
DB or parser dependencies, so it behaves identically whether the set lives in
the local data dir or on a mounted cloud volume.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from game_types import StaticMap
from settings import EVAL_MAP_SETS_DIR


# Default local home for map sets; cloud callers pass the volume mount instead.
MAP_SETS_ROOT = EVAL_MAP_SETS_DIR


def static_from_record(rec: dict) -> StaticMap:
    """Rebuild a StaticMap from one pack line. `usernames` carries no gameplay
    content and isn't stored in the pack — synthesized here (it only needs the
    right length, which sets the player count)."""
    return StaticMap(
        map_width=rec["map_width"],
        map_height=rec["map_height"],
        usernames=[f"p{i}" for i in range(len(rec["initial_generals"]))],
        mountains=rec["mountains"],
        initial_cities=rec["initial_cities"],
        initial_city_armies=rec["initial_city_armies"],
        initial_generals=rec["initial_generals"],
        initial_neutrals=rec["initial_neutrals"],
        initial_neutral_armies=rec["initial_neutral_armies"],
    )


def load_bucket(
    set_name: str, num_players: int, root: Path = MAP_SETS_ROOT,
) -> list[tuple[str, StaticMap]]:
    """All maps in one player-count bucket, as (replay_id, StaticMap), in the
    set's canonical order. Raises FileNotFoundError with the resolved path if
    the set or bucket doesn't exist."""
    path = root / set_name / f"maps-{num_players}p.jsonl.gz"
    if not path.exists():
        raise FileNotFoundError(
            f"map set '{set_name}' has no {num_players}p bucket at {path}"
        )
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [
            (rec["replay_id"], static_from_record(rec))
            for rec in map(json.loads, f)
        ]
