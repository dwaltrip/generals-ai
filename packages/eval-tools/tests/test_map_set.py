"""Smoke test for the map-set pack loader: a hand-built two-map pack round-trips
through `load_bucket` into StaticMaps that match the records."""

import gzip
import json

from eval_tools.map_set import load_bucket


def _record(replay_id: str, generals: list[int]) -> dict:
    return {
        "replay_id": replay_id,
        "month": "2026-01",
        "version": 16,
        "map_width": 4,
        "map_height": 3,
        "mountains": [5],
        "initial_cities": [2],
        "initial_city_armies": [40],
        "initial_generals": generals,
        "initial_neutrals": [],
        "initial_neutral_armies": [],
    }


def test_load_bucket_roundtrip(tmp_path):
    set_dir = tmp_path / "tiny-set"
    set_dir.mkdir()
    records = [_record("aaa", [0, 11]), _record("bbb", [3, 8])]
    with gzip.open(set_dir / "maps-2p.jsonl.gz", "wt", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    loaded = load_bucket("tiny-set", 2, root=tmp_path)

    assert [rid for rid, _ in loaded] == ["aaa", "bbb"]  # canonical order kept
    rid, static = loaded[0]
    assert static.initial_generals == [0, 11]
    assert static.mountains == [5]
    assert static.usernames == ["p0", "p1"]  # synthesized, length = player count
