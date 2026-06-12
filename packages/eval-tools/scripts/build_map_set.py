#!/usr/bin/env -S uv run python
"""Build a frozen eval map set: a git-tracked manifest + per-bucket map packs.

Selects replays from the collector DB that are eligible as eval maps — never
referenced by any training-split manifest, vanilla FFA (no modifier/special
tiles), v15+ — stratified by month so the set covers its collection window
evenly. Each selected replay's `StaticMap` is extracted into a gzipped-JSONL
pack file per player-count bucket.

The manifest (replay ids + provenance) is the durable artifact: it versions the
set and doubles as the exclusion list future training-split builds subtract.
Packs are derived — regenerable from manifest + DB — and immutable once
published: a new set is a new name, never an edit (re-running without --force
refuses to touch an existing set dir).

Selection is deterministic for a given (DB state, splits dir, seed). Candidate
order is fixed (SQL ORDER BY id), per-month shuffles use a per-bucket seeded
RNG, and acceptance walks months round-robin in chronological order — so the
pack's line order (the set's canonical order) interleaves months, and any
prefix of it is itself month-balanced.

Usage:
    build_map_set.py                          # eval-map-set-v1, 2000/bucket
    build_map_set.py --name my-set --target 500 --seed 7
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import datetime as dt
import gzip
import io
import json
from pathlib import Path
import random
import sqlite3
import statistics

# Wire-level helpers shared with the parser. `_shared` / `_collector` are
# parser-private modules; fine for an in-repo tool, and `is_vanilla_ffa` is the
# single source of truth for the vanilla filter (same check the training-corpus
# pipeline applies).
from replay_parser._collector.wire import decode as decode_wire_blob
from replay_parser._shared import is_vanilla_ffa
from replay_parser.decode import decode_wire_array
from settings import DB_PATH, EVAL_MAP_SETS_DIR, PROJECT_ROOT
from utils.json_io import write_json


SPLITS_DIR = PROJECT_ROOT / "data" / "training" / "splits"

PLAYER_COUNTS = range(2, 9)
MIN_VERSION = 15


@dataclass
class Candidate:
    replay_id: str
    month: str  # "YYYY-MM", UTC, from the replay's `started` timestamp
    version: int


@dataclass
class SelectedMap:
    cand: Candidate
    static: object  # game_types.StaticMap


def load_used_ids(splits_dir: Path) -> set[str]:
    """Union of replay ids over every training-split manifest (train + val).

    Val rows never receive gradients, but excluding them too is free and keeps
    the eval set disjoint from everything training has ever touched.
    """
    used: set[str] = set()
    for path in sorted(splits_dir.glob("*.json")):
        if path.name.endswith(".metrics.json"):
            continue
        manifest = json.loads(path.read_text())
        for key in ("train", "val"):
            used.update(pair[0] for pair in manifest.get(key, []))
    return used


def month_of(started_ms: int) -> str:
    return dt.datetime.fromtimestamp(started_ms / 1000, tz=dt.UTC).strftime("%Y-%m")


def bucket_candidates(
    conn: sqlite3.Connection, num_players: int, used: set[str],
) -> dict[str, list[Candidate]]:
    """Eligible-by-metadata candidates for one bucket, grouped by month.

    Metadata eligibility only (version, never-trained); the vanilla check needs
    the wire blob and runs lazily during selection.
    """
    rows = conn.execute(
        "SELECT id, started, version FROM replays"
        " WHERE player_count = ? AND wire_data IS NOT NULL AND version >= ?"
        " ORDER BY id",
        (num_players, MIN_VERSION),
    ).fetchall()
    by_month: dict[str, list[Candidate]] = defaultdict(list)
    for replay_id, started, version in rows:
        if replay_id in used:
            continue
        cand = Candidate(replay_id, month_of(started), version)
        by_month[cand.month].append(cand)
    return by_month


def try_load_map(conn: sqlite3.Connection, cand: Candidate, num_players: int,
                 rejects: dict[str, int]):
    """Decode one candidate; return its StaticMap or None (tallying the reason)."""
    blob = conn.execute(
        "SELECT wire_data FROM replays WHERE id = ?", (cand.replay_id,)
    ).fetchone()[0]
    try:
        wire = decode_wire_blob(blob)
        if not is_vanilla_ffa(wire):
            rejects["non_vanilla"] += 1
            return None
        static = decode_wire_array(wire).static.map
    except Exception:
        rejects["decode_error"] += 1
        return None
    gens = static.initial_generals
    if len(gens) != num_players or min(gens) < 0:
        # Null general slots or a player-count mismatch vs the DB column.
        rejects["bad_generals"] += 1
        return None
    return static


def select_bucket(
    conn: sqlite3.Connection,
    num_players: int,
    used: set[str],
    target: int,
    seed: int,
) -> tuple[list[SelectedMap], dict[str, int]]:
    """Pick up to `target` vanilla maps, round-robin over months.

    Each round accepts at most one map per month; exhausted months drop out and
    later rounds keep filling from the rest — i.e. monthly stratification with
    shortfall automatically redistributed. Take-all buckets (pool < target)
    degenerate to "every candidate that passes the filters".
    """
    by_month = bucket_candidates(conn, num_players, used)
    rng = random.Random(f"{seed}:{num_players}p")
    for cands in by_month.values():
        rng.shuffle(cands)
    iters = {m: iter(by_month[m]) for m in sorted(by_month)}

    selected: list[SelectedMap] = []
    rejects: dict[str, int] = defaultdict(int)
    while iters and len(selected) < target:
        for month in list(iters):
            if len(selected) >= target:
                break
            # Pull candidates from this month until one is accepted or it runs dry.
            for cand in iters[month]:
                static = try_load_map(conn, cand, num_players, rejects)
                if static is not None:
                    selected.append(SelectedMap(cand, static))
                    break
            else:
                del iters[month]
    return selected, dict(rejects)


def pack_line(sel: SelectedMap) -> str:
    """One pack record: id + provenance + the StaticMap fields.

    `usernames` is deliberately not stored — it carries no gameplay content
    (player count is pinned by the bucket) — and is synthesized at load time.
    """
    s = sel.static
    return json.dumps({
        "replay_id": sel.cand.replay_id,
        "month": sel.cand.month,
        "version": sel.cand.version,
        "map_width": s.map_width,
        "map_height": s.map_height,
        "mountains": s.mountains,
        "initial_cities": s.initial_cities,
        "initial_city_armies": s.initial_city_armies,
        "initial_generals": s.initial_generals,
        "initial_neutrals": s.initial_neutrals,
        "initial_neutral_armies": s.initial_neutral_armies,
    }, separators=(",", ":"))


def write_pack(path: Path, selected: list[SelectedMap]) -> None:
    # mtime=0 keeps the gzip byte-identical across rebuilds of the same set.
    with gzip.GzipFile(path, "wb", mtime=0) as gz:
        with io.TextIOWrapper(gz, encoding="utf-8") as out:
            for sel in selected:
                out.write(pack_line(sel) + "\n")


def print_bucket_stats(num_players: int, selected: list[SelectedMap],
                       rejects: dict[str, int]) -> None:
    months = sorted({s.cand.month for s in selected})
    per_month = defaultdict(int)
    for s in selected:
        per_month[s.cand.month] += 1
    cells = [s.static.map_width * s.static.map_height for s in selected]
    n_mountains = [len(s.static.mountains) for s in selected]
    n_cities = [len(s.static.initial_cities) for s in selected]

    print(f"\n{num_players}p: {len(selected)} maps")
    print(f"  window: {months[0]} .. {months[-1]} ({len(months)} months, "
          f"{min(per_month.values())}-{max(per_month.values())} maps/month)")
    print(f"  board cells: min={min(cells)} median={statistics.median(cells):.0f} "
          f"max={max(cells)}")
    print(f"  mountains: median={statistics.median(n_mountains):.0f}   "
          f"cities: median={statistics.median(n_cities):.0f}")
    if rejects:
        drops = ", ".join(f"{k}={v}" for k, v in sorted(rejects.items()))
        print(f"  rejected while sampling: {drops}")


def build_manifest(name: str, seed: int, target: int,
                   buckets: dict[int, list[SelectedMap]]) -> dict:
    all_months = sorted({s.cand.month for sel in buckets.values() for s in sel})
    return {
        "name": name,
        "built_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "seed": seed,
        "target_per_bucket": target,
        "eligibility": {
            "min_version": MIN_VERSION,
            "vanilla": "replay_parser._shared.is_vanilla_ffa",
            "never_trained": "id not in any data/training/splits manifest at build time",
        },
        "collection_window": {"first_month": all_months[0], "last_month": all_months[-1]},
        "entry_format": ["replay_id", "month", "version"],
        "buckets": {
            str(pc): [[s.cand.replay_id, s.cand.month, s.cand.version] for s in sel]
            for pc, sel in buckets.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="eval-map-set-v1")
    parser.add_argument("--target", type=int, default=2000,
                        help="maps per player-count bucket (default 2000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing set dir (sets are otherwise immutable)")
    args = parser.parse_args()

    out_dir = EVAL_MAP_SETS_DIR / args.name
    if (out_dir / "manifest.json").exists() and not args.force:
        raise SystemExit(
            f"{out_dir} already exists — published sets are immutable. "
            f"Pick a new --name, or --force to rebuild during development."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    used = load_used_ids(SPLITS_DIR)
    print(f"excluding {len(used)} ever-trained replay ids "
          f"(union over {SPLITS_DIR.relative_to(PROJECT_ROOT)})")

    conn = sqlite3.connect(DB_PATH)
    buckets: dict[int, list[SelectedMap]] = {}
    for pc in PLAYER_COUNTS:
        selected, rejects = select_bucket(conn, pc, used, args.target, args.seed)
        buckets[pc] = selected
        print_bucket_stats(pc, selected, rejects)
        write_pack(out_dir / f"maps-{pc}p.jsonl.gz", selected)

    manifest = build_manifest(args.name, args.seed, args.target, buckets)
    write_json(out_dir / "manifest.json", manifest)

    total = sum(len(sel) for sel in buckets.values())
    size_kb = sum(f.stat().st_size for f in out_dir.glob("maps-*.jsonl.gz")) / 1024
    print(f"\n{args.name}: {total} maps, packs {size_kb:.0f} KiB -> {out_dir}")


if __name__ == "__main__":
    main()
