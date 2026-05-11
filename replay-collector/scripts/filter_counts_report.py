"""Filter-counts report for the replay-parser corpus.

Walks the cached `.gior` corpus and produces a report covering:

  1. Vanilla FFA Filters funnel — the §4 rules from docs/replay-parser-design.md
     applied in order against each FFA replay with wire_data present.
  2. Standalone counts on survivors — generalTrades, map aspect/dims,
     game-length-by-version, and a check for games whose ranking has no
     curated-list player in it (sourcing audit).
  3. Perspective-level stats — per-game curated-player count distribution,
     true rolling 100-prior-FFA-game 1st-rate + top-3-rate per curated
     perspective (SQL window), and a per-player 50-game-bucket secondary
     view aligned with `winrate_star_buckets.py`.

Outputs to `replay-collector/tmp/`:
  filter_counts_report-<ts>.md     — human-readable report
  filter_counts_raw-<ts>.json      — raw numbers for re-aggregation
  filter_counts_buckets-<ts>.csv   — per-curated-player 50-game buckets

Usage (from replay-collector/):
    uv run python scripts/filter_counts_report.py
"""

import datetime as dt
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

from replay_collector import wire
from replay_collector.cli._shared import TMP_DIR, load_players
from replay_collector.db import create_conn
from replay_collector.usernames import display_name


log = logging.getLogger("filter_counts")


# --- Constants ---------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data"

CURATED_LIST_FILES = [
    DATA / "leaderboards" / "leadeboard-s42-ffa-elite-gsheets.txt",
    DATA / "leaderboards" / "leadeboard-s42-ffawin-elite-gsheets.txt",
    DATA / "top-players" / "2026-05-10-new-top-players-from-wr-start-bucket-analysis.txt",
    DATA / "top-players" / "_archive" / "2026-04-30-leaderboard-ffa-top-100-combined.txt",
]

# §4 funnel rule names, in funnel order. Stage 0 is the decode pass.
FUNNEL_STAGES = [
    "decoded_cleanly",
    "player_count_4_to_8",
    "teams_null",
    "version_ge_15",
    "modifiers_empty",
    "modifier_tile_arrays_empty",
    "map_null",
    "chess_clock_null",
]

# Wire-shape slot indices we read directly (see docs/replay-format.md §Schema).
SLOT_VERSION = 0
SLOT_MAP_W = 2
SLOT_MAP_H = 3
SLOT_TEAMS = 12
SLOT_MAP = 13
SLOT_SWAMPS = 16
SLOT_MODIFIERS = 21
SLOT_OBSERVATORIES = 22
SLOT_LOOKOUTS = 23
SLOT_DESERTS = 24
SLOT_GENERAL_TRADES = 27
SLOT_TUNNELS = 28
SLOT_CHESS_CLOCK = 30
SLOT_STRONGHOLDS = 34

# Modifier-tile-array slots, used by rule 5 (defensive cross-check).
MODIFIER_TILE_SLOTS = {
    "swamps": SLOT_SWAMPS,
    "observatories": SLOT_OBSERVATORIES,
    "lookouts": SLOT_LOOKOUTS,
    "deserts": SLOT_DESERTS,
    "tunnels": SLOT_TUNNELS,
    "strongholds": SLOT_STRONGHOLDS,
}


def _slot(wire_list: list, idx: int, default=None):
    """Safe slot access — newer slots may not exist in older-version wire arrays."""
    return wire_list[idx] if idx < len(wire_list) else default


# --- Vanilla FFA Filters funnel ----------------------------------------------


def evaluate_row(
    replay_id: str,
    db_version: int | None,
    db_player_count: int,
    db_turns: int,
    wire_blob: bytes,
) -> tuple[str | None, dict]:
    """Run a single replay through the funnel. Returns (drop_stage, info).

    `drop_stage` is None if the replay survives, otherwise the FUNNEL_STAGES
    entry that dropped it. `info` carries per-rule diagnostics (e.g., which
    modifier tile arrays were non-empty for rule 5) plus per-survivor
    metadata used by section 2 if the replay survives."""
    info: dict = {}

    # Stage 0: decode
    try:
        w = wire.decode(wire_blob)
    except Exception as e:
        info["decode_error"] = repr(e)
        return "decoded_cleanly", info

    # Rule 1: player_count ∈ [4, 8]
    if not (4 <= db_player_count <= 8):
        info["player_count"] = db_player_count
        return "player_count_4_to_8", info

    # Rule 2: teams slot null (FFA)
    if _slot(w, SLOT_TEAMS) is not None:
        return "teams_null", info

    # Rule 3: version ≥ 15
    if db_version is None or db_version < 15:
        info["version"] = db_version
        return "version_ge_15", info

    # Rule 4: modifiers slot empty
    modifiers = _slot(w, SLOT_MODIFIERS, [])
    if modifiers:
        info["modifiers"] = list(modifiers)
        return "modifiers_empty", info

    # Rule 5: all modifier tile arrays empty (defensive cross-check)
    nonempty_arrays = {
        name: len(_slot(w, idx, []) or [])
        for name, idx in MODIFIER_TILE_SLOTS.items()
        if _slot(w, idx, []) or []  # truthy = non-empty list
    }
    if nonempty_arrays:
        info["nonempty_tile_arrays"] = nonempty_arrays
        return "modifier_tile_arrays_empty", info

    # Rule 6: custom map slot null
    if _slot(w, SLOT_MAP) is not None:
        return "map_null", info

    # Rule 7: chess clock slot null
    if _slot(w, SLOT_CHESS_CLOCK) is not None:
        return "chess_clock_null", info

    # Survivor — capture metadata for section 2.
    general_trades = _slot(w, SLOT_GENERAL_TRADES, []) or []
    info["survivor"] = {
        "id": replay_id,
        "version": db_version,
        "map_w": _slot(w, SLOT_MAP_W),
        "map_h": _slot(w, SLOT_MAP_H),
        "player_count": db_player_count,
        "turns": db_turns,
        "general_trades_count": len(general_trades),
    }
    return None, info


def run_funnel(conn) -> dict:
    """Single sequential pass over all FFA + wire_data rows. Returns the
    aggregated funnel result + survivor list."""
    drops = {stage: 0 for stage in FUNNEL_STAGES}
    rule5_breakdown: Counter[str] = Counter()
    survivors: list[dict] = []
    decode_errors: list[tuple[str, str]] = []

    total = conn.execute(
        "SELECT COUNT(*) FROM replays WHERE ladder_id='ffa' AND wire_data IS NOT NULL"
    ).fetchone()[0]
    log.info("funnel: %d rows to scan", total)

    cur = conn.execute(
        """
        SELECT id, version, player_count, turns, wire_data
        FROM replays
        WHERE ladder_id = 'ffa' AND wire_data IS NOT NULL
        ORDER BY started
        """
    )

    t0 = time.monotonic()
    scanned = 0
    for replay_id, version, player_count, turns, wire_blob in cur:
        drop, info = evaluate_row(replay_id, version, player_count, turns, wire_blob)
        if drop == "decoded_cleanly":
            drops[drop] += 1
            decode_errors.append((replay_id, info.get("decode_error", "")))
        elif drop is not None:
            drops[drop] += 1
            if drop == "modifier_tile_arrays_empty":
                for arr_name in info.get("nonempty_tile_arrays", {}):
                    rule5_breakdown[arr_name] += 1
        else:
            survivors.append(info["survivor"])
        scanned += 1
        if scanned % 20000 == 0:
            elapsed = time.monotonic() - t0
            log.info("  funnel progress: %d/%d (%.1fs, %.0f rows/s)",
                     scanned, total, elapsed, scanned / max(elapsed, 1e-6))
    log.info("funnel: scanned %d in %.1fs, %d survivors", scanned, time.monotonic() - t0, len(survivors))
    return {
        "drops": drops,
        "rule5_per_array_breakdown": dict(rule5_breakdown),
        "decode_errors": decode_errors[:20],  # keep a sample, not all
        "decode_error_count": len(decode_errors),
        "survivors": survivors,
        "scanned": scanned,
    }


# --- Curated list loader -----------------------------------------------------


def load_curated_union() -> tuple[list[str], dict]:
    """Union the 4 curated-list files (one name per line), drop invalid
    usernames via `filter_valid`, return (sorted unique names, per-file diag).

    Per-file diag = {file_path: {"raw": int, "valid": int}} so the report can
    surface where each name came from."""
    union: set[str] = set()
    per_file: dict[str, dict[str, int]] = {}
    for path in CURATED_LIST_FILES:
        if not path.exists():
            log.warning("curated list missing: %s", path)
            per_file[str(path.relative_to(REPO_ROOT))] = {"raw": 0, "valid": 0}
            continue
        valid = load_players(path)  # drops invalid + warns
        # Re-count raw lines for the diag (load_players already dropped invalid).
        raw_lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        per_file[str(path.relative_to(REPO_ROOT))] = {
            "raw": len(raw_lines),
            "valid": len(valid),
        }
        union.update(valid)
    return sorted(union), per_file


# --- Corpus at a glance ------------------------------------------------------


def corpus_at_a_glance(conn) -> dict:
    """Counts that don't require wire-decoding. Cheap SQL pass."""
    total_listings = conn.execute(
        "SELECT COUNT(*) FROM replays WHERE ladder_id = 'ffa'"
    ).fetchone()[0]
    with_wire = conn.execute(
        "SELECT COUNT(*) FROM replays WHERE ladder_id = 'ffa' AND wire_data IS NOT NULL"
    ).fetchone()[0]
    non_ffa = conn.execute(
        "SELECT COUNT(*) FROM replays WHERE ladder_id IS NOT 'ffa' OR ladder_id IS NULL"
    ).fetchone()[0]
    version_hist = dict(conn.execute(
        """
        SELECT version, COUNT(*)
        FROM replays
        WHERE ladder_id = 'ffa' AND wire_data IS NOT NULL
        GROUP BY version
        ORDER BY version
        """
    ).fetchall())
    return {
        "ffa_listings_total": total_listings,
        "ffa_with_wire_data": with_wire,
        "non_ffa_listings": non_ffa,
        "version_histogram_pre_funnel": version_hist,
    }


# --- Main --------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        conn = create_conn()
    except FileNotFoundError as e:
        sys.exit(str(e))

    log.info("loading curated list")
    curated, per_file_diag = load_curated_union()
    log.info("curated union: %d unique names across %d files",
             len(curated), len(CURATED_LIST_FILES))

    log.info("corpus at a glance")
    glance = corpus_at_a_glance(conn)
    for k, v in glance.items():
        log.info("  %s = %s", k, v)

    log.info("running funnel")
    funnel = run_funnel(conn)
    for stage, n in funnel["drops"].items():
        log.info("  drop[%s] = %d", stage, n)
    if funnel["rule5_per_array_breakdown"]:
        log.info("  rule 5 per-array breakdown: %s", funnel["rule5_per_array_breakdown"])

    # TODO: section 2 + section 3 + markdown rendering in subsequent commits.
    raw_dump = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "curated_list": {
            "files": per_file_diag,
            "unique_valid_names": len(curated),
        },
        "corpus_at_a_glance": glance,
        "funnel": {
            "drops": funnel["drops"],
            "rule5_per_array_breakdown": funnel["rule5_per_array_breakdown"],
            "decode_error_count": funnel["decode_error_count"],
            "decode_error_sample": funnel["decode_errors"],
            "scanned": funnel["scanned"],
            "survivor_count": len(funnel["survivors"]),
        },
    }
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    out = TMP_DIR / f"filter_counts_raw-{ts}.json"
    out.write_text(json.dumps(raw_dump, indent=2))
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
