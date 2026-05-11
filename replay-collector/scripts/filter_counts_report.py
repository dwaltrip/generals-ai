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
import statistics
from collections import Counter, defaultdict
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


# --- Section 2: standalone counts on survivors -------------------------------


def _quartiles(values: list[int]) -> tuple[float, float, float]:
    """(p25, p50, p75) via stdlib linear interpolation."""
    if len(values) < 2:
        v = values[0] if values else 0
        return (v, v, v)
    qs = statistics.quantiles(values, n=4, method="inclusive")
    return (qs[0], qs[1], qs[2])


def _games_without_curated(
    conn, survivor_ids: list[str], curated: list[str]
) -> dict:
    """Return {"count": int, "sample": [first 10 ids]} for survivor games whose
    ranking contains no curated-list player. Uses a temp table to keep the
    IN-clause cost off the survivor side."""
    conn.execute("DROP TABLE IF EXISTS _filter_counts_survivor_ids")
    conn.execute("CREATE TEMP TABLE _filter_counts_survivor_ids (id TEXT PRIMARY KEY)")
    conn.executemany(
        "INSERT INTO _filter_counts_survivor_ids VALUES (?)",
        [(s,) for s in survivor_ids],
    )

    if not curated:
        return {"count": len(survivor_ids), "sample": survivor_ids[:10]}

    placeholders = ",".join("?" * len(curated))
    rows = conn.execute(
        f"""
        SELECT DISTINCT s.id
        FROM _filter_counts_survivor_ids s
        JOIN replay_players rp ON rp.replay_id = s.id
        JOIN players p ON p.id = rp.player_id
        WHERE p.name IN ({placeholders})
        """,
        list(curated),
    ).fetchall()
    with_curated = {row[0] for row in rows}
    without = [sid for sid in survivor_ids if sid not in with_curated]
    return {"count": len(without), "sample": without[:10]}


def compute_section_2(survivors: list[dict], curated: list[str], conn) -> dict:
    """Aggregate the §2 standalone counts from the in-memory survivor list,
    plus the D check (games without any curated-list player) via SQL."""
    n = len(survivors)
    if n == 0:
        return {}

    # generalTrades — overall and by-version. Slot 27 is v ≥ 16 only, so
    # per-version rate is the more honest read; the overall number is
    # diluted by v15 games that can't have the field at all.
    gt_nonempty = sum(1 for s in survivors if s["general_trades_count"] > 0)
    by_version_total: Counter[int] = Counter()
    by_version_gt: Counter[int] = Counter()
    for s in survivors:
        v = s["version"]
        by_version_total[v] += 1
        if s["general_trades_count"] > 0:
            by_version_gt[v] += 1
    gt_by_version = {
        v: {
            "non_empty": by_version_gt[v],
            "total": by_version_total[v],
            "rate": round(by_version_gt[v] / by_version_total[v], 4),
        }
        for v in sorted(by_version_total)
    }

    # Map dims and aspect
    dim_hist: Counter[tuple] = Counter((s["map_w"], s["map_h"]) for s in survivors)
    aspect: Counter[str] = Counter()
    for s in survivors:
        w, h = s["map_w"], s["map_h"]
        if w is None or h is None:
            aspect["unknown"] += 1
        elif w == h:
            aspect["square"] += 1
        elif w > h:
            aspect["wide"] += 1
        else:
            aspect["tall"] += 1

    # Game length by version (half-turns, per A.F3)
    lengths_by_version: dict[int, list[int]] = defaultdict(list)
    for s in survivors:
        lengths_by_version[s["version"]].append(s["turns"])
    length_stats = {}
    for v, lengths in sorted(lengths_by_version.items()):
        p25, p50, p75 = _quartiles(sorted(lengths))
        length_stats[v] = {
            "n": len(lengths),
            "mean": round(statistics.fmean(lengths), 1),
            "p25": round(p25, 1),
            "p50": round(p50, 1),
            "p75": round(p75, 1),
        }

    # D check — survivor games with no curated player in ranking
    no_curated = _games_without_curated(conn, [s["id"] for s in survivors], curated)

    return {
        "general_trades": {
            "non_empty_overall": gt_nonempty,
            "non_empty_overall_rate": round(gt_nonempty / n, 4),
            "by_version": gt_by_version,
        },
        "map_dims_top_10": [
            {"w": w, "h": h, "n": cnt} for (w, h), cnt in dim_hist.most_common(10)
        ],
        "map_aspect": dict(aspect),
        "length_half_turns_by_version": length_stats,
        "games_without_curated_player": no_curated,
    }


# --- Section 3: perspective-level stats --------------------------------------

ROLLING_WINDOW = 100
PRIOR_GAMES_FLOOR = 50
RATE_HIST_BUCKETS = [round(i * 0.05, 2) for i in range(21)]  # 0.00, 0.05, ..., 1.00
THRESHOLD_1ST = 0.25
THRESHOLD_TOP3 = 0.375  # random baseline for 8-player FFA
BUCKET_SIZE = 50
NUM_BUCKETS = 10
BUCKET_WINDOW = BUCKET_SIZE * NUM_BUCKETS  # 500


def _curated_perspective_count_distribution(conn, curated: list[str]) -> dict:
    """Per-survivor-game count of curated players in the ranking. Survivors
    live in the temp table `_filter_counts_survivor_ids` populated by section 2."""
    placeholders = ",".join("?" * len(curated))
    rows = conn.execute(
        f"""
        SELECT s.id, COUNT(DISTINCT p.id) AS curated_count
        FROM _filter_counts_survivor_ids s
        LEFT JOIN replay_players rp ON rp.replay_id = s.id
        LEFT JOIN players p
          ON p.id = rp.player_id AND p.name IN ({placeholders})
        GROUP BY s.id
        """,
        list(curated),
    ).fetchall()
    dist: Counter[int] = Counter()
    for _, n in rows:
        dist[n if n < 4 else 4] += 1  # bucket as 0,1,2,3,4+
    total = sum(dist.values())
    mean = sum(k * v for k, v in dist.items()) / total if total else 0
    return {
        "distribution": {
            "0": dist[0],
            "1": dist[1],
            "2": dist[2],
            "3": dist[3],
            "4+": dist[4],
        },
        "mean_curated_per_game": round(mean, 3),
        "total_games": total,
    }


def _fetch_rolling_perspectives(conn, curated: list[str]) -> list[tuple]:
    """Per (curated player, FFA game) tuple, compute rolling stats. Returns
    only rows for survivor games. Window is over the player's *full* FFA
    history (any version, any player_count) — this is the skill estimate
    at game-time, not corpus membership."""
    placeholders = ",".join("?" * len(curated))
    sql = f"""
    WITH perspectives AS (
        SELECT
            rp.replay_id,
            p.name AS player_name,
            rp.position,
            r.started,
            AVG(CASE WHEN rp.position = 0 THEN 1.0 ELSE 0.0 END) OVER w AS rolling_1st_rate,
            AVG(CASE WHEN rp.position <= 2 THEN 1.0 ELSE 0.0 END) OVER w AS rolling_top3_rate,
            COUNT(*) OVER (
                PARTITION BY p.id
                ORDER BY r.started, r.id
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ) AS prior_games_count
        FROM replay_players rp
        JOIN players p ON p.id = rp.player_id
        JOIN replays r ON r.id = rp.replay_id
        WHERE p.name IN ({placeholders})
          AND r.ladder_id = 'ffa'
        WINDOW w AS (
            PARTITION BY p.id
            ORDER BY r.started, r.id
            ROWS BETWEEN {ROLLING_WINDOW} PRECEDING AND 1 PRECEDING
        )
    )
    SELECT replay_id, player_name, position, started,
           rolling_1st_rate, rolling_top3_rate, prior_games_count
    FROM perspectives
    WHERE replay_id IN (SELECT id FROM _filter_counts_survivor_ids)
    """
    return conn.execute(sql, list(curated)).fetchall()


def _bucket_rate(rate: float | None) -> str | None:
    if rate is None:
        return None
    if rate >= 1.0:
        return "1.00"
    edge = int(rate * 20) * 0.05
    return f"{edge:.2f}"


def _aggregate_perspectives(rows: list[tuple]) -> dict:
    """Build histograms + per-player summaries from the window-function output."""
    hist_1st_all: Counter[str] = Counter()
    hist_top3_all: Counter[str] = Counter()
    hist_1st_above_floor: Counter[str] = Counter()
    hist_top3_above_floor: Counter[str] = Counter()
    above_25 = 0
    below_25 = 0
    above_375_top3 = 0
    below_375_top3 = 0
    below_floor = 0
    null_window = 0  # perspective with no preceding games at all

    per_player_1st: defaultdict[str, list[float]] = defaultdict(list)
    per_player_top3: defaultdict[str, list[float]] = defaultdict(list)
    per_player_below_floor: Counter[str] = Counter()
    per_player_perspectives: Counter[str] = Counter()

    for _replay_id, name, _pos, _started, r1, r3, prior in rows:
        per_player_perspectives[name] += 1

        if r1 is None or r3 is None:
            null_window += 1
            continue

        per_player_1st[name].append(r1)
        per_player_top3[name].append(r3)

        hist_1st_all[_bucket_rate(r1)] += 1
        hist_top3_all[_bucket_rate(r3)] += 1

        if r1 >= THRESHOLD_1ST:
            above_25 += 1
        else:
            below_25 += 1
        if r3 >= THRESHOLD_TOP3:
            above_375_top3 += 1
        else:
            below_375_top3 += 1

        if prior < PRIOR_GAMES_FLOOR:
            below_floor += 1
            per_player_below_floor[name] += 1
        else:
            hist_1st_above_floor[_bucket_rate(r1)] += 1
            hist_top3_above_floor[_bucket_rate(r3)] += 1

    per_player_table = []
    for name in sorted(per_player_perspectives, key=lambda n: -per_player_perspectives[n]):
        rates_1st = per_player_1st[name]
        if not rates_1st:
            per_player_table.append({
                "name": display_name(name),
                "perspectives": per_player_perspectives[name],
                "below_floor": per_player_below_floor[name],
                "mean_1st": None,
                "p25_1st": None,
                "p50_1st": None,
                "p75_1st": None,
            })
            continue
        sorted_1st = sorted(rates_1st)
        p25, p50, p75 = _quartiles(sorted_1st)
        per_player_table.append({
            "name": display_name(name),
            "perspectives": per_player_perspectives[name],
            "below_floor": per_player_below_floor[name],
            "mean_1st": round(statistics.fmean(rates_1st), 4),
            "p25_1st": round(p25, 4),
            "p50_1st": round(p50, 4),
            "p75_1st": round(p75, 4),
        })

    return {
        "total_perspectives": sum(per_player_perspectives.values()),
        "null_window_perspectives": null_window,
        "histograms": {
            "rolling_1st_rate_all": dict(hist_1st_all),
            "rolling_1st_rate_above_floor": dict(hist_1st_above_floor),
            "rolling_top3_rate_all": dict(hist_top3_all),
            "rolling_top3_rate_above_floor": dict(hist_top3_above_floor),
        },
        "thresholds": {
            "above_25pct_1st": above_25,
            "below_25pct_1st": below_25,
            "above_375pct_top3": above_375_top3,
            "below_375pct_top3": below_375_top3,
            "below_floor": below_floor,
        },
        "per_player_table": per_player_table,
    }


def _compute_50_game_buckets(conn, curated: list[str]) -> list[dict]:
    """Per-curated-player: 10 buckets of 50 most-recent FFA games each.
    Bucket 0 = most recent. Position == 0 is winner. Matches
    `winrate_star_buckets.py` conventions."""
    rows: list[dict] = []
    for name in curated:
        cur = conn.execute(
            """
            SELECT rp.position
            FROM replay_players rp
            JOIN players p ON p.id = rp.player_id
            JOIN replays r ON r.id = rp.replay_id
            WHERE p.name = ?
              AND r.ladder_id = 'ffa'
            ORDER BY r.started DESC, r.id
            LIMIT ?
            """,
            (name, BUCKET_WINDOW),
        ).fetchall()
        positions = [row[0] for row in cur]
        row = {"name": display_name(name), "total_games": len(positions)}
        for i in range(NUM_BUCKETS):
            chunk = positions[i * BUCKET_SIZE : (i + 1) * BUCKET_SIZE]
            if chunk:
                wins = sum(1 for p in chunk if p == 0)
                row[f"b{i}_wr"] = round(wins / len(chunk), 4)
                row[f"b{i}_n"] = len(chunk)
            else:
                row[f"b{i}_wr"] = ""
                row[f"b{i}_n"] = 0
        rows.append(row)
    return rows


def compute_section_3(conn, curated: list[str]) -> dict:
    if not curated:
        return {}
    log.info("  curated count distribution per survivor game")
    counts = _curated_perspective_count_distribution(conn, curated)
    log.info("    %s mean=%s", counts["distribution"], counts["mean_curated_per_game"])

    log.info("  rolling-window perspectives (SQL window)")
    rows = _fetch_rolling_perspectives(conn, curated)
    log.info("    fetched %d curated perspectives in survivor games", len(rows))

    perspectives = _aggregate_perspectives(rows)
    log.info("    above-25%% 1st-rate: %d / below: %d / below-floor: %d",
             perspectives["thresholds"]["above_25pct_1st"],
             perspectives["thresholds"]["below_25pct_1st"],
             perspectives["thresholds"]["below_floor"])

    log.info("  per-player 50-game bucket view")
    buckets = _compute_50_game_buckets(conn, curated)

    return {
        "curated_count_per_game": counts,
        "rolling_perspectives": perspectives,
        "buckets": buckets,
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

    log.info("section 2: standalone counts on survivors")
    section2 = compute_section_2(funnel["survivors"], curated, conn)
    log.info("  generalTrades non-empty by version: %s",
             {v: d["rate"] for v, d in section2["general_trades"]["by_version"].items()})
    log.info("  map aspect: %s", section2["map_aspect"])
    log.info("  games without curated player: %d",
             section2["games_without_curated_player"]["count"])

    log.info("section 3: perspective-level stats")
    section3 = compute_section_3(conn, curated)

    # TODO: markdown rendering in next commit.
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
        "section_2": section2,
        "section_3": section3,
    }
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    out = TMP_DIR / f"filter_counts_raw-{ts}.json"
    out.write_text(json.dumps(raw_dump, indent=2))
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
