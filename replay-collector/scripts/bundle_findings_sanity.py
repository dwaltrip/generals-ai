"""Quick empirical sanity checks for the JS-bundle reading findings.

Reads a sample of cached FFA replays (matching the §4 corpus filters) and
produces three checks against the claims in docs/2026-05/5.11-2-bundle-reading.md:

  A. AFK pairing structure + paired-event gap distribution.
     Predicted: per-player AFK event counts are heavily {0, 2}; paired-event
     gaps are bimodal at 1 (disconnect/leave) and 50 (regular surrender at
     game_speed=1).

  B. Game-speed distribution. Predicted: speed=1 for all FFA ladder games.

  C. Null-general occurrence rate. Predicted: rare; want to know the actual
     rate so we know whether to treat as a first-class case in the simulator
     setup.

Outputs to `replay-collector/tmp/`:
  bundle_findings_sanity-<ts>.md     — human-readable report

Usage (from replay-collector/):
    uv run python scripts/bundle_findings_sanity.py [--sample N]
"""

import argparse
from collections import Counter, defaultdict
import datetime as dt
import logging
import time

from replay_collector import wire
from replay_collector.cli._shared import TMP_DIR
from replay_collector.db import create_conn


log = logging.getLogger("bundle_sanity")


SLOT_GENERALS = 8
SLOT_AFKS = 11
SLOT_TEAMS = 12
SLOT_MAP = 13
SLOT_SETTINGS = 20
SLOT_MODIFIERS = 21
SLOT_CHESS_CLOCK = 30

MODIFIER_TILE_SLOTS = [16, 22, 23, 24, 28, 34]  # swamps, observatories, lookouts, deserts, tunnels, strongholds

DEFAULT_SAMPLE = 5000


def _slot(w, idx, default=None):
    return w[idx] if idx < len(w) else default


def _passes_vanilla_ffa(w, player_count: int) -> bool:
    """Replicates the §4 corpus filters that aren't already in the SQL pre-filter."""
    if not (4 <= player_count <= 8):
        return False
    if _slot(w, SLOT_TEAMS) is not None:
        return False
    if _slot(w, SLOT_MODIFIERS, []) or []:
        return False
    if any((_slot(w, idx, []) or []) for idx in MODIFIER_TILE_SLOTS):
        return False
    if _slot(w, SLOT_MAP) is not None:
        return False
    if _slot(w, SLOT_CHESS_CLOCK) is not None:
        return False
    return True


def analyze_replay(w):
    """Extract the three metric inputs from one decoded wire array.

    Returns a dict with:
      speed: float (slot 20[0])
      n_null_generals: int (# of null entries in `generals`)
      n_generals: int (length of `generals` array)
      afk_events_by_player: {player_idx: [turn, turn, ...]} sorted by turn
    """
    settings = _slot(w, SLOT_SETTINGS, []) or []
    speed = settings[0] if settings else None

    generals = _slot(w, SLOT_GENERALS, []) or []
    n_null_generals = sum(1 for g in generals if g is None)

    afks = _slot(w, SLOT_AFKS, []) or []
    # Wire shape per row: [player_index, turn]. Group by player, preserve order.
    afk_by_player: dict = defaultdict(list)
    for row in afks:
        player_idx, turn = row[0], row[1]
        afk_by_player[player_idx].append(turn)

    return {
        "speed": speed,
        "n_null_generals": n_null_generals,
        "n_generals": len(generals),
        "afk_events_by_player": dict(afk_by_player),
    }


def aggregate(rows_iter, max_rows: int):
    """Walk SQL rows, decode, accumulate aggregate counters."""
    speed_counter: Counter = Counter()
    null_general_games = 0
    total_null_general_slots = 0
    total_general_slots = 0

    # AFK pairing
    afk_event_count_dist: Counter = Counter()  # how many events does a (game, player) have?
    paired_gap_counter: Counter = Counter()    # gap value -> count
    paired_gap_buckets = [0, 1, 2, 5, 10, 25, 49, 50, 51, 75, 100, 200, 500, 1000]
    paired_gap_examples_by_bucket: dict = defaultdict(list)
    unpaired_examples: list = []   # (replay_id, player_idx, [turns])
    triple_plus_examples: list = []

    n_considered = 0
    n_decode_fail = 0
    n_filter_drop = 0
    n_analyzed = 0

    for row in rows_iter:
        if n_analyzed >= max_rows:
            break
        n_considered += 1
        replay_id, _version, player_count, wire_blob = row

        try:
            w = wire.decode(wire_blob)
        except Exception:
            n_decode_fail += 1
            continue

        if not _passes_vanilla_ffa(w, player_count):
            n_filter_drop += 1
            continue

        info = analyze_replay(w)
        n_analyzed += 1

        if info["speed"] is not None:
            speed_counter[info["speed"]] += 1

        if info["n_null_generals"] > 0:
            null_general_games += 1
        total_null_general_slots += info["n_null_generals"]
        total_general_slots += info["n_generals"]

        for player_idx, turns in info["afk_events_by_player"].items():
            afk_event_count_dist[len(turns)] += 1
            if len(turns) == 2:
                gap = turns[1] - turns[0]
                paired_gap_counter[gap] += 1
                bucket = _bucket(gap, paired_gap_buckets)
                if len(paired_gap_examples_by_bucket[bucket]) < 3:
                    paired_gap_examples_by_bucket[bucket].append(
                        (replay_id, player_idx, turns)
                    )
            elif len(turns) == 1:
                if len(unpaired_examples) < 25:
                    unpaired_examples.append((replay_id, player_idx, turns))
            elif len(turns) >= 3:
                if len(triple_plus_examples) < 25:
                    triple_plus_examples.append((replay_id, player_idx, turns))

    return {
        "n_considered": n_considered,
        "n_decode_fail": n_decode_fail,
        "n_filter_drop": n_filter_drop,
        "n_analyzed": n_analyzed,
        "speed_counter": speed_counter,
        "null_general_games": null_general_games,
        "total_null_general_slots": total_null_general_slots,
        "total_general_slots": total_general_slots,
        "afk_event_count_dist": afk_event_count_dist,
        "paired_gap_counter": paired_gap_counter,
        "paired_gap_examples_by_bucket": dict(paired_gap_examples_by_bucket),
        "unpaired_examples": unpaired_examples,
        "triple_plus_examples": triple_plus_examples,
    }


def _bucket(value: int, edges: list) -> str:
    """Return a label string for the bucket the value falls into."""
    for e in edges:
        if value == e:
            return f"={e}"
    for i in range(len(edges) - 1):
        if edges[i] < value < edges[i + 1]:
            return f"{edges[i] + 1}-{edges[i + 1] - 1}"
    return f">{edges[-1]}"


def render_report(agg: dict, sample_size: int, elapsed_s: float) -> str:
    out: list = []
    p = out.append

    p("# Bundle-Findings Sanity Report")
    p("")
    p(f"**Generated:** {dt.datetime.now().isoformat(timespec='seconds')}")
    p(f"**Sample requested:** {sample_size:,}  |  **analyzed:** {agg['n_analyzed']:,}  |  **runtime:** {elapsed_s:.1f}s")
    p(f"**Considered:** {agg['n_considered']:,}  |  **decode failures:** {agg['n_decode_fail']:,}  |  **non-vanilla drops:** {agg['n_filter_drop']:,}")
    p("")
    p("Validates the claims in `docs/2026-05/5.11-2-bundle-reading.md` against actual replay data.")
    p("")

    # --- B: Game-speed -------------------------------------------------------
    p("## B. Game-speed distribution")
    p("")
    p("**Claim:** all FFA ladder games are speed=1.0.")
    p("")
    if agg["speed_counter"]:
        p("| Speed | Count | % |")
        p("|---|---|---|")
        total = sum(agg["speed_counter"].values())
        for speed, count in sorted(agg["speed_counter"].items()):
            pct = 100.0 * count / total
            p(f"| {speed} | {count:,} | {pct:.2f}% |")
    else:
        p("(no data)")
    p("")

    # --- C: Null generals ----------------------------------------------------
    p("## C. Null-general occurrence")
    p("")
    p("**Claim:** rare; players occasionally have null general slots (never connected).")
    p("")
    p(f"- Games with ≥1 null general: **{agg['null_general_games']:,}** / {agg['n_analyzed']:,}  "
      f"({100.0 * agg['null_general_games'] / max(agg['n_analyzed'], 1):.3f}%)")
    p(f"- Total null general slots: **{agg['total_null_general_slots']:,}** / {agg['total_general_slots']:,}  "
      f"({100.0 * agg['total_null_general_slots'] / max(agg['total_general_slots'], 1):.3f}%)")
    p("")

    # --- A: AFK pairing ------------------------------------------------------
    p("## A. AFK pairing structure")
    p("")
    p("**Claim:** every AFK'd player gets exactly 2 events (kill + neutralize). "
      "Edge case: games that end before neutralization fires might leave a player with just 1.")
    p("")
    p("### Per-(game, player) AFK event count distribution")
    p("")
    p("| #events | Count | % of (game, player) pairs with ≥1 event |")
    p("|---|---|---|")
    total_pairs = sum(agg["afk_event_count_dist"].values())
    for n_events, count in sorted(agg["afk_event_count_dist"].items()):
        pct = 100.0 * count / max(total_pairs, 1)
        p(f"| {n_events} | {count:,} | {pct:.2f}% |")
    p("")

    if agg["unpaired_examples"]:
        p("**Unpaired (1-event) examples** (up to 25 shown):")
        p("")
        p("| replay_id | player_idx | turn |")
        p("|---|---|---|")
        for replay_id, player_idx, turns in agg["unpaired_examples"][:25]:
            p(f"| `{replay_id}` | {player_idx} | {turns[0]} |")
        p("")

    if agg["triple_plus_examples"]:
        p("**3+ event examples** (up to 25 shown — would be surprising):")
        p("")
        p("| replay_id | player_idx | turns |")
        p("|---|---|---|")
        for replay_id, player_idx, turns in agg["triple_plus_examples"][:25]:
            p(f"| `{replay_id}` | {player_idx} | `{turns}` |")
        p("")

    # --- Gap histogram -------------------------------------------------------
    p("### Paired-event gap distribution")
    p("")
    p("**Claim:** bimodal at 1 (disconnect/leave) and 50 (regular surrender at speed=1).")
    p("")
    p("Exact-value counts for the predicted peaks + nearby values:")
    p("")
    p("| Gap (timesteps) | Count | % of paired events |")
    p("|---|---|---|")
    total_paired = sum(agg["paired_gap_counter"].values())
    interest_gaps = sorted(set([1, 2, 3, 49, 50, 51, 99, 100, 101]))
    shown_zero_gaps = []
    for g in interest_gaps:
        c = agg["paired_gap_counter"].get(g, 0)
        pct = 100.0 * c / max(total_paired, 1)
        p(f"| {g} | {c:,} | {pct:.2f}% |")
        if c == 0:
            shown_zero_gaps.append(g)
    p("")
    p("Coarse bucket view (every paired-event gap):")
    p("")
    p("| Gap range | Count | % |")
    p("|---|---|---|")
    buckets_ordered = [
        ("=1", 1, 1), ("=2-4", 2, 4), ("=5-9", 5, 9), ("=10-24", 10, 24),
        ("=25-49", 25, 49), ("=50", 50, 50), ("=51-74", 51, 74),
        ("=75-99", 75, 99), ("=100-199", 100, 199), ("=200-499", 200, 499),
        ("=500-999", 500, 999), (">=1000", 1000, 10_000_000),
    ]
    for label, lo, hi in buckets_ordered:
        c = sum(cnt for g, cnt in agg["paired_gap_counter"].items() if lo <= g <= hi)
        pct = 100.0 * c / max(total_paired, 1)
        p(f"| {label} | {c:,} | {pct:.2f}% |")
    p("")
    p(f"**Total paired events:** {total_paired:,}")
    p("")

    if agg["paired_gap_examples_by_bucket"]:
        # Spot-check non-{1, 50} buckets for sanity
        p("### Sample examples by gap bucket")
        p("")
        for bucket, examples in sorted(agg["paired_gap_examples_by_bucket"].items()):
            if not examples:
                continue
            p(f"- **gap {bucket}:** " + ", ".join(
                f"`{rid}` (p{pidx}: {turns})" for rid, pidx, turns in examples[:2]
            ))
        p("")

    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE,
                        help=f"Max replays to analyze (default {DEFAULT_SAMPLE}).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    log.info("Opening DB connection (read-only).")
    conn = create_conn()
    cur = conn.cursor()

    # SQL pre-filter: FFA + wire_data present + version ≥ 15. Post-decode
    # filters in _passes_vanilla_ffa handle the wire-only filter rules.
    # Sample is ORDER BY id LIMIT N for determinism + sequential read speed.
    log.info("Streaming up to %d replays.", args.sample * 3)  # over-fetch for filter loss
    cur.execute("""
        SELECT id, version, player_count, wire_data
        FROM replays
        WHERE ladder_id = 'ffa' AND wire_data IS NOT NULL AND version >= 15
        ORDER BY id
        LIMIT ?
    """, (args.sample * 3,))

    t0 = time.time()
    agg = aggregate(cur, max_rows=args.sample)
    elapsed = time.time() - t0

    log.info("Analyzed %d / considered %d (%.1fs).", agg["n_analyzed"], agg["n_considered"], elapsed)

    report = render_report(agg, args.sample, elapsed)
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = TMP_DIR / f"bundle_findings_sanity-{ts}.md"
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    log.info("Wrote report: %s", out_path)

    # Brief stdout summary
    print(f"\nAnalyzed: {agg['n_analyzed']:,} replays in {elapsed:.1f}s")
    print(f"\nSpeed: {dict(agg['speed_counter'])}")
    null_pct = 100.0 * agg["total_null_general_slots"] / max(agg["total_general_slots"], 1)
    print(f"Null-general rate: {agg['total_null_general_slots']:,} / "
          f"{agg['total_general_slots']:,} slots ({null_pct:.3f}%)")
    print(f"AFK event-count distribution: {dict(agg['afk_event_count_dist'])}")
    gap_top = sorted(agg["paired_gap_counter"].items(), key=lambda kv: -kv[1])[:5]
    print(f"Top paired-gap values: {gap_top}")
    print(f"\nFull report: {out_path}")


if __name__ == "__main__":
    main()
