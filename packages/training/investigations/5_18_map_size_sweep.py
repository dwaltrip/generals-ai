"""Map-size sweep — for every parsed game in `intermediate/`, look up
`map_width × map_height` from the sqlite DB and report the distribution.
Settles the padding policy for >32-side games (Phase 2 step 5 in the plan).

Plan thresholds for >32-side fraction:
  <1%:  drop them at dataloader time.
  1-10%: pad to 36x36 instead of 32x32.
  >10%: pad to actual max + variable masking.
"""

from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
import sqlite3
import time

from utils.distribution import print_counts_with_percents

from _helpers import DB_PATH, iter_intermediate_files, replay_id_from_path


def load_map_dims_and_started(db_path: Path) -> dict[str, tuple[int, int, int]]:
    """{replay_id: (map_width, map_height, started_ms)} for every replays row."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT id, map_width, map_height, started FROM replays"
        ).fetchall()
    finally:
        conn.close()
    return {rid: (w, h, ms) for rid, w, h, ms in rows if w is not None and h is not None}


def main() -> None:
    print("Loading map dims + started from sqlite ...")
    t0 = time.perf_counter()
    db_info = load_map_dims_and_started(DB_PATH)
    print(f"  loaded {len(db_info)} rows in {time.perf_counter() - t0:.2f}s")

    print("Walking intermediate/ for parsed replay IDs ...")
    t0 = time.perf_counter()

    # `iter_intermediate_files` is the load-bearing filter for this sweep:
    # we never open the .npz files — we only use IDs to intersect with the
    # DB rows loaded above. The intersection restricts results to the
    # parser-kept corpus (FFA + v15+ + player_count ∈ [4,8] + wire_data +
    # noise-floor-passing perspectives).
    # If we drop `iter_intermediate_files`, we need to filter to the appropriate
    # FFA subset some other way.
    parsed_ids = [replay_id_from_path(p) for p in iter_intermediate_files("sim")]
    print(f"  {len(parsed_ids)} parsed games in {time.perf_counter() - t0:.2f}s")

    pair_counter: Counter[tuple[int, int]] = Counter()
    by_month_total: dict[str, int] = defaultdict(int)
    by_month_over32: dict[str, int] = defaultdict(int)
    max_size_counts: dict[int, int] = defaultdict(int)
    over32 = 0
    missing = 0
    max_side: int = 0

    for rid in parsed_ids:
        info = db_info.get(rid)
        if info is None:
            missing += 1
            continue
        w, h, ms = info
        pair_counter[(w, h)] += 1
        max_side = max(max_side, w, h)
        month = datetime.fromtimestamp(ms / 1000, tz=UTC).strftime("%Y-%m")
        by_month_total[month] += 1

        max_size_counts[max(w, h)] += 1
        if max(w, h) > 32:
            over32 += 1
            by_month_over32[month] += 1

    total = sum(pair_counter.values())
    print()
    print(f"Total parsed games with map dims: {total} (missing: {missing})")
    print(
        f"max(w, h) > 32: {over32} / {total} ({100 * over32 / total:.2f}%)  "
        f"  largest side seen: {max_side}"
    )

    print()
    print("Top (w, h) pairs:")
    for (w, h), count in pair_counter.most_common(20):
        marker = " *" if max(w, h) > 32 else ""
        print(f"  {w:3d} x {h:3d}: {count:>7}  ({100 * count / total:5.2f}%){marker}")

    print()
    print("Max-side distribution across the corpus:")
    print_counts_with_percents(sorted(max_size_counts.items()), cumulative=True)

    print()
    print("By month (parsed games + >32-side fraction):")
    for month in sorted(by_month_total):
        n = by_month_total[month]
        o = by_month_over32[month]
        print(f"  {month}: games={n:>6}  over32={o:>5}  ({100 * o / n:5.2f}%)")


if __name__ == "__main__":
    main()
