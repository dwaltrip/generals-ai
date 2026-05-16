"""Compute activity-aware persistence scores for the season-42 FFA leaderboard.

Population: any player who appeared in top 100 of `ffa` OR `ffawin` in any
of the 10 weekly snapshots. (Top 50 ⊂ top 100, so the cutoff = 100.)

Activity classifier (per consecutive-week pair):
  - per-board: inactive iff delta == 0 OR (delta on the 0.5 grid AND delta < 0)
  - cross-board OR rule: a player-week is active if EITHER `ffa` OR
    `ffawin` says active. This denoises ~2% per-metric coincidental landings
    on the 0.5 grid by a continuous game delta.
  - first appearance on a board = active for that week.

Scoring (linear, per active week, per metric, per cutoff N):
  contribution = max(0, N - rank + 1)

Output: top-30 players sorted by `ffa_top_100_avg_rank` (ascending), full
table written to `tmp/persistence_scores.csv`, plus a summary-stats table
across all population players for each metric column.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics

from replay_collector.usernames import display_name


EPS = 1e-9
HALF_EPS = 1e-6
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "leaderboards" / "season-42.json"
OUT_DIR = ROOT / "tmp"
OUT_CSV = OUT_DIR / "persistence_scores.csv"

METRICS = ("ffa", "ffawin")
CUTOFFS = (50, 100)
PERCENTILES = (5, 25, 50, 75, 95)


def per_board_inactive(delta: float) -> bool:
    """A delta is decay-only (inactive) iff it's zero OR an exact negative half-step."""
    if abs(delta) < EPS:
        return True
    nearest_half = round(delta * 2) / 2
    if abs(delta - nearest_half) < HALF_EPS and delta < 0:
        return True
    return False


def load_boards():
    """Return n_weeks plus per-metric stars and rank lookups indexed by week."""
    raw = json.loads(DATA.read_text(encoding="utf-8"))
    snaps = raw["rankings"][1:11]
    n_weeks = len(snaps)
    stars = {m: [] for m in METRICS}
    ranks = {m: [] for m in METRICS}
    for s in snaps:
        for m in METRICS:
            entries = s[m]
            stars[m].append({e["username"]: e["stars"] for e in entries})
            ranks[m].append({e["username"]: i + 1 for i, e in enumerate(entries)})
    return n_weeks, stars, ranks


def compute_activity(stars, n_weeks):
    """Return dict[(user, week_idx)] -> bool for every (user, week) we observe.

    Cross-board OR rule: a week is active iff at least one metric has
    delta-evidence of activity, OR no prior-week signal exists at all
    (first appearance on any board).
    """
    activity: dict[tuple[str, int], bool] = {}
    all_users = set().union(*(b.keys() for m in METRICS for b in stars[m]))
    for u in all_users:
        for w in range(n_weeks):
            on_any_now = any(u in stars[m][w] for m in METRICS)
            if not on_any_now:
                continue
            if w == 0:
                activity[(u, w)] = True
                continue
            inactive_signals = active_signals = 0
            for m in METRICS:
                prev, cur = stars[m][w - 1], stars[m][w]
                if u in prev and u in cur:
                    if per_board_inactive(cur[u] - prev[u]):
                        inactive_signals += 1
                    else:
                        active_signals += 1
            if active_signals > 0:
                activity[(u, w)] = True
            elif inactive_signals > 0:
                activity[(u, w)] = False
            else:
                # First appearance on a board — no prior delta to compare.
                activity[(u, w)] = True
    return activity


def build_rows(stars, ranks, activity, n_weeks):
    population = set()
    for w in range(n_weeks):
        for m in METRICS:
            for u, r in ranks[m][w].items():
                if r <= max(CUTOFFS):
                    population.add(u)

    rows = []
    for u in population:
        sums = {f"{m}_top_{c}": 0 for m in METRICS for c in CUTOFFS}
        top_100_ranks = {m: [] for m in METRICS}
        active_weeks = 0
        for w in range(n_weeks):
            is_active = activity.get((u, w), False)
            on_any = any(u in stars[m][w] for m in METRICS)
            if is_active and on_any:
                active_weeks += 1
            for m in METRICS:
                r = ranks[m][w].get(u)
                if r is None or not is_active:
                    continue
                for c in CUTOFFS:
                    if r <= c:
                        sums[f"{m}_top_{c}"] += c - r + 1
                if r <= 100:
                    top_100_ranks[m].append(r)
        rows.append({
            "username": u,
            "ffa_top_50": sums["ffa_top_50"],
            "ffa_top_100": sums["ffa_top_100"],
            "ffa_top_100_avg_rank": (statistics.mean(top_100_ranks["ffa"])
                                      if top_100_ranks["ffa"] else None),
            "ffa_top_100_n": len(top_100_ranks["ffa"]),
            "ffawin_top_50": sums["ffawin_top_50"],
            "ffawin_top_100": sums["ffawin_top_100"],
            "ffawin_top_100_avg_rank": (statistics.mean(top_100_ranks["ffawin"])
                                         if top_100_ranks["ffawin"] else None),
            "ffawin_top_100_n": len(top_100_ranks["ffawin"]),
            "active_weeks": active_weeks,
        })

    # Sort: ascending by ffa_top_100_avg_rank; players with no active top-100
    # ffa weeks (avg=None) sort to the bottom.
    rows.sort(key=lambda r: (
        r["ffa_top_100_avg_rank"] is None,
        r["ffa_top_100_avg_rank"] if r["ffa_top_100_avg_rank"] is not None else 0,
    ))
    return rows


def percentile(sorted_vals, p):
    if not sorted_vals:
        return None
    idx = (p / 100) * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def summary_stats(rows, columns):
    """Return dict[column] -> dict of stats. Skips None values per column."""
    out = {}
    for col in columns:
        vals = [r[col] for r in rows if r[col] is not None]
        if not vals:
            out[col] = None
            continue
        vs = sorted(vals)
        out[col] = {
            "n": len(vs),
            "min": vs[0],
            **{f"p{p}": percentile(vs, p) for p in (5, 25)},
            "median": percentile(vs, 50),
            "avg": sum(vs) / len(vs),
            **{f"p{p}": percentile(vs, p) for p in (75, 95)},
            "max": vs[-1],
        }
    return out


def fmt(v, width=8, prec=2):
    if v is None:
        return f"{'—':>{width}}"
    if isinstance(v, float):
        return f"{v:>{width}.{prec}f}"
    return f"{v:>{width}}"


def print_top_table(rows, n=30):
    headers = [
        ("username", 22, "<"),
        ("ffa_t50", 8, ">"),
        ("ffa_t100", 9, ">"),
        ("ffa_avg", 8, ">"),
        ("fw_t50", 8, ">"),
        ("fw_t100", 8, ">"),
        ("fw_avg", 8, ">"),
        ("active", 7, ">"),
    ]
    keymap = {
        "ffa_t50": "ffa_top_50",
        "ffa_t100": "ffa_top_100",
        "ffa_avg": "ffa_top_100_avg_rank",
        "fw_t50": "ffawin_top_50",
        "fw_t100": "ffawin_top_100",
        "fw_avg": "ffawin_top_100_avg_rank",
        "active": "active_weeks",
    }
    line = "  ".join(f"{h:{a}{w}}" for h, w, a in headers)
    print(line)
    print("-" * len(line))
    for r in rows[:n]:
        cells = []
        for h, w, a in headers:
            if h == "username":
                name = display_name(r["username"])
                if len(name) > w:
                    name = name[:w-1] + "…"
                cells.append(f"{name:{a}{w}}")
            else:
                v = r[keymap[h]]
                if isinstance(v, float):
                    cells.append(f"{v:{a}{w}.2f}")
                elif v is None:
                    cells.append(f"{'—':{a}{w}}")
                else:
                    cells.append(f"{v:{a}{w}}")
        print("  ".join(cells))


def print_summary_stats(stats_dict):
    cols = list(stats_dict.keys())
    stat_keys = ["n", "min", "p5", "p25", "median", "avg", "p75", "p95", "max"]
    header = f"{'metric':<24}  " + "  ".join(f"{k:>9}" for k in stat_keys)
    print(header)
    print("-" * len(header))
    for col in cols:
        s = stats_dict[col]
        if s is None:
            print(f"{col:<24}  {'(no data)':>9}")
            continue
        cells = []
        for k in stat_keys:
            v = s[k]
            if k == "n":
                cells.append(f"{v:>9}")
            else:
                cells.append(f"{v:>9.2f}" if isinstance(v, float) else f"{v:>9}")
        print(f"{col:<24}  " + "  ".join(cells))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_weeks, stars, ranks = load_boards()
    activity = compute_activity(stars, n_weeks)
    rows = build_rows(stars, ranks, activity, n_weeks)

    print(f"=== Persistence scores: top 30 of {len(rows)} (sort: ffa_top_100_avg_rank asc) ===")
    print()
    print_top_table(rows, n=30)
    print()

    metric_cols = [
        "ffa_top_50", "ffa_top_100", "ffa_top_100_avg_rank", "ffa_top_100_n",
        "ffawin_top_50", "ffawin_top_100", "ffawin_top_100_avg_rank", "ffawin_top_100_n",
        "active_weeks",
    ]
    stats_dict = summary_stats(rows, metric_cols)
    print(f"=== Summary stats across all {len(rows)} population players ===")
    print()
    print_summary_stats(stats_dict)

    fields = ["username"] + metric_cols
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            row_out = {k: r[k] for k in fields}
            row_out["username"] = display_name(row_out["username"])
            w.writerow(row_out)
    print()
    print(f"full table: {OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
