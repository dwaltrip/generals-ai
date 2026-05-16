"""Analyze 2-delta and 3-delta FFA sequence patterns by labeling each delta
as either its exact 0.5-multiple value or 'x' for continuous values.

This collapses messy continuous-valued deltas into a small set of discrete
patterns, making decay-related sequences (e.g. (-3.5, -3.5) or
(-2.5, -3.0, -3.5)) easy to identify and count. Continuous deltas indicate
the player played at least one game that week; exact 0.5-multiples indicate
zero games (decay-only weeks).

Outputs four tables per run:
  1. Overall 2-delta pattern counts
  2. 2-delta patterns segmented by starting week
  3. Overall 3-delta pattern counts
  4. 3-delta patterns segmented by starting week

The starting-week segmentation is intended to reveal the impact of the
"big team event" weekends (during which FFA decay is paused) on specific
calendar weeks.

Usage:
    uv run python scripts/analyze_sequence_patterns.py [season]
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys


METRIC = "ffa"
DATA_DIR = Path("data/leaderboards")
TOP_N_OVERALL = 25
TOP_N_SEGMENTED = 20
HALF_TOLERANCE = 1e-6


def label(delta: float) -> str:
    """Return delta's exact 0.5-multiple value as a string, else 'x'."""
    if abs(delta * 2 - round(delta * 2)) < HALF_TOLERANCE:
        return "0.0" if delta == 0 else f"{delta:+.1f}"
    return "x"


def load_trajectories(path: Path, metric: str) -> dict[str, dict[int, float]]:
    """Return {username: {week: stars}} for the given metric."""
    with path.open() as f:
        data = json.load(f)
    rankings = data["rankings"]
    traj: dict[str, dict[int, float]] = defaultdict(dict)
    for week in range(1, len(rankings)):
        for entry in rankings[week][metric]:
            traj[entry["username"]][week] = entry["stars"]
    return traj


def consecutive_runs(
    weeks_dict: dict[int, float], length: int
) -> list[list[int]]:
    sw = sorted(weeks_dict.keys())
    runs = []
    for i in range(len(sw) - length + 1):
        run = sw[i : i + length]
        if all(run[j + 1] == run[j] + 1 for j in range(length - 1)):
            runs.append(run)
    return runs


def collect_patterns(
    traj: dict[str, dict[int, float]], num_deltas: int
) -> list[tuple[int, tuple[str, ...]]]:
    """Return [(starting_week, pattern), ...] for every consecutive sequence
    of (num_deltas + 1) weeks across all players."""
    out: list[tuple[int, tuple[str, ...]]] = []
    run_length = num_deltas + 1
    for weeks in traj.values():
        for run in consecutive_runs(weeks, run_length):
            deltas = [
                weeks[run[i + 1]] - weeks[run[i]] for i in range(num_deltas)
            ]
            pattern = tuple(label(d) for d in deltas)
            out.append((run[0], pattern))
    return out


def fmt_pattern(pat: tuple[str, ...]) -> str:
    return "(" + ", ".join(pat) + ")"


def print_overall(
    num_deltas: int, patterns: list[tuple[int, tuple[str, ...]]]
) -> None:
    counter = Counter(p for _, p in patterns)
    total = sum(counter.values())
    print()
    print(
        f"=== overall {num_deltas}-delta patterns "
        f"({num_deltas + 1}-week sequences) ==="
    )
    print(f"total sequences: {total}    unique patterns: {len(counter)}")
    print()
    print(f"  {'count':>6} {'pct':>6}   pattern")
    print(f"  {'-' * 6} {'-' * 6}   {'-' * 40}")
    for pat, c in counter.most_common(TOP_N_OVERALL):
        print(f"  {c:>6} {c / total:>5.1%}    {fmt_pattern(pat)}")


def print_segmented(
    num_deltas: int,
    patterns: list[tuple[int, tuple[str, ...]]],
    season_weeks: int,
) -> None:
    """Pivot table: rows = top patterns, cols = starting weeks, cells = counts."""
    counter = Counter(p for _, p in patterns)
    top_patterns = [p for p, _ in counter.most_common(TOP_N_SEGMENTED)]

    by_pattern_week: dict[tuple[str, ...], dict[int, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for start_week, pat in patterns:
        by_pattern_week[pat][start_week] += 1

    max_start = season_weeks - num_deltas
    starts = list(range(1, max_start + 1))

    print()
    print(
        f"=== {num_deltas}-delta patterns segmented by starting week "
        f"(top {TOP_N_SEGMENTED}) ==="
    )
    print(
        "Each cell = count of that pattern in sequences starting at that week."
    )
    print()

    pat_col_w = max(len(fmt_pattern(p)) for p in top_patterns)
    header = (
        f"  {'pattern':<{pat_col_w}}  {'total':>6}   "
        + " ".join(f"{'w' + str(s):>4}" for s in starts)
    )
    print(header)
    print(
        f"  {'-' * pat_col_w}  {'-' * 6}   "
        + " ".join("-" * 4 for _ in starts)
    )
    for pat in top_patterns:
        total = counter[pat]
        cells = " ".join(f"{by_pattern_week[pat][s]:>4}" for s in starts)
        print(f"  {fmt_pattern(pat):<{pat_col_w}}  {total:>6}   {cells}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("season", type=int, nargs="?", default=42)
    args = parser.parse_args()

    path = DATA_DIR / f"season-{args.season}.json"
    if not path.exists():
        parser.error(f"missing {path}")

    print(f"loading {path}", file=sys.stderr)
    with path.open() as f:
        data = json.load(f)
    season_weeks = len(data["rankings"]) - 1  # rankings[0] is the 1v1 special

    traj = load_trajectories(path, METRIC)
    print(
        f"metric: {METRIC}, {len(traj)} unique players, {season_weeks} weeks",
        file=sys.stderr,
    )

    for num_deltas in (2, 3):
        patterns = collect_patterns(traj, num_deltas)
        print_overall(num_deltas, patterns)
        print_segmented(num_deltas, patterns, season_weeks)


if __name__ == "__main__":
    main()
