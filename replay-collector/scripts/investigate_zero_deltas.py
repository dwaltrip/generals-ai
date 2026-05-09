"""Investigate the 0.0 weekly-delta edge case.

For each consecutive-week pair on the FFA leaderboards, find players whose
stars value did not change at all. We want to know:
  - how many distinct players are affected
  - what stars values they sit at (clustered = clamping, scattered = coincidence)
  - whether 0.0 deltas chain across multiple weeks for the same player

Operates on snapshots `rankings[1..10]` (weekly FFA boards). `rankings[0]` is
a 1v1-related special entry and is skipped.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "leaderboards" / "season-42.json"
EPS = 1e-9
METRICS = ("ffa", "ffawin")


def load_weekly_boards(metric: str) -> list[dict[str, float]]:
    """Return weeks 1..10 as a list of {username: stars} dicts."""
    raw = json.loads(DATA.read_text(encoding="utf-8"))
    snaps = raw["rankings"][1:11]
    return [{e["username"]: e["stars"] for e in snap[metric]} for snap in snaps]


def find_zero_deltas(boards: list[dict[str, float]]):
    """Yield (week_index_of_first, username, stars) for each exact-zero pair."""
    for w in range(len(boards) - 1):
        a, b = boards[w], boards[w + 1]
        for user, stars_a in a.items():
            stars_b = b.get(user)
            if stars_b is None:
                continue
            if abs(stars_b - stars_a) < EPS:
                yield w, user, stars_a


def total_consecutive_pairs(boards: list[dict[str, float]]) -> int:
    n = 0
    for w in range(len(boards) - 1):
        a, b = boards[w], boards[w + 1]
        n += sum(1 for u in a if u in b)
    return n


def report(metric: str) -> None:
    boards = load_weekly_boards(metric)
    print(f"=== {metric} ===")
    print(f"weeks loaded: {len(boards)}, sizes: {[len(b) for b in boards]}")
    total_pairs = total_consecutive_pairs(boards)
    zeros = list(find_zero_deltas(boards))
    print(f"consecutive-week pairs: {total_pairs}")
    print(f"exact-zero deltas: {len(zeros)} ({len(zeros) / total_pairs * 100:.2f}%)")

    affected = {u for _, u, _ in zeros}
    print(f"distinct players with at least one 0.0 delta: {len(affected)}")

    # Per-player counts
    per_player = Counter(u for _, u, _ in zeros)
    multi = sorted(((c, u) for u, c in per_player.items() if c >= 2), reverse=True)
    print(f"players with 2+ zero deltas: {len(multi)}")
    if multi:
        print("  top repeat offenders:")
        for c, u in multi[:10]:
            print(f"    {c}x  {u}")

    # Stars distribution at the time of the 0.0 — clustered or scattered?
    star_buckets = Counter()
    for _, _, s in zeros:
        star_buckets[round(s, 2)] += 1
    most_common = star_buckets.most_common(15)
    print("most common stars values during 0.0 weeks:")
    for s, c in most_common:
        print(f"  {s:>7.2f}: {c}")

    # Range
    if zeros:
        all_stars = [s for _, _, s in zeros]
        print(f"stars range during 0.0: min={min(all_stars):.2f} max={max(all_stars):.2f} "
              f"mean={sum(all_stars)/len(all_stars):.2f}")

    # Chained zeros: same player, 0.0 in week w then 0.0 in week w+1
    by_user: dict[str, set[int]] = defaultdict(set)
    for w, u, _ in zeros:
        by_user[u].add(w)
    chains = sum(1 for u, ws in by_user.items() if any((w + 1) in ws for w in ws))
    print(f"players with consecutive 0.0 deltas (chain ≥2): {chains}")
    print()


if __name__ == "__main__":
    for m in METRICS:
        report(m)
