"""Per-player winrate + stars stats across the 500 most-recent FFA games, in 10x50 buckets.

For each player in the input list, pulls their 500 most-recent FFA listings
from the local DB, splits them into 10 buckets of 50 (bucket 0 = most
recent), and per bucket computes:
  - winrate (share of games where the player finished position 0)
  - stars p10, p50, p90 (NULL stars excluded; linear-interp percentiles)

Output: one row per player. Columns: username, total_games, then 40 bucket
columns: b{i}_winrate, b{i}_stars_p{10,50,90} for i in 0..9. Cells are
blank when the bucket is empty (player has fewer games in DB than the
bucket starts at).

Usage (from replay-collector/):
    uv run python scripts/winrate_star_buckets.py players.txt --output out.csv
    uv run python scripts/winrate_star_buckets.py most_games_n_1000.csv --output out.csv

Player-list format: a .csv with a `name` column, or any other extension
treated as one-username-per-line (blank lines and `#` comments skipped).
"""

import argparse
import csv
from pathlib import Path
import sys

from replay_collector.cli._shared import load_players_raw
from replay_collector.db import create_conn
from replay_collector.sql_helpers import ffa_match_filter, from_player_games
from replay_collector.usernames import display_name, filter_valid
from utils.docstring import doc_summary


BUCKET_SIZE = 50
NUM_BUCKETS = 10
WINDOW = BUCKET_SIZE * NUM_BUCKETS


def read_player_list(path: Path) -> list[str]:
    """Parse `path` and drop invalid names (warned to stderr). Matches the
    `filter_valid`-at-load-boundary convention used by `_shared.load_players`
    and `build_player_list.load_users_from_json`."""
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "name" not in reader.fieldnames:
                sys.exit(f"{path}: CSV needs a 'name' column (got {reader.fieldnames})")
            names = [row["name"] for row in reader if row.get("name")]
    else:
        names = load_players_raw(path)
    return filter_valid(names)


def fetch_recent_games(conn, name: str) -> list[tuple[int, int | None]]:
    """Last WINDOW (position, stars) tuples for `name`'s FFA games, newest first."""
    return conn.execute(
        f"""
        SELECT rp.position, rp.stars
        {from_player_games()}
        WHERE p.name = ?
          AND {ffa_match_filter("r")}
        ORDER BY r.started DESC, r.id
        LIMIT ?
        """,
        (name, WINDOW),
    ).fetchall()


def percentile(values: list[float], q: float) -> float:
    """Linear interpolation between adjacent ranks (NumPy-style). q in [0, 100]."""
    s = sorted(values)
    n = len(s)
    if n == 1:
        return float(s[0])
    rank = (q / 100.0) * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    return s[lo] + (s[hi] - s[lo]) * (rank - lo)


def bucket_stats(games: list[tuple[int, int | None]]) -> dict:
    if not games:
        return {"winrate": "", "p10": "", "p50": "", "p90": ""}
    wins = sum(1 for pos, _ in games if pos == 0)
    stars = [s for _, s in games if s is not None]
    if stars:
        p10 = round(percentile(stars, 10), 1)
        p50 = round(percentile(stars, 50), 1)
        p90 = round(percentile(stars, 90), 1)
    else:
        p10 = p50 = p90 = ""
    return {
        "winrate": round(wins / len(games), 4),
        "p10": p10,
        "p50": p50,
        "p90": p90,
    }


def build_row(name: str, games: list[tuple[int, int | None]]) -> dict:
    row = {"username": display_name(name), "total_games": len(games)}
    for i in range(NUM_BUCKETS):
        chunk = games[i * BUCKET_SIZE : (i + 1) * BUCKET_SIZE]
        stats = bucket_stats(chunk)
        row[f"b{i}_wr"] = stats["winrate"]
        row[f"b{i}_sp10"] = stats["p10"]
        row[f"b{i}_sp50"] = stats["p50"]
        row[f"b{i}_sp90"] = stats["p90"]
    return row


def fieldnames() -> list[str]:
    cols = ["username", "total_games"]
    for i in range(NUM_BUCKETS):
        cols += [
            f"b{i}_wr",
            f"b{i}_sp10",
            f"b{i}_sp50",
            f"b{i}_sp90",
        ]
    return cols


def main() -> None:
    parser = argparse.ArgumentParser(description=doc_summary(__doc__))
    parser.add_argument(
        "players_file",
        type=Path,
        help="CSV with a 'name' column, or text file with one username per line",
    )
    parser.add_argument("--output", type=Path, required=True, help="output CSV path")
    args = parser.parse_args()

    names = read_player_list(args.players_file)
    if not names:
        sys.exit(f"{args.players_file}: no players")

    try:
        conn = create_conn()
    except FileNotFoundError as e:
        sys.exit(str(e))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames())
        writer.writeheader()
        for name in names:
            games = fetch_recent_games(conn, name)
            writer.writerow(build_row(name, games))
    print(f"wrote {len(names)} rows to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
