"""Show DB-listings ranking vs simulator-deduced ranking for one or more replays.

Usage (from replay-parser/):
    uv run python scripts/show_ranking.py YC-ZjiMiY KVt1xVdFd
"""
import argparse
import sqlite3
from datetime import datetime, timezone

from tabulate import tabulate

from replay_parser._collector.config import DB_PATH
from replay_parser._collector.wire import decode as decode_blob
from replay_parser.errors import ArmyOverflowError
from replay_parser.parser import parse_replay
from replay_parser.validator import apply_surrender_bonus, deduce_ranking_for_replay


def fetch(conn, replay_id):
    row = conn.execute(
        "SELECT started, version, wire_data FROM replays WHERE id = ?",
        (replay_id,),
    ).fetchone()
    if row is None:
        return None, None, None, None
    started, version, blob = row
    listings_names = [
        name for (name,) in conn.execute(
            "SELECT current_name FROM replay_players "
            "WHERE replay_id = ? ORDER BY position",
            (replay_id,),
        )
    ]
    return started, version, blob, listings_names


DAMAGE_VARIANTS = [
    ("damage_sym_all", "sym/all"),
    ("damage_sym_pre", "sym/pre"),
    ("damage_off_all", "off/all"),
    ("damage_off_pre", "off/pre"),
]


def top_damager(matrix, victim, usernames, allowed=None):
    """Return 'name (value)' for the top damager of `victim`, or '—'.
    If `allowed` is given, restrict candidates to that set of player ids."""
    n = matrix.shape[0]
    best, best_val = None, 0
    for q in range(n):
        if q == victim:
            continue
        if allowed is not None and q not in allowed:
            continue
        v = int(matrix[q, victim])
        if v > best_val:
            best, best_val = q, v
    if best is None:
        return "—"
    return f"{usernames[best][:10]} ({best_val})"


def render_damage(state, usernames):
    captured = {ce.captured for ce in state.capture_events}
    surrenderers = [
        (de.player, de.timestep) for de in state.death_events
        if de.player not in captured
    ]
    if not surrenderers:
        print("(no surrenders in this replay)\n")
        return

    death_t = {de.player: de.timestep for de in state.death_events}
    n = state.num_players

    headers = ["surrenderer (t)", "filter", *[label for _, label in DAMAGE_VARIANTS]]
    rows = []
    for p, t in surrenderers:
        alive_set = {
            q for q in range(n)
            if q != p and death_t.get(q, float("inf")) > t
        }
        for label, allowed in (("any", None), ("alive", alive_set)):
            row = [f"{usernames[p][:10]} ({t})", label]
            for attr, _ in DAMAGE_VARIANTS:
                row.append(top_damager(getattr(state, attr), p, usernames, allowed))
            rows.append(row)
    print("Top damager per surrenderer × damage variant:")
    print(tabulate(rows, headers=headers))
    print()


def render(replay_id, started, version, listings_names, deduced_names, deduced_has_kill):
    when = datetime.fromtimestamp(started / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n = max(len(listings_names), len(deduced_names))
    rows = []
    for i in range(n):
        a = listings_names[i] if i < len(listings_names) else ""
        b = deduced_names[i] if i < len(deduced_names) else ""
        if b and i < len(deduced_has_kill) and deduced_has_kill[i]:
            b = f"{b} ☠"
        mark = "✓" if a and b and (a == b or b.startswith(a + " ")) else "✗"
        rows.append([i + 1, a, b, mark])
    header = f"=== {replay_id}  (v{version}, {when}, {n} players) ==="
    print(header)
    print(tabulate(rows, headers=["rank", "listings (DB)", "deduced (sim)", ""]))
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("replay_ids", nargs="+")
    ap.add_argument("--damage", action="store_true",
                    help="Also print the damage_dealt matrix for each replay.")
    args = ap.parse_args()

    conn = sqlite3.connect(DB_PATH)
    try:
        for replay_id in args.replay_ids:
            started, version, blob, listings_names = fetch(conn, replay_id)
            if blob is None:
                print(f"=== {replay_id} ===\n  NOT FOUND in DB\n")
                continue
            try:
                state, replay = parse_replay(blob)
            except ArmyOverflowError as e:
                print(f"=== {replay_id} ===\n  parse skipped: {e}\n")
                continue
            except Exception as e:
                print(f"=== {replay_id} ===\n  parse error: {type(e).__name__}: {e}\n")
                continue
            deduced_slots = deduce_ranking_for_replay(state, started)
            has_kill = apply_surrender_bonus(state)
            deduced_names = [replay.static.usernames[p] for p in deduced_slots]
            deduced_has_kill = [has_kill[p] for p in deduced_slots]
            render(replay_id, started, version, listings_names, deduced_names, deduced_has_kill)
            if args.damage:
                render_damage(state, replay.static.usernames)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
