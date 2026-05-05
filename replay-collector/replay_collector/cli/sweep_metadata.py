import sys
from pathlib import Path

from replay_collector import db, sweep
from replay_collector.cli._shared import TMP_DIR, fmt_duration, load_players
from replay_collector.logging_setup import setup_sweep_logging
from replay_collector.runner import DEFAULT_MAX_FAILURES

DEFAULT_MAX_LISTINGS_PER_USER = 100_000  # safety rail, not a target
PASS_TWO_RATE_PER_SEC = 1.0


def add_parser(sub) -> None:
    p = sub.add_parser(
        "sweep-metadata",
        help="walk all replay listings per player; no .gior fetches",
    )
    p.add_argument(
        "players_file", type=Path,
        help="text file with one generals.io username per line",
    )
    p.add_argument(
        "--max-listings-per-player", type=int,
        default=DEFAULT_MAX_LISTINGS_PER_USER,
        help=f"safety rail; sweep stops at this many listings per player "
             f"(default: {DEFAULT_MAX_LISTINGS_PER_USER:,})",
    )
    p.add_argument(
        "--max-failures", type=int, default=DEFAULT_MAX_FAILURES,
        help=f"abort the run after this many HTTP failures (default: {DEFAULT_MAX_FAILURES})",
    )
    p.set_defaults(func=run)


def run(args) -> None:
    players = load_players(args.players_file)
    if not players:
        sys.exit(f"no usernames found in {args.players_file}")

    log_path = setup_sweep_logging(TMP_DIR)
    print(f"  log: {log_path}")

    result = sweep.sweep_many(
        players,
        max_listings=args.max_listings_per_player,
        max_failures=args.max_failures,
    )

    _emit_summary(players, log_path)
    sys.exit(1 if result.aborted else 0)


def _emit_summary(players: list[str], log_path: Path) -> None:
    """Cumulative DB summary — reflects all prior runs, not just this one.
    Written raw (no log-record prefix) to stdout and appended to log_path."""
    rows = db.replay_counts_by_player(players)
    lines = _format_summary_lines(rows)
    for line in lines:
        print(line)
    with log_path.open("a") as f:
        f.write("\n" + "\n".join(lines) + "\n")


def _format_summary_lines(
    rows: list[tuple[str, int, int, int]],
) -> list[str]:
    if not rows:
        return ["", "No replay data in DB for these players."]

    total_listings = sum(r[1] for r in rows)
    total_ffa = sum(r[2] for r in rows)
    total_metadata_only = sum(r[3] for r in rows)
    est_seconds = int(total_metadata_only / PASS_TWO_RATE_PER_SEC)

    name_w = max(len("player"), max(len(r[0]) for r in rows))
    header = f"  {'player':<{name_w}}  {'listings':>10}  {'ffa_total':>10}  {'metadata_only':>14}"
    sep = "  " + "─" * (len(header) - 2)

    lines = [
        "",
        "Pass 2 budget (FFA replays needing .gior fetch):",
        "",
        header,
    ]
    for name, listings, ffa_total, metadata_only in rows:
        lines.append(
            f"  {name:<{name_w}}  {listings:>10,}  {ffa_total:>10,}  {metadata_only:>14,}"
        )
    lines.append(sep)
    lines.append(
        f"  {'TOTAL':<{name_w}}  {total_listings:>10,}  {total_ffa:>10,}  {total_metadata_only:>14,}"
        f"   ({fmt_duration(est_seconds)} at 1/sec)"
    )
    return lines
