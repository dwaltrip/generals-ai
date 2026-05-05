import sys
from pathlib import Path

from replay_collector import db, fill
from replay_collector.cli._shared import TMP_DIR, fmt_duration, load_players
from replay_collector.client import DEFAULT_RATES, host_of
from replay_collector.config import S3_BASE
from replay_collector.logging_setup import setup_simple_logging
from replay_collector.runner import DEFAULT_MAX_FAILURES

# Hardcoded for now — stop modes (--limit, --for) come later.
MAX_FETCHES = 1000


def add_parser(sub) -> None:
    p = sub.add_parser(
        "fetch-gior",
        help=f"download .gior bytes for up to {MAX_FETCHES:,} pending FFA replays "
             "(round-robin newest-first per player)",
    )
    p.add_argument(
        "--players", type=Path, default=None,
        help="optional file of usernames; restricts the work set to replays "
             "where at least one listed player is in the ranking, and "
             "balances round-robin among those players",
    )
    p.add_argument(
        "--max-failures", type=int, default=DEFAULT_MAX_FAILURES,
        help=f"abort the run after this many HTTP failures (default: {DEFAULT_MAX_FAILURES})",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", dest="mode", action="store_const", const="dry-run",
        help="report the plan (count, ETA) without fetching anything (default)",
    )
    mode.add_argument(
        "--no-dry-run", dest="mode", action="store_const", const="real",
        help="execute the run for real",
    )
    p.set_defaults(mode="dry-run", func=run)


def run(args) -> None:
    player_filter = _load_player_filter(args.players)

    if args.mode == "dry-run":
        _print_dry_run(player_filter)
        return

    log_path = setup_simple_logging(TMP_DIR, "fetch_gior")
    print(f"  log: {log_path}")

    work_rows = db.pending_full_data_work_set(
        player_filter=player_filter, limit=MAX_FETCHES,
    )
    result = fill.fill(work_rows, max_failures=args.max_failures)
    sys.exit(1 if result.aborted else 0)


def _load_player_filter(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    players = load_players(path)
    if not players:
        sys.exit(f"no usernames found in {path}")
    return players


def _print_dry_run(player_filter: list[str] | None) -> None:
    s3_rate = DEFAULT_RATES[host_of(S3_BASE)]
    total_pending = db.pending_full_data_count(player_filter)
    fetches_this_run = min(total_pending, MAX_FETCHES)
    eta_this_run = int(fetches_this_run / s3_rate)
    eta_full = int(total_pending / s3_rate)

    print("Dry run (pass --no-dry-run to execute).")
    print()
    print("Inputs:")
    print(f"  players filter: {len(player_filter) if player_filter else 'none — all DB players'}")
    print(f"  S3 rate:        {s3_rate:g} req/sec")
    print(f"  cap this run:   {MAX_FETCHES:,} fetches")
    print()
    print("Plan:")
    print(f"  total pending:    {total_pending:,}")
    print(f"  this run fetches: {fetches_this_run:,}   (ETA {fmt_duration(eta_this_run)})")
    if total_pending > fetches_this_run:
        print(f"  full backlog ETA: {fmt_duration(eta_full)} ({total_pending:,} fetches)")
    print()
    print("Note: errors are logged but not tracked in the DB; failed replays retry next run.")
