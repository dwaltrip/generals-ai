import argparse
import math
import sys
from pathlib import Path

from replay_collector.generals_api import PAGE_SIZE
from replay_collector.logging_setup import setup_logging
from replay_collector.runner import (
    DEFAULT_MAX_FAILURES,
    DEFAULT_MAX_LISTINGS_PER_USER,
    collect_many,
)

# Dry-run uses these to bracket the estimate. Real-world FFA share among top
# players' recent games tends to land somewhere in this band.
DRY_RUN_FFA_RATES = (0.25, 1.00)

# tmp/ at the subproject root, where log files land.
TMP_DIR = Path(__file__).resolve().parent.parent / "tmp"


def load_players(path: Path) -> list[str]:
    """One username per line. Strips whitespace and skips blanks; preserves
    internal spaces (generals.io usernames may contain them)."""
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _estimate(n_users: int, n_ffa: int, max_listings: int, rate: float) -> dict:
    walked = min(math.ceil(n_ffa / rate), max_listings)
    ffa_found = min(n_ffa, math.floor(walked * rate))
    pages = math.ceil(walked / PAGE_SIZE)
    api_calls = 1 + pages  # 1 starsAndRanks + N listing pages
    s3_calls = ffa_found
    # Per host: 1 req/s. Calls on different hosts interleave during waits, so
    # per-user wall time is bounded by the slower stream, not the sum.
    per_user_seconds = max(api_calls, s3_calls)
    return {
        "walked": walked,
        "ffa": ffa_found,
        "api_calls": n_users * api_calls,
        "s3_calls": n_users * s3_calls,
        "wall_seconds": n_users * per_user_seconds,
    }


def _fmt_duration(seconds: int) -> str:
    if seconds < 60:
        return f"~{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"~{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"~{h}h {m:02d}m {s:02d}s"


def print_dry_run(args, players: list[str]) -> None:
    n = len(players)
    low, high = (_estimate(n, args.n_ffa, args.max_listings, r) for r in DRY_RUN_FFA_RATES)

    print("Dry run (pass --no-dry-run to execute).")
    print()
    print("Inputs:")
    print(f"  players: {n} (from {args.players_file})")
    print(f"  per-player target: {args.n_ffa} FFA replays")
    print(f"  caps: max-listings={args.max_listings}, max-failures={args.max_failures}")
    print()
    pct_lo, pct_hi = (int(r * 100) for r in DRY_RUN_FFA_RATES)
    print(f"{'':<28}{'FFA=' + str(pct_lo) + '%':<16}{'FFA=' + str(pct_hi) + '%'}")
    rows = [
        ("listings walked / player", low["walked"], high["walked"]),
        ("/api calls (total)", low["api_calls"], high["api_calls"]),
        ("S3 .gior fetches (total)", low["s3_calls"], high["s3_calls"]),
        ("est. wall time", _fmt_duration(low["wall_seconds"]), _fmt_duration(high["wall_seconds"])),
    ]
    for label, lo, hi in rows:
        print(f"  {label:<26}{str(lo):<16}{hi}")
    print()
    print("Note: .gior fetch count assumes nothing is already cached; re-runs will be faster.")


def print_banner(
    n_players: int,
    players_file: Path,
    condensed_path: Path,
    verbose_path: Path,
    test_logger: bool,
) -> None:
    label = "test-logger run" if test_logger else "running"
    print(f"{label} replay-collector for {n_players} players (from {players_file}).")
    print(f"  condensed log: {condensed_path}")
    print(f"  verbose log:   {verbose_path}")
    print(f"  tail with: tail -f {condensed_path}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="replay_collector")
    parser.add_argument(
        "players_file", type=Path,
        help="text file with one generals.io username per line",
    )
    parser.add_argument(
        "--n-ffa", type=int, required=True,
        help="target number of FFA replays to fetch per player",
    )
    parser.add_argument(
        "--max-listings", type=int, default=DEFAULT_MAX_LISTINGS_PER_USER,
        help=f"max listings to walk per player (default: {DEFAULT_MAX_LISTINGS_PER_USER})",
    )
    parser.add_argument(
        "--max-failures", type=int, default=DEFAULT_MAX_FAILURES,
        help=f"abort the run after this many HTTP failures (default: {DEFAULT_MAX_FAILURES})",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", dest="mode", action="store_const", const="dry-run",
        help="print estimates without making API calls (default)",
    )
    mode.add_argument(
        "--test-logger", dest="mode", action="store_const", const="test-logger",
        help="walk one bucket per player and skip .gior fetches; for testing log output",
    )
    mode.add_argument(
        "--no-dry-run", dest="mode", action="store_const", const="real",
        help="execute the run for real",
    )
    parser.set_defaults(mode="dry-run")
    args = parser.parse_args()

    players = load_players(args.players_file)
    if not players:
        parser.error(f"no usernames found in {args.players_file}")

    if args.mode == "dry-run":
        print_dry_run(args, players)
        return

    test_logger = args.mode == "test-logger"
    if test_logger:
        # One bucket per player. Overrides any user-supplied --max-listings.
        args.max_listings = PAGE_SIZE

    condensed_path, verbose_path, progress = setup_logging(TMP_DIR)
    print_banner(len(players), args.players_file, condensed_path, verbose_path, test_logger)

    run = collect_many(
        players,
        n_ffa=args.n_ffa,
        progress=progress,
        max_listings=args.max_listings,
        max_failures=args.max_failures,
        skip_full_fetch=test_logger,
    )
    sys.exit(1 if run.aborted else 0)


if __name__ == "__main__":
    main()
