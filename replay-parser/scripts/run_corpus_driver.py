"""CLI entry point for the corpus driver.

Loads the union of curated player lists (one path per line in the
top-level `curated-player-lists.txt`), builds a `DriverConfig`, and
calls `run_corpus_driver`.

Usage (from replay-parser/):
    uv run python scripts/run_corpus_driver.py [--limit N] [--workers W] [--output-dir DIR]
"""
import argparse
from pathlib import Path
import sys

from replay_parser.driver import (
    DEFAULT_MIN_PRIOR_GAMES,
    DEFAULT_ROLLING_1ST_FLOOR,
    DEFAULT_ROLLING_TOP3_FLOOR,
    DriverConfig,
    NoiseFloor,
    run_corpus_driver,
)
from replay_parser.git_state import DirtyWorkingTreeError
from settings import CURATED_LISTS_MANIFEST, DB_PATH, INTERMEDIATE_DIR, PROJECT_ROOT
from utils.player_name_lists import load_union


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Cap candidate count for smoke runs.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--output-dir", type=Path, default=INTERMEDIATE_DIR)
    ap.add_argument("--manifest", type=Path, default=CURATED_LISTS_MANIFEST)
    ap.add_argument("--rolling-1st-floor", type=float, default=DEFAULT_ROLLING_1ST_FLOOR)
    ap.add_argument("--rolling-top3-floor", type=float, default=DEFAULT_ROLLING_TOP3_FLOOR)
    ap.add_argument("--min-prior-games", type=int, default=DEFAULT_MIN_PRIOR_GAMES)
    ap.add_argument(
        "--allow-dirty", action="store_true",
        help="Proceed even if the working tree has uncommitted changes "
             "(output stamped <sha>-dirty).",
    )
    args = ap.parse_args()

    if not args.manifest.exists():
        print(f"manifest not found: {args.manifest}", file=sys.stderr)
        return 1
    if not DB_PATH.exists():
        print(f"DB not found: {DB_PATH}", file=sys.stderr)
        return 1

    print(f"Loading curated lists from {args.manifest}")
    curated = load_union(args.manifest, PROJECT_ROOT)
    if not curated:
        print("No curated names loaded.", file=sys.stderr)
        return 1
    print(f"Total curated names: {len(curated)}\n")

    config = DriverConfig(
        db_path=DB_PATH,
        intermediate_dir=args.output_dir,
        curated_names=tuple(curated),
        repo_root=PROJECT_ROOT,
        noise_floor=NoiseFloor(
            rolling_1st=args.rolling_1st_floor,
            rolling_top3=args.rolling_top3_floor,
            min_prior_games=args.min_prior_games,
        ),
        allow_dirty=args.allow_dirty,
    )
    try:
        run_corpus_driver(config, workers=args.workers, limit=args.limit)
    except DirtyWorkingTreeError as e:
        print(str(e), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
