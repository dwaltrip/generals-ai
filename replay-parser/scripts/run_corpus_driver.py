"""CLI entry point for the corpus driver.

Loads the union of curated player lists (one path per line in the
top-level `curated-player-lists.txt`), builds a `DriverConfig`, and
calls `run_corpus_driver`.

Usage (from replay-parser/):
    uv run python scripts/run_corpus_driver.py [--limit N] [--workers W] [--output-dir DIR]
"""
import argparse
import sys
from pathlib import Path

from replay_collector.cli._shared import load_players
from replay_parser._collector.config import DB_PATH
from replay_parser.driver import (
    DEFAULT_MIN_PRIOR_GAMES,
    DEFAULT_ROLLING_1ST_FLOOR,
    DEFAULT_ROLLING_TOP3_FLOOR,
    DriverConfig,
    NoiseFloor,
    run_corpus_driver,
)


# Project-root-relative path to the curated-list manifest.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CURATED_LISTS_MANIFEST = PROJECT_ROOT / "curated-player-lists.txt"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "intermediate"


def _load_union(manifest_path: Path) -> list[str]:
    """Read one filepath per line from the manifest; union their names."""
    files = [
        line.strip()
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    seen: set[str] = set()
    union: list[str] = []
    for rel in files:
        path = PROJECT_ROOT / rel
        if not path.exists():
            print(f"  WARN missing curated file: {path}", file=sys.stderr)
            continue
        for name in load_players(path):
            if name not in seen:
                seen.add(name)
                union.append(name)
        print(f"  loaded {rel}  → cumulative union size = {len(union)}")
    return union


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Cap candidate count for smoke runs.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--manifest", type=Path, default=CURATED_LISTS_MANIFEST)
    ap.add_argument("--rolling-1st-floor", type=float, default=DEFAULT_ROLLING_1ST_FLOOR)
    ap.add_argument("--rolling-top3-floor", type=float, default=DEFAULT_ROLLING_TOP3_FLOOR)
    ap.add_argument("--min-prior-games", type=int, default=DEFAULT_MIN_PRIOR_GAMES)
    args = ap.parse_args()

    if not args.manifest.exists():
        print(f"manifest not found: {args.manifest}", file=sys.stderr)
        return 1
    if not DB_PATH.exists():
        print(f"DB not found: {DB_PATH}", file=sys.stderr)
        return 1

    print(f"Loading curated lists from {args.manifest}")
    curated = _load_union(args.manifest)
    if not curated:
        print("No curated names loaded.", file=sys.stderr)
        return 1
    print(f"Total curated names: {len(curated)}\n")

    config = DriverConfig(
        db_path=DB_PATH,
        intermediate_dir=args.output_dir,
        curated_names=tuple(curated),
        noise_floor=NoiseFloor(
            rolling_1st=args.rolling_1st_floor,
            rolling_top3=args.rolling_top3_floor,
            min_prior_games=args.min_prior_games,
        ),
    )
    run_corpus_driver(config, workers=args.workers, limit=args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
