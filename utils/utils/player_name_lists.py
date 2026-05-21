"""Loaders for player-name list files — the txt-format building blocks
of the corpus pipeline.

Two layout tiers: single files (one username per line, edges stripped,
blanks skipped) and manifest files (one referenced filepath per line,
resolved against a caller-supplied project root, union'd with insertion-
order dedup). Validity-filtered and raw variants of the single-file
loader both exist; validity uses `replay_collector.usernames.filter_valid`.

Missing referenced files in the manifest loader log to stderr but don't
error, so the corpus pipeline keeps running on a partial manifest.
"""

from pathlib import Path
import sys

# ============================================================================
# TODO(cross-package-dep): `filter_valid` lives in `replay_collector.usernames`,
# which makes this import a backwards dependency — `utils` (a leaf utility
# package) reaching into `replay_collector` (a domain package). This is a
# deliberate compromise: moving `usernames.py` into `utils` is the right
# long-term fix but a wider change. Revisit when `usernames.py` gets touched
# for any other reason.
# ============================================================================
from replay_collector.usernames import filter_valid


def _read_lines(path: Path) -> list[str]:
    """Read non-blank lines with edge whitespace stripped. Internal spaces
    preserved — generals.io usernames may contain them, and manifest paths
    might too."""
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_players_raw(path: Path) -> list[str]:
    """Read one username per line. Strips edge whitespace, skips blanks
    (internal spaces preserved — generals.io usernames may contain them).
    No validity filtering; action sites should use `load_players`."""
    return _read_lines(path)


def load_players(path: Path) -> list[str]:
    """Like `load_players_raw`, with invalid names dropped + warned."""
    return filter_valid(load_players_raw(path))


def load_union(manifest_path: Path, project_root: Path) -> list[str]:
    """Read one filepath per line from `manifest_path`, resolve each
    against `project_root`, return the dedup'd union of names across the
    referenced files (insertion order preserved — first occurrence wins).

    Each referenced file is read via `load_players`, so invalid names are
    dropped + warned along the way. Missing referenced files log a
    warning to stderr but don't error, letting the pipeline iterate on a
    partial manifest.
    """
    refs = _read_lines(manifest_path)
    seen: set[str] = set()
    union: list[str] = []
    for rel in refs:
        path = project_root / rel
        if not path.exists():
            print(f"  WARN missing player-list file: {path}", file=sys.stderr)
            continue
        for name in load_players(path):
            if name not in seen:
                seen.add(name)
                union.append(name)
    return union
