"""Git provenance helpers."""

from __future__ import annotations

from pathlib import Path
import subprocess


def git_sha() -> str:
    """Short git SHA of HEAD, or 'unknown' if git can't answer.

    Resolved against this module's location, not the process cwd. Returns
    'unknown' rather than raising when there's no reachable `.git` (e.g. source
    shipped to a remote without it).
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
