from __future__ import annotations

from pathlib import Path
import subprocess


def git_sha() -> str:
    """Short git SHA of HEAD, resolved against this package's checkout (not the
    process cwd). Raises if the `git` call fails.
    """
    out = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=Path(__file__).resolve().parent,
        text=True,
    )
    return out.strip()
