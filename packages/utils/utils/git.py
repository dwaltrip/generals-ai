from __future__ import annotations

from pathlib import Path
import subprocess
from typing import overload


@overload
def git_sha(default: str) -> str: ...

@overload
def git_sha(default: None = None) -> str | None: ...

def git_sha(default: str | None  = None) -> str | None:
    """Short git SHA of HEAD for the repository."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return default
