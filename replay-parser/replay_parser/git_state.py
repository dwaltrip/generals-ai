"""Git working-tree state capture for stamping output artifacts.

Used by the corpus driver to mark every meta sidecar with the exact
build of the source tree that produced it. The capture is runtime
(not build-time) so the SHA reflects "what code was actually loaded for
this corpus run," not "what was committed at maturin-rebuild time."

Returns one of:
  - `<short_sha>`         — clean working tree at this commit.
  - `<short_sha>-dirty`   — uncommitted changes to tracked files; only
                            returned when `allow_dirty=True`.
  - `"no-git"`            — git unavailable or repo metadata missing
                            (e.g., source-tarball deploy).

"Dirty" is restricted to tracked-file changes — untracked files like
scratch outputs in `tmp/` or `data/` don't trigger the gate.
"""
import subprocess
from pathlib import Path


class DirtyWorkingTreeError(RuntimeError):
    """Raised when a corpus run is started against a dirty working tree
    without explicit `allow_dirty=True`."""


def capture_git_version(repo_root: Path, allow_dirty: bool = False) -> str:
    sha = _run_git(["rev-parse", "--short", "HEAD"], repo_root)
    if sha is None:
        return "no-git"

    dirty = _is_dirty(repo_root)
    if dirty and not allow_dirty:
        raise DirtyWorkingTreeError(
            f"Working tree has uncommitted changes to tracked files (HEAD={sha}).\n"
            "Commit first, or re-run with --allow-dirty to proceed "
            "(output will be stamped <sha>-dirty)."
        )
    return f"{sha}-dirty" if dirty else sha


def _run_git(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _is_dirty(repo_root: Path) -> bool:
    # `git diff --quiet HEAD` exits non-zero if tracked-file changes exist
    # (in working tree or staged). Untracked files don't count.
    result = subprocess.run(
        ["git", "diff", "--quiet", "HEAD"],
        cwd=repo_root,
        capture_output=True,
    )
    return result.returncode != 0
