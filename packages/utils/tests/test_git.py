"""git_sha returns a SHA in a checkout and 'unknown' if `git` fails."""

from __future__ import annotations

import subprocess

from utils.git import git_sha


def test_git_sha_in_checkout():
    sha = git_sha()
    assert isinstance(sha, str) and sha and sha != "unknown"


def test_git_sha_unknown_on_failure(monkeypatch):
    """The contract: return 'unknown' on git failure (don't raise)."""

    def boom(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "check_output", boom)
    assert git_sha() == "unknown"
