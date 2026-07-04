"""git_sha returns a short SHA in a checkout."""

from __future__ import annotations

from utils.git import git_sha


def test_git_sha_in_checkout():
    sha = git_sha()
    assert isinstance(sha, str) and sha
