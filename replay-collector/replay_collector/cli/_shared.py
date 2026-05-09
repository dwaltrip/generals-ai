from pathlib import Path

from replay_collector.usernames import filter_valid

# tmp/ at the subproject root, where log files land.
TMP_DIR = Path(__file__).resolve().parent.parent.parent / "tmp"


def load_players_raw(path: Path) -> list[str]:
    """Read one username per line. Strips edge whitespace, skips blanks
    (internal spaces preserved — generals.io usernames may contain them).
    No validity filtering; action sites should use `load_players`."""
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_players(path: Path) -> list[str]:
    """Like `load_players_raw`, with invalid names dropped + warned."""
    return filter_valid(load_players_raw(path))


def fmt_duration(seconds: int) -> str:
    if seconds < 60:
        return f"~{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"~{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"~{h}h {m:02d}m {s:02d}s"
