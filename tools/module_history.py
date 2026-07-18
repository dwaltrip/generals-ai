#!/usr/bin/env -S uv run python
"""Per-session edit statistics for a file or directory, from git history.

Commits touching the path are bucketed into work sessions wherever the gap
between consecutive author dates stays under --max-gap (author dates survive
rebases and squashes, unlike committer dates). Each session's line counts come
from one whitespace-ignoring net diff between its first and last commits in
DAG order, so intra-session churn collapses to what the session actually
changed.

Known limitation: for a renamed file, --follow recovers the full commit
history, but the net diffs only match the current name, so sessions from
before the rename report near-zero line counts.
"""

import argparse
import json
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from utils.format import format_count, md_table

# git's well-known hash of the empty tree: the diff base when a session starts
# at the root commit.
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"


@dataclass
class Commit:
    sha: str
    parents: list[str]
    author_ts: int
    subject: str
    topo_idx: int  # position in `git log` output: 0 is newest


@dataclass
class Session:
    commits: list[Commit]  # author-date order, newest first
    added: int
    removed: int
    non_contiguous: bool

    @property
    def start_ts(self) -> int:
        return min(c.author_ts for c in self.commits)

    @property
    def end_ts(self) -> int:
        return max(c.author_ts for c in self.commits)


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    )
    return result.stdout


def collect_commits(path: str) -> tuple[list[Commit], bool]:
    """Return (commits touching path, whether renames were followed)."""
    follow = Path(path).is_file()  # --follow only works for a single file
    log_args = ["log", "--format=%H%x09%P%x09%at%x09%s"]
    if follow:
        log_args.append("--follow")
    out = git(*log_args, "--", path)
    commits = []
    for idx, line in enumerate(out.splitlines()):
        sha, parents, ts, subject = line.split("\t", 3)
        commits.append(Commit(sha, parents.split(), int(ts), subject, topo_idx=idx))
    return commits, follow


def bucket_sessions(commits: list[Commit], max_gap_hours: float) -> list[list[Commit]]:
    """Bucket commits by author-date gaps; sessions and their commits newest first."""
    ordered = sorted(commits, key=lambda c: c.author_ts)
    max_gap = max_gap_hours * 3600
    buckets = [[ordered[0]]]
    for prev, cur in zip(ordered, ordered[1:]):
        if cur.author_ts - prev.author_ts > max_gap:
            buckets.append([])
        buckets[-1].append(cur)
    return [b[::-1] for b in reversed(buckets)]


def build_session(commits: list[Commit], path: str) -> Session:
    # If a rewrite moved a commit across a session boundary, the session's
    # commits no longer form a run in topo order and the net diff below
    # misattributes that commit's lines. Flag it rather than fix it.
    idxs = [c.topo_idx for c in commits]
    non_contiguous = max(idxs) - min(idxs) + 1 != len(idxs)

    # git log lists newest first, so the topo-oldest commit has the highest index.
    oldest = max(commits, key=lambda c: c.topo_idx)
    newest = min(commits, key=lambda c: c.topo_idx)
    base = f"{oldest.sha}^" if oldest.parents else EMPTY_TREE
    added = removed = 0
    for line in git("diff", "-w", "--numstat", base, newest.sha, "--", path).splitlines():
        a, r, _ = line.split("\t", 2)
        if a != "-":  # numstat reports "-" counts for binary files
            added += int(a)
            removed += int(r)
    return Session(commits=commits, added=added, removed=removed, non_contiguous=non_contiguous)


def local_date(ts: int) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def date_range(session: Session) -> str:
    start, end = local_date(session.start_ts), local_date(session.end_ts)
    return start if start == end else f"{start} → {end}"


def render_text(
    path: str,
    sessions: list[Session],
    max_gap_hours: float,
    followed: bool,
    min_lines: int,
    n_omitted: int,
) -> str:
    def render_row(num: int, session: Session) -> list[str]:
        return [
            str(num),
            date_range(session),
            str(len(session.commits)),
            f"+{format_count(session.added)}",
            f"-{format_count(session.removed)}",
            "non-contiguous" if session.non_contiguous else "",
        ]

    headers = ["#", "date range", "commits", "+added", "-removed", "note"]
    align = ["right", "left", "right", "right", "right", "left"]
    rows = [render_row(i, session) for i, session in enumerate(sessions, 1)]

    n_commits = sum(len(s.commits) for s in sessions)
    notes = f"max gap {max_gap_hours:g}h"
    if followed:
        notes += ", following renames"
    if n_omitted:
        notes += f", {n_omitted} sessions under {min_lines} changed lines omitted"
    total_added = sum(s.added for s in sessions)
    total_removed = sum(s.removed for s in sessions)

    return "\n".join([
        f"{path} — {n_commits} commits, {len(sessions)} sessions ({notes})",
        "",
        md_table(headers, rows, align=align) if rows else "(no sessions)",
        "",
        f"total: +{format_count(total_added)} / -{format_count(total_removed)}",
    ])


def render_json(
    path: str, sessions: list[Session], max_gap_hours: float, followed: bool,
    min_lines: int, n_omitted: int,
) -> str:
    def iso(ts: int) -> str:
        return datetime.fromtimestamp(ts).isoformat(timespec="seconds")

    data = {
        "path": path,
        "max_gap_hours": max_gap_hours,
        "min_lines": min_lines,
        "n_sessions_omitted": n_omitted,
        "followed_renames": followed,
        "n_commits": sum(len(s.commits) for s in sessions),
        "n_sessions": len(sessions),
        "total_added": sum(s.added for s in sessions),
        "total_removed": sum(s.removed for s in sessions),
        "sessions": [
            {
                "start": iso(s.start_ts),
                "end": iso(s.end_ts),
                "n_commits": len(s.commits),
                "added": s.added,
                "removed": s.removed,
                "non_contiguous": s.non_contiguous,
                "commits": [
                    {"sha": c.sha, "author_date": iso(c.author_ts), "subject": c.subject}
                    for c in s.commits
                ],
            }
            for s in sessions
        ],
    }
    return json.dumps(data, indent=2, ensure_ascii=False)


def render_gaps(commits: list[Commit], max_gap_hours: float) -> str:
    timestamps = sorted(c.author_ts for c in commits)
    gaps = sorted((b - a) / 3600 for a, b in zip(timestamps, timestamps[1:]))
    if not gaps:
        return f"{len(commits)} commit, no gaps to show"
    parts = [f"{g:.1f}" for g in gaps]
    parts.insert(sum(1 for g in gaps if g <= max_gap_hours), "|")
    body = textwrap.fill(
        "  ".join(parts), width=100, initial_indent="  ", subsequent_indent="  "
    )
    return (
        f"{len(commits)} commits, {len(gaps)} gaps, sorted (hours);"
        f" | marks max-gap {max_gap_hours:g}h:\n{body}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-session edit statistics for a file or directory, from git history."
    )
    parser.add_argument("path", help="file or directory to analyze")
    parser.add_argument(
        "--max-gap", type=float, default=6.0, metavar="HOURS",
        help="commits with author dates within this gap share a session (default: 6)",
    )
    parser.add_argument(
        "--min-lines", type=int, default=0, metavar="N",
        help="omit sessions with fewer than N changed lines (added + removed)",
    )
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument(
        "--debug-gaps", action="store_true",
        help="print the inter-commit gap distribution instead of the report (for tuning --max-gap)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        commits, followed = collect_commits(args.path)
        if not commits:
            sys.exit(f"no commits found touching {args.path!r}")
        if args.debug_gaps:
            print(render_gaps(commits, args.max_gap))
            return
        sessions = [build_session(b, args.path) for b in bucket_sessions(commits, args.max_gap)]
    except subprocess.CalledProcessError as e:
        sys.exit(e.stderr.strip() or f"git failed with exit code {e.returncode}")
    kept = [s for s in sessions if s.added + s.removed >= args.min_lines]
    render = render_json if args.format == "json" else render_text
    print(render(args.path, kept, args.max_gap, followed, args.min_lines, len(sessions) - len(kept)))


if __name__ == "__main__":
    main()
