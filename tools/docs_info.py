#!/usr/bin/env python3
"""docs_info — show mtime + recent commit history for markdown docs.

Run with: `uv run tools/docs_info.py [root] [options]`
"""

import argparse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import subprocess
import sys

from tabulate import tabulate

from utils.docstring import doc_summary


COMMIT_MARK = "===COMMIT==="


@dataclass
class Commit:
    short_hash: str
    date: str
    added: int      # -1 sentinel = binary / unknown
    removed: int
    subject: str


@dataclass
class DocInfo:
    path: Path
    mtime: datetime
    commits: list[Commit] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=doc_summary(__doc__))
    p.add_argument("root", nargs="?", default="docs", type=Path,
                   help="folder to search (default: ./docs)")
    p.add_argument("-i", "--include", default="",
                   help="comma-separated substrings; keep if any matches the path")
    p.add_argument("-e", "--exclude", default="",
                   help="comma-separated substrings; drop if any matches (wins over --include)")
    p.add_argument("-c", "--commits", type=int, default=1,
                   help="commits to show per doc (default: 1)")
    p.add_argument("-l", "--limit", type=int, default=20,
                   help="show most-recently-modified N docs (default: 20)")
    p.add_argument("-a", "--all", action="store_true",
                   help="show all docs; overrides --limit")
    return p.parse_args()


def keep(path: Path, includes: list[str], excludes: list[str]) -> bool:
    name = str(path)
    if excludes and any(e in name for e in excludes):
        return False
    if includes and not any(i in name for i in includes):
        return False
    return True


def git_log_for_file(path: Path, n: int) -> list[Commit]:
    out = subprocess.run(
        ["git", "log",
         f"-n{n}",
         "--follow",
         "--numstat",
         f"--format={COMMIT_MARK}%h|%ad|%s",
         "--date=short",
         "--", str(path)],
        capture_output=True, text=True, check=False,
    )
    if out.returncode != 0 or not out.stdout.strip():
        return []

    commits: list[Commit] = []
    cur: Commit | None = None
    for line in out.stdout.splitlines():
        if line.startswith(COMMIT_MARK):
            if cur is not None:
                commits.append(cur)
            h, d, s = line[len(COMMIT_MARK):].split("|", 2)
            cur = Commit(short_hash=h, date=d, added=0, removed=0, subject=s)
        elif line.strip() and cur is not None:
            parts = line.split("\t", 2)
            if len(parts) < 3:
                continue
            a, r, _ = parts
            cur.added = -1 if a == "-" else int(a)
            cur.removed = -1 if r == "-" else int(r)
    if cur is not None:
        commits.append(cur)
    return commits


def build_rows(infos: list[DocInfo], root: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    for d in infos:
        try:
            rel = str(d.path.relative_to(root))
        except ValueError:
            rel = str(d.path)
        mtime = d.mtime.strftime("%Y-%m-%d")
        if not d.commits:
            rows.append([rel, mtime, "(untracked)", "", "", "", ""])
            continue
        for i, c in enumerate(d.commits):
            rows.append([
                rel if i == 0 else "",
                mtime if i == 0 else "",
                c.short_hash,
                c.date,
                f"+{c.added}" if c.added >= 0 else "-",
                f"-{c.removed}" if c.removed >= 0 else "-",
                c.subject if len(c.subject) <= 60 else c.subject[:57] + "...",
            ])
    return rows


def main() -> int:
    args = parse_args()
    root: Path = args.root

    if not root.exists():
        print(f"error: {root} does not exist", file=sys.stderr)
        return 2
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 2

    includes = [s for s in args.include.split(",") if s]
    excludes = [s for s in args.exclude.split(",") if s]

    md_files = sorted(p for p in root.rglob("*.md") if keep(p, includes, excludes))
    if not md_files:
        print(f"no markdown files found under {root}")
        return 0

    infos = [
        DocInfo(
            path=p,
            mtime=datetime.fromtimestamp(p.stat().st_mtime),
            commits=git_log_for_file(p, args.commits) if args.commits > 0 else [],
        )
        for p in md_files
    ]
    infos.sort(key=lambda d: d.mtime, reverse=True)
    total = len(infos)
    truncated = not args.all and total > args.limit
    if not args.all:
        infos = infos[: args.limit]

    if truncated:
        print(
            f"showing {len(infos)} of {total} docs (most recently modified);",
            "pass --all to show all",
        )
        print()

    headers = ["Doc", "mtime", "Commit", "Date", "+add", "-del", "Subject"]
    rows = build_rows(infos, root)
    print(tabulate(
        rows,
        headers=headers,
        tablefmt="github",
        colalign=("left", "left", "left", "left", "right", "right", "left"),
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
