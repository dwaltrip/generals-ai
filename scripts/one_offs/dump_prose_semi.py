#!/usr/bin/env python3
"""Dump semicolon-chain candidates in Python prose (comments + docstrings).

Two signals:
  [single] one prose line with 2+ semicolons
  [pair]   two consecutive prose lines that each contain a semicolon

Reads file paths on stdin, writes a human-scannable report to the path given
as argv[1]. Prose = real comments (tokenize) + real docstrings (ast); triple-
quoted string literals (SQL/CSS/data) are excluded.
"""
import ast
import io
import sys
import tokenize
from pathlib import Path

CTX = 1  # context lines on each side of a hit region


def prose_lines(src: str) -> set[int]:
    lines: set[int] = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                lines.add(tok.start[0])
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return lines
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef,
                                 ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            for ln in range(first.lineno, first.end_lineno + 1):
                lines.add(ln)
    return lines


def main() -> None:
    out_path = Path(sys.argv[1])
    paths = [Path(p) for p in sys.stdin.read().split()]
    chunks: list[str] = []
    n_files = 0
    n_single = 0
    n_pair = 0

    for path in paths:
        try:
            src = path.read_text()
        except OSError:
            continue
        text = src.splitlines()
        prose = prose_lines(src)

        def semi(ln: int) -> bool:
            return 0 < ln <= len(text) and ";" in text[ln - 1]

        def multi(ln: int) -> bool:
            return 0 < ln <= len(text) and text[ln - 1].count(";") >= 2

        flagged: dict[int, str] = {}  # lineno -> tag
        for ln in sorted(prose):
            tags = []
            if multi(ln):
                tags.append("single")
            if (ln + 1) in prose and semi(ln) and semi(ln + 1):
                tags.append("pair")
                flagged.setdefault(ln + 1, "pair-cont")
            if tags:
                flagged[ln] = "+".join(tags)

        if not flagged:
            continue
        n_files += 1

        # Build context regions, merging nearby flags.
        flag_lines = sorted(flagged)
        regions: list[list[int]] = []
        for ln in flag_lines:
            lo, hi = ln - CTX, ln + CTX
            if regions and lo <= regions[-1][1] + 1:
                regions[-1][1] = max(regions[-1][1], hi)
            else:
                regions.append([lo, hi])

        chunks.append(f"\n## {path}")
        for lo, hi in regions:
            lo = max(1, lo)
            hi = min(len(text), hi)
            chunks.append("")
            for ln in range(lo, hi + 1):
                tag = flagged.get(ln, "")
                mark = f"  <-- {tag}" if tag else ""
                chunks.append(f"{ln:4d} | {text[ln - 1]}{mark}")
                if tag == "single":
                    n_single += 1
                elif tag and tag != "pair-cont":
                    if "single" in tag:
                        n_single += 1
                    if "pair" in tag:
                        n_pair += 1
                elif tag == "pair":
                    n_pair += 1

    header = (
        "# Semicolon-chain candidates in Python prose (comments + docstrings)\n"
        "\n"
        "Signals: `single` = one prose line with 2+ semicolons; "
        "`pair` = start of two consecutive prose lines that each have a "
        "semicolon (`pair-cont` marks the second line).\n"
        f"\nFiles with hits: {n_files}\n"
    )
    out_path.write_text(header + "\n".join(chunks) + "\n")
    print(f"wrote {out_path}  ({n_files} files)")


if __name__ == "__main__":
    main()
