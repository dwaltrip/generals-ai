"""
Produce and write golden references for all registry entries.
Every fixture is computed and checked before anything is written.

Run from packages/training:
    uv run python -m training.goldens.regen [--dry-run]
"""

from __future__ import annotations

import argparse
import sys

from training.goldens.paths import REFERENCES_DIR
from training.goldens.regen.apply import apply
from training.goldens.regen.plan import RegenAbort, assemble, plan_fixture
from training.goldens.regen.report import render_report
from training.goldens.registry import FIXTURES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="regen")
    parser.add_argument("--dry-run", action="store_true", help="plan and report, write nothing")
    args = parser.parse_args(argv)
    root = REFERENCES_DIR

    parts = []
    try:
        for fx in FIXTURES:
            part = plan_fixture(fx, root)
            info = part.info
            print(f"planned {fx.id}  T={info.T} end_t={info.end_t}  ({fx.note})")
            parts.append(part)
        plan = assemble(parts, root)
    except RegenAbort as e:
        print(f"\n{e}\nNothing written.")
        return 1

    if not args.dry_run:
        apply(plan, root)
    print()
    print(render_report(plan, root), end="")
    if args.dry_run:
        print("\ndry run: nothing written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
