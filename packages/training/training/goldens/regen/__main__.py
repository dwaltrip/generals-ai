"""
Produce and write golden references for all registry entries.
Every fixture is computed and checked before anything is written.

Run from packages/training:
    uv run python -m training.goldens.regen
"""

from __future__ import annotations

import sys

from training.goldens.regen.apply import apply
from training.goldens.regen.plan import RegenAbort, assemble, plan_fixture
from training.goldens.regen.report import render_file_lines
from training.goldens.registry import FIXTURES


def main() -> int:
    parts = []
    try:
        for fx in FIXTURES:
            part = plan_fixture(fx)
            info = part.info
            print(f"planned {fx.id}  T={info.T} end_t={info.end_t}  ({fx.note})")
            parts.append(part)
        plan = assemble(parts)
    except RegenAbort as e:
        print(f"\n{e}\nNothing written.")
        return 1

    apply(plan)
    print()
    for line in render_file_lines(plan):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
