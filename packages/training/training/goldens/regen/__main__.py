"""
Produce and write golden references for all registry entries.
Every fixture is computed and checked before anything is written.

Run from packages/training:
    uv run python -m training.goldens.regen
"""

from __future__ import annotations

import sys

from training.goldens import store
from training.goldens.regen.plan import Plan, RegenAbort, Status, assemble, plan_fixture
from training.goldens.regen.report import render_file_lines
from training.goldens.registry import FIXTURES


def _apply(plan: Plan) -> None:
    for item in plan.obs:
        store.save_obs(item.ref, item.digest)
    for item in plan.supervision:
        store.save_supervision(item.ref, item.array)
    moved_from = [i.moved_from for i in [*plan.obs, *plan.supervision] if i.status is Status.MOVED]
    for rid in [*plan.removed, *moved_from]:
        assert rid is not None
        store.remove(rid)
    for path in plan.unrecognized:
        path.unlink()


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

    _apply(plan)
    print()
    for line in render_file_lines(plan):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
