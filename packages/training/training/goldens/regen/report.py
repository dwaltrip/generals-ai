"""
Renderers over a Plan. Pure functions from the plan to text.
"""

from __future__ import annotations

from training.goldens import store
from training.goldens.regen.plan import Plan, Status
from training.goldens.store import RefId


# TODO: interim output, one line per file. Replaced by the regen report (9.18-1
# section 8) in the next chunk.
def render_file_lines(plan: Plan) -> list[str]:
    lines = []
    for item in plan.obs:
        lines.append(_line(item.status, item.ref, item.diff, item.moved_from))
    for item in plan.supervision:
        line = _line(item.status, item.ref, item.diff, item.moved_from)
        lines.append(f"{line}   <- {', '.join(item.points)}")
    for rid in plan.removed:
        lines.append(f"  {'removed':12s} {store.rel(rid)}")
    for path in plan.unrecognized:
        lines.append(f"  {'removed':12s} {path.name} (unrecognized file)")
    lines += plan.warnings
    return lines


def _line(status: Status, ref: RefId, diff, moved_from: RefId | None) -> str:
    label = status.value
    if status is Status.CHANGED and diff is not None:
        label = f"changed ({diff.summary()})"
    line = f"  {label:12s} {store.rel(ref)}"
    if moved_from is not None:
        line += f"   (from {store.rel(moved_from)})"
    return line
