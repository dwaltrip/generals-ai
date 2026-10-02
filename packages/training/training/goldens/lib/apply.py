from __future__ import annotations

from training.goldens.lib import store
from training.goldens.lib.entry import Changed, Moved, New, Unchanged
from training.goldens.lib.plan import RegenPlan


def apply_regen(plan: RegenPlan) -> None:
    root = plan.tree.root
    for entry in plan.planned_entries():
        match entry.status:
            case Unchanged():
                pass
            case New() | Changed():
                store.save(entry.id, entry.content, root)
            case Moved(source=source):
                store.save(entry.id, entry.content, root)
                store.remove(source, root)

    for surface in plan.surfaces:
        for eid in surface.planned_removals:
            store.remove(eid, root)
