"""
The write step: save what the plan says to save, remove what it says to remove.
Nothing else on the tree is touched.
"""

from __future__ import annotations

from pathlib import Path

from training.goldens import store
from training.goldens.paths import REFERENCES_DIR
from training.goldens.regen.plan import Plan, Status


def apply(plan: Plan, root: Path = REFERENCES_DIR) -> None:
    for o in plan.obs:
        if o.status is not Status.UNCHANGED:
            store.save_obs(o.ref, o.digest, root)
    for s in plan.supervision:
        if s.status is not Status.UNCHANGED:
            store.save_supervision(s.ref, s.array, root)
    moved_from = [i.moved_from for i in [*plan.obs, *plan.supervision] if i.status is Status.MOVED]
    for rid in [*plan.removed, *moved_from]:
        assert rid is not None
        store.remove(rid, root)
