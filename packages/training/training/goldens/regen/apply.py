"""Apply the regen plan. Here we update the on-disk references."""

from __future__ import annotations

from training.goldens import store
from training.goldens.regen.plan import Plan, Status


def apply(plan: Plan) -> None:
    root = plan.root
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
