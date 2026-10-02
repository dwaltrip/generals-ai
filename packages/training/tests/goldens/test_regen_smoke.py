"""
The full regen pipeline on the project's registry, into an empty tree. Covers the
wiring that the golden test doesn't use: compute_all, planning, apply, and the
report on real content.
"""

from pathlib import Path

from training.goldens.lib import (
    apply_regen,
    compute_all,
    entries,
    fixture_infos,
    plan_regen,
    render_regen_report,
    store,
)
from training.goldens.registry import REGISTRY


def test_regen_into_empty_tree(tmp_path: Path) -> None:
    computed = compute_all(REGISTRY)
    fixtures = fixture_infos(REGISTRY)

    plan = plan_regen(computed, fixtures, tmp_path)
    render_regen_report(plan)
    apply_regen(plan)

    surfaces = [s.name for s in REGISTRY.surfaces]
    assert len(store.load_references(tmp_path, surfaces).references) == len(entries(REGISTRY))

    replan = plan_regen(computed, fixtures, tmp_path)
    render_regen_report(replan)
    assert replan.is_noop()
