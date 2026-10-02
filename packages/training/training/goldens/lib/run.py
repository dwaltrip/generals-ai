from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from training.goldens.lib.apply import apply_regen
from training.goldens.lib.compute import compute_all
from training.goldens.lib.plan import RegenPlan, plan_regen
from training.goldens.lib.registry import Fixture, Registry, fixture_infos


def run_regen[FixtureT: Fixture, DataT](
    registry: Registry[FixtureT, DataT],
    root: Path,
    *,
    write: bool,
    on_fixture: Callable[[FixtureT, DataT], None] | None = None,
) -> RegenPlan:
    computed = compute_all(registry, on_fixture=on_fixture)
    regen_plan = plan_regen(computed, fixture_infos(registry), root)
    if write:
        apply_regen(regen_plan)
    return regen_plan
