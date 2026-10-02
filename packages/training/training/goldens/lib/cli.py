"""The regen command line. A project calls `regen_main` from its own entry point."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from training.goldens.lib.registry import Fixture, Registry
from training.goldens.lib.report import render_regen_report
from training.goldens.lib.run import run_regen


# `describe_fixture` adds project-specific detail to each fixture's progress line.
def regen_main[FixtureT: Fixture, DataT](
    registry: Registry[FixtureT, DataT],
    root: Path,
    *,
    describe_fixture: Callable[[FixtureT, DataT], str] | None = None,
    argv: list[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(prog="regen")
    parser.add_argument("--dry-run", action="store_true", help="plan and report, write nothing")
    args = parser.parse_args(argv)

    def on_fixture(fixture: FixtureT, data: DataT) -> None:
        line = f"computed {fixture.id}"
        if describe_fixture is not None:
            line += f"  {describe_fixture(fixture, data)}"
        print(line)

    # TODO: Nicer handling of regen errors (invalid references, case-only renames),
    # and a better UI for them in the TUI. For now they surface as tracebacks.
    plan = run_regen(registry, root, write=not args.dry_run, on_fixture=on_fixture)

    print()
    print(render_regen_report(plan), end="")
    if args.dry_run:
        print("\ndry run: nothing written")
    return 0
