"""fq's generic CLI: build a table and run an ad-hoc numpy snippet on it.

`build_cli(registry, default_manifest)` returns a `main()` an entry point calls;
the domain (the table registry, the default split) is injected, so this module
stays free of any specific spec. See `scripts/fq.py` for the entry.

`-e` prints its expression's value; a stdin snippet self-prints. The snippet runs
with `t` (the built table, derived columns attached), `np`, `select`, `join_dump`,
`truth_map` (the spec's shared-truth map for `join_dump`), `check_representative`,
and the spec's derived-column helpers in scope.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
import sys

import numpy as np

from training.analysis.fq.frame_table import (
    GROUND_TRUTH_OBS_CFG,
    build_frame_table,
    check_representative,
    join_dump,
    select,
)
from training.analysis.fq.registry import TableRegistry
from training.bc.splits import load_manifest, samples_for_split


def build_cli(registry: TableRegistry, *, default_manifest: Path) -> Callable[[], None]:
    """Return a `main()` bound to this table `registry` and default split."""

    def main() -> None:
        ap = argparse.ArgumentParser(
            description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
        )
        ap.add_argument("--table", required=True, choices=registry.names(), help="table name")
        ap.add_argument("--split", default="val", choices=("train", "val"))
        ap.add_argument("--manifest", type=Path, default=default_manifest)
        ap.add_argument("--max-games", type=int, default=None, help="cap to first N games")
        ap.add_argument(
            "-e", "--expr", help="expression to print; omit to read a snippet from stdin"
        )
        args = ap.parse_args()

        spec = registry[args.table]
        man = load_manifest(args.manifest)
        samples = samples_for_split(man, args.split, Path(man["intermediate_root"]))
        t = build_frame_table(spec, samples, GROUND_TRUTH_OBS_CFG, args.max_games)
        print(
            f"built '{spec.name}' [{args.split}]: {t.frame_t.size} frames / {t.n_games} games  "
            f"cols={list(t.cols)}",
            file=sys.stderr,
        )

        ns: dict = {
            "t": t,
            "np": np,
            "select": select,
            "join_dump": join_dump,
            "truth_map": spec.truth_map,
            "check_representative": check_representative,
            **{fn.__name__: fn for fn in spec.derived_cols.values()},
        }
        if args.expr:
            print(eval(args.expr, ns))  # noqa: S307 — deliberate ad-hoc eval surface
        else:
            exec(sys.stdin.read(), ns)  # noqa: S102 — deliberate ad-hoc exec surface

    return main
