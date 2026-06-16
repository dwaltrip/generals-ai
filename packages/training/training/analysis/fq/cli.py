"""fq CLI — build a family table and run an ad-hoc numpy snippet against it.

    uv run python -m training.analysis.fq.cli --table elim_head_debug --split val \\
        -e "t.cols['army_sim'][t.cols['alive']].mean()"

    uv run python -m training.analysis.fq.cli --table elim_head_debug <<'PY'
    imm = select(t, (t.cols['dt'] >= 0) & (t.cols['dt'] < 10))
    print(imm.n_games, np.median(imm.cols['margin'][imm.cols['margin'] >= 0]))
    PY

`-e` prints its expression's value; a stdin snippet self-prints. The snippet runs
with `t` (the built table, derived columns attached), `np`, `select`, `join_dump`,
`truth_map` (the family's shared-truth map for `join_dump`), `check_representative`,
and the family's derived-column helpers in scope.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

from settings import TRAINING_DATA_DIR
from training.analysis.fq.families import REGISTRY
from training.analysis.fq.frame_table import (
    build_frame_table,
    check_representative,
    join_dump,
    select,
)
from training.bc.obs_config import OBS_CONFIG_DEFAULTS, ObsConfig
from training.bc.splits import load_manifest, samples_for_split


DEFAULT_MANIFEST = TRAINING_DATA_DIR / "splits" / "cloud_24k_sub_11k.json"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--table", required=True, choices=sorted(REGISTRY), help="family name")
    ap.add_argument("--split", default="val", choices=("train", "val"))
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--max-games", type=int, default=None, help="cap to first N games")
    ap.add_argument("-e", "--expr", help="expression to print; omit to read a snippet from stdin")
    args = ap.parse_args()

    spec = REGISTRY[args.table]
    man = load_manifest(args.manifest)
    samples = samples_for_split(man, args.split, Path(man["intermediate_root"]))
    # Config-only obs (ground-truth tables): fp32 to avoid fp16 storage noise.
    obs_cfg = ObsConfig(dense_history_n=OBS_CONFIG_DEFAULTS.dense_history_n, obs_dtype="fp32")

    t = build_frame_table(spec, samples, obs_cfg, args.max_games)
    for name, fn in spec.derived_cols.items():
        t.cols[name] = fn(t)
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


if __name__ == "__main__":
    main()
