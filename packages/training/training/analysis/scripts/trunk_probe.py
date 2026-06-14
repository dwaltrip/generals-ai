#!/usr/bin/env -S uv run python
"""Generic frozen-representation probe CLI — runs any registered `ProbeTask`.

The reusable flow lives in `training/analysis/probes/cli.py`; one-off experiment
tasks live in `training/analysis/probe_runs/` (each its own runnable script). This
entry point covers the registry-backed tasks in `probes/tasks.py`.

    trunk_probe.py --task lowest_army --source trunk --checkpoint <ckpt> --manifest <split>
"""

from __future__ import annotations

from training.analysis.probes.cli import build_probe_arg_parser, run_probe_cli
from training.analysis.probes.tasks import TASKS
from utils.docstring import doc_summary


def main() -> None:
    parser = build_probe_arg_parser(doc_summary(__doc__))
    parser.add_argument("--task", choices=list(TASKS), required=True)
    args = parser.parse_args()
    run_probe_cli(lambda cfg: TASKS[args.task](dense_history_n=cfg.dense_history_n), args)


if __name__ == "__main__":
    main()
