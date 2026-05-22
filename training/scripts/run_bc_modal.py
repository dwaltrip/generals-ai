"""Modal cloud entry point for BC training.

Mirrors `run_bc_local.py` in shape — same `build_arg_parser` /
`config_from_args` pipeline, just with cloud-specific path defaults
(Volume mounts) and a `--modal-gpu` flag added by this wrapper.

The shared training parser (`bc.train_cli.build_arg_parser`) stays the
single source of truth for training flags. Cloud-only flags are
`add_argument`'d to the *same* parser by this wrapper; collisions with
training flags raise `argparse.ArgumentError` at add-time.

Run:
    uv run modal run training/scripts/run_bc_modal.py::train \\
        --modal-gpu T4 --max-batches 1 --epochs 1
"""

from __future__ import annotations

from pathlib import Path

import modal

from bc.train_cli import build_arg_parser, config_from_args


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRAINING_REQS = REPO_ROOT / "training" / "modal_requirements.txt"

image = (
    modal.Image.debian_slim(python_version="3.14")
    .uv_pip_install(requirements=[str(TRAINING_REQS)])
    .add_local_python_source("bc", "shared", "utils")
)

app = modal.App("bc-train", image=image)

parsed_replays_vol = modal.Volume.from_name("generals-ai.parsed-replays")
training_runs_vol = modal.Volume.from_name("generals-ai.training-runs")


@app.local_entrypoint()
def train(*arglist: str) -> None:
    """Parse cloud + training args, build a config, print it.

    Step-4 scaffold: validates the extended-parser pattern end-to-end
    (cloud defaults, run-id generation, config construction). The
    `@app.function` that actually runs `bc_run(config)` on a GPU is the
    next piece — to be wired in once we agree on its shape.
    """
    parser = build_arg_parser()
    parser.add_argument(
        "--modal-gpu",
        default="T4",
        help="Modal GPU class (e.g. T4, A100, H100). Cloud-only.",
    )
    parser.set_defaults(
        intermediate=Path("/data/intermediate"),
        manifest=Path("/data/probe_500.json"),
        out_dir=Path("/runs"),
        device="cuda",
    )
    args = parser.parse_args(arglist)
    config = config_from_args(args)

    print(f"modal-gpu: {args.modal_gpu}")
    print(f"run_dir:   {config.run_dir}")
    print(f"config:    {config}")
