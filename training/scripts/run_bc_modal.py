"""Modal cloud entry point for BC training.

Mirrors `run_bc_local.py` in shape — same `build_arg_parser` /
`config_from_args` pipeline, plus a `--modal-gpu` cloud-only flag that
selects the Modal GPU class. The shared training parser remains the
single source of truth for training flags; cloud-only flags are
`add_argument`'d to the same parser by this wrapper, so collisions with
training flags raise `argparse.ArgumentError` at add-time.

Run:
    uv run modal run training/scripts/run_bc_modal.py \\
        --modal-gpu T4 --max-batches 5 --epochs 2
"""

from __future__ import annotations

import json
import platform
import socket
from datetime import datetime, timezone
from pathlib import Path

import modal

from bc.train_cli import build_arg_parser, config_from_args
from bc.train_config import TrainConfig


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


@app.function(
    gpu="T4",  # default; overridden via `with_options(gpu=...)` per-call
    volumes={
        "/data": parsed_replays_vol,
        "/runs": training_runs_vol,
    },
    timeout=60 * 60 * 6,
)
def train_remote(config: TrainConfig, modal_gpu: str) -> None:
    """Run `bc_run` on a Modal GPU.

    Initializes the run dir, drops `args_cloud.json` *before* training
    starts (so cloud-side provenance is captured even if training
    raises), then hands off to `bc_run`. `args_cloud.json` sits next to
    bc's `args.json` and captures what the training contract doesn't
    know about: which GPU class the operator requested, which device
    CUDA actually surfaced, container hostname.
    """
    from bc.train import bc_run, initialize_run_dir

    initialize_run_dir(config)
    _write_args_cloud(config.run_dir, modal_gpu)
    bc_run(config)


def _write_args_cloud(run_dir: Path, modal_gpu: str) -> None:
    """Write a `args_cloud.json` sibling to bc's `args.json`."""
    import torch

    cuda_device_name: str | None = None
    if torch.cuda.is_available():
        cuda_device_name = torch.cuda.get_device_name(0)

    args_cloud = {
        "modal_gpu": modal_gpu,
        "cuda_device_name": cuda_device_name,
        "python": platform.python_version(),
        "hostname": socket.gethostname(),
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with (run_dir / "args_cloud.json").open("w") as fp:
        json.dump(args_cloud, fp, indent=2)


@app.local_entrypoint()
def train(*arglist: str) -> None:
    """Parse cloud + training args, run `bc_run` on a Modal GPU."""
    parser = build_arg_parser()
    parser.add_argument(
        "--modal-gpu",
        required=True,
        help="Modal GPU class (e.g. T4, A100, A100-80GB, H100, L4, L40S). Cloud-only.",
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
    print()

    # Volume-relative path (no `/runs/` prefix — that's the container mount).
    run_id = config.run_dir.name
    print()
    print("Once the run is done, pull artifacts:")
    print(f"  uv run modal volume get generals-ai.training-runs /{run_id} training/data/runs-cloud")

    train_remote.with_options(gpu=args.modal_gpu).spawn(config, args.modal_gpu)

    print()
    print("Run spawned.")
