"""Modal cloud entry point for BC training.

Mirrors `run_bc_local.py` in shape — same `build_arg_parser` / `--config`
pipeline. The GPU class is sourced from the run config (`config.gpu`, set in the
`--config` file), so there's no separate `--modal-gpu` flag: each sweep cell
declares the card it needs in its config. The local entrypoint reads it to apply
`train_remote.with_options(gpu=...)`; it also rides into the remote as recorded
provenance.

Run with `--detach` so the spawned training survives the local process —
without it, Modal stops the ephemeral app when the local entrypoint returns
and cancels the in-flight run (the function uses `.spawn()`, fire-and-forget):
    uv run modal run --detach training/scripts/run_bc_modal.py \\
        --config training/configs/my_run.json --max-batches 5
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import platform
import socket

import modal

from bc.train_cli import (
    build_arg_parser,
    config_from_args,
    load_config_overlay,
    operational_overrides,
    resume_run_dir,
)
from bc.train_config import TrainConfig


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRAINING_REQS = REPO_ROOT / "training" / "modal_requirements.txt"

image = (
    modal.Image.debian_slim(python_version="3.14")
    .uv_pip_install(requirements=[str(TRAINING_REQS)])
    .add_local_python_source("bc", "shared", "utils", "game_types")
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
    timeout=60 * 60 * 12,
)
def train_remote(
    fresh_config: TrainConfig | None,
    config_input_text: str | None,
    resume_run_dir: str | None,
    overlay: dict,
    operational: dict,
    gpu: str,
    force_config_mismatch: bool,
    legacy_lr_warmup_batches: int | None,
) -> None:
    """Run a fresh or resumed BC training segment on a Modal GPU.

    Fresh (`fresh_config` set): initialize the run dir — writing `args.json`,
    the verbatim `config.input.json`, and `args_cloud.json` — *before* training
    starts (so cloud-side provenance is captured even if training raises), then
    hand off to `bc_run`. Resume (`resume_run_dir` set): the effective config is
    resolved remotely against the parent's `args.json` on the Volume; write the
    segment-suffixed `args_cloud_resume_NN.json`, then `bc_resume`.

    `args_cloud*.json` sits next to bc's args file and captures what the training
    contract doesn't know: the GPU class, the device CUDA surfaced, the hostname.
    """
    from bc.resume import bc_resume
    from bc.run_dir import initialize_run_dir, prepare_resume
    from bc.train import bc_run

    if resume_run_dir is not None:
        # Compute the suffix once so the cloud-only provenance lands on the
        # same segment bc_resume writes; pass `info` through so bc_resume
        # doesn't recompute (and double-count) the suffix.
        run_dir = Path(resume_run_dir)
        info = prepare_resume(run_dir)
        _write_args_cloud(run_dir, gpu, suffix=info.next_suffix)
        bc_resume(run_dir, info, overlay, operational, force_config_mismatch,
                  legacy_lr_warmup_batches)
    else:
        assert fresh_config is not None
        initialize_run_dir(fresh_config, config_input_text)
        _write_args_cloud(fresh_config.run_dir, gpu)
        bc_run(fresh_config)


def _write_args_cloud(run_dir: Path, gpu: str, suffix: str = "") -> None:
    """Write an `args_cloud{suffix}.json` sibling to bc's args file."""
    import torch

    cuda_device_name: str | None = None
    if torch.cuda.is_available():
        cuda_device_name = torch.cuda.get_device_name(0)

    args_cloud = {
        "gpu": gpu,
        "cuda_device_name": cuda_device_name,
        "python": platform.python_version(),
        "hostname": socket.gethostname(),
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    with (run_dir / f"args_cloud{suffix}.json").open("x") as fp:
        json.dump(args_cloud, fp, indent=2)


@app.local_entrypoint()
def train(*arglist: str) -> None:
    """Parse args, resolve the GPU class from the config, spawn on a Modal GPU."""
    parser = build_arg_parser()
    parser.set_defaults(out_dir=Path("/runs"))
    args = parser.parse_args(arglist)

    if args.resume:
        run_dir = resume_run_dir(args)
        overlay = load_config_overlay(args)
        # Card to provision. `with_options(gpu=...)` is decided *locally* before
        # the remote can read the parent run's config off the Volume, so the
        # parent's card is invisible here — only the overlay is. Rather than
        # silently default (a T4 would OOM a resumed wide run — pyramid+bs=512
        # doesn't fit T4's 16 GB), require the card to be restated in the resume
        # overlay. It may still differ from the parent (gpu is free to change on
        # resume); it just can't be implicit.
        if "gpu" not in overlay:
            raise SystemExit(
                "--resume requires a `gpu` field in the --config overlay: the "
                "card is provisioned locally before the remote can read the "
                "parent run's config, so it can't be inferred. Restate it "
                '(e.g. {"gpu": "L4"}) — it may differ from the parent run\'s card.'
            )
        gpu = overlay["gpu"]
        fresh_config: TrainConfig | None = None
        config_input_text: str | None = None
        operational = operational_overrides(args)
        resume_run_dir_arg: str | None = str(run_dir)
    else:
        fresh_config = config_from_args(args)
        gpu = fresh_config.gpu
        config_input_text = Path(args.config).read_text()
        overlay = {}
        operational = {}
        resume_run_dir_arg = None
        run_dir = fresh_config.run_dir

    print(f"gpu:     {gpu}")
    print(f"run_dir: {run_dir}")
    print()
    print("Once the run is done, pull artifacts:")
    print(
        "  uv run modal volume get generals-ai.training-runs",
        f"/{run_dir.name} training/data/runs-cloud",
    )

    train_remote.with_options(gpu=gpu).spawn(
        fresh_config, config_input_text, resume_run_dir_arg,
        overlay, operational, gpu, args.force_config_mismatch,
        args.legacy_lr_warmup_batches,
    )

    print()
    print("Run spawned.")
