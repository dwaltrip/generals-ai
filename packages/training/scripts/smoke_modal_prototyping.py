"""Modal cloud entry point for BC training.

Three smoke functions, none of them does training:

- `smoke` — image smoke. Builds the Modal image with the full training stack
  installed and verifies `bc.train` is importable inside the container.
  No GPU, no Volume.
- `smoke_volume` — inputs-Volume read smoke. Mounts the
  `generals-ai.parsed-replays` Volume RO, opens the bundled manifest,
  resolves the first training sample, and loads its `.npz` + `.meta.npz`.
- `smoke_outputs` — outputs-Volume write smoke. Mounts the
  `generals-ai.training-runs` Volume RW, creates a fresh run dir under
  `/runs/<run_id>/`, and writes a stub `run_metadata.json` + `metrics.jsonl`.
  Pull artifacts back with `modal volume get generals-ai.training-runs ...`.

Real training runs (GPU + actual `bc_run`) land in subsequent commits.

Run:
    uv run modal run training/scripts/run_bc_modal.py                   # smoke
    uv run modal run training/scripts/run_bc_modal.py::smoke_volume     # inputs
    uv run modal run training/scripts/run_bc_modal.py::smoke_outputs    # outputs
"""

from __future__ import annotations

from datetime import UTC
from pathlib import Path

import modal


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRAINING_REQS = REPO_ROOT / "training" / "modal_requirements.txt"

# Two layers, in cache-friendly order:
#   1. pinned external deps — rarely changes, big install (torch + transitives)
#   2. our source packages — change every iteration, mounted at runtime via
#      `add_local_python_source` (no layer rebuild on edit). Modal looks the
#      modules up via Python import, so this works because `bc`, `shared`,
#      and `utils` are installed as workspace members in the local env.
image = (
    modal.Image.debian_slim(python_version="3.14")
    .uv_pip_install(requirements=[str(TRAINING_REQS)])
    .add_local_python_source("bc", "shared", "utils")
)

app = modal.App("bc-train", image=image)

parsed_replays_vol = modal.Volume.from_name("generals-ai.parsed-replays")
training_runs_vol = modal.Volume.from_name("generals-ai.training-runs")


@app.function()
def smoke() -> dict:
    """Step-1 smoke: import the training stack, report container info."""
    import platform
    import sys

    import torch

    # The actual import-chain check — resolves bc + shared (from training)
    # and the workspace cross-dep utils.
    from training.bc.train import bc_run  # noqa: F401
    from training.bc.train_cli import build_arg_parser, config_from_args  # noqa: F401
    from training.bc.train_config import TrainConfig, make_run_id  # noqa: F401

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "imports_ok": True,
    }


@app.function(volumes={"/data": parsed_replays_vol})
def smoke_volume() -> dict:
    """Step-2 smoke: open the manifest from the mounted Volume, load one sample."""
    import numpy as np

    from training.bc.splits import load_manifest, samples_for_split
    from training.bc.utils import meta_path_for

    manifest = load_manifest(Path("/data/probe_500.json"))
    samples = samples_for_split(manifest, "train", Path("/data/intermediate"))
    sim_path, k = samples[0]
    meta_path = meta_path_for(sim_path)

    with np.load(sim_path) as sim:
        sim_keys = {name: (arr.shape, str(arr.dtype)) for name, arr in sim.items()}
    with np.load(meta_path) as meta:
        meta_keys = {name: (arr.shape, str(arr.dtype)) for name, arr in meta.items()}

    result = {
        "manifest_kept_pairs": manifest["kept_pairs"],
        "n_samples_train": len(samples),
        "first_sim_path": str(sim_path),
        "first_perspective_k": k,
        "sim_arrays": sim_keys,
        "meta_arrays": meta_keys,
    }
    print("smoke_volume result:")
    for key, val in result.items():
        print(f"  {key}: {val}")
    return result


@app.function(volumes={"/runs": training_runs_vol})
def smoke_outputs() -> dict:
    """Step-3 smoke: create a fresh run dir on the outputs Volume + write stubs.

    Writes are auto-committed to the Volume on clean function exit, so
    `modal volume get generals-ai.training-runs /<run_id>/ <local-dest>`
    will see them once this returns.
    """
    from datetime import datetime
    import json
    import platform
    import socket

    from training.bc.train_config import make_run_id

    run_id = make_run_id()
    run_dir = Path("/runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    metadata = {
        "run_id": run_id,
        "smoke": True,
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "hostname": socket.gethostname(),
    }
    metadata_path = run_dir / "run_metadata.json"
    with metadata_path.open("w") as fp:
        json.dump(metadata, fp, indent=2)

    metrics_path = run_dir / "metrics.jsonl"
    with metrics_path.open("w") as fp:
        fp.write(json.dumps({"epoch": 1, "loss": 1.234, "smoke": True}) + "\n")

    # The pull-back target directory must exist beforehand and the remote
    # path must NOT have a trailing slash; otherwise `modal volume get`
    # collapses recursive contents into a single output file (the last
    # write wins). See `docs/working-with-modal-cloud-gpu.md`.
    result = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "metadata": metadata,
        "files_written": [str(metadata_path), str(metrics_path)],
        "pull_back_cmd": (
            "mkdir -p tmp/cloud-smoke-outputs && "
            f"modal volume get generals-ai.training-runs /{run_id} "
            "tmp/cloud-smoke-outputs"
        ),
    }
    print("smoke_outputs result:")
    for key, val in result.items():
        print(f"  {key}: {val}")
    return result


@app.local_entrypoint()
def main() -> None:
    result = smoke.remote()
    print("smoke result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
