"""Modal cloud entry point for BC training.

Two smoke functions, neither does training:

- `smoke` — image smoke. Builds the Modal image with the full training stack
  installed and verifies `bc.train` is importable inside the container.
  No GPU, no Volume.
- `smoke_volume` — Volume smoke. Mounts the `generals-ai.parsed-replays`
  Volume RO, opens the bundled manifest, resolves the first training sample,
  and loads its `.npz` + `.meta.npz` to confirm path translation works
  end-to-end.

Real training runs (GPU + outputs Volume) land in subsequent commits.

Run:
    uv run modal run training/scripts/run_bc_modal.py                  # smoke
    uv run modal run training/scripts/run_bc_modal.py::smoke_volume    # volume
"""

from __future__ import annotations

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


@app.function()
def smoke() -> dict:
    """Step-1 smoke: import the training stack, report container info."""
    import platform
    import sys

    import torch

    # The actual import-chain check — resolves bc + shared (from training)
    # and the workspace cross-dep utils.
    from bc.train import TrainConfig, bc_run  # noqa: F401
    from bc.train_cli import build_arg_parser, config_from_args  # noqa: F401

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

    from bc.splits import load_manifest, samples_for_split
    from bc.utils import meta_path_for

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


@app.local_entrypoint()
def main() -> None:
    result = smoke.remote()
    print("smoke result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
