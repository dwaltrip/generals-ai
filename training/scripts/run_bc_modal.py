"""Modal cloud entry point for BC training.

Step 1: CPU smoke. Builds the Modal image with the full training stack
installed and verifies `bc.train` is importable inside the container.
No GPU, no Volume, no actual training step — just confirms the
install chain works for our real package layout.

Real training runs (Volume mounts + GPU) land in subsequent commits.

Run:
    uv run modal run training/scripts/run_bc_modal.py
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


@app.local_entrypoint()
def main() -> None:
    result = smoke.remote()
    print("smoke result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
