"""
POC Modal entrypoint — validates that our uv workspace installs cleanly inside a Modal image.

Run with:
    uv run modal run cloud-poc/modal_entry.py

The image is built minimally: a stand-in workspace root that lists only `cloud-poc`
as a member, plus the package source. Nothing else from the repo is uploaded.
"""

from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
CLOUD_POC = REPO_ROOT / "cloud-poc"

# Image build — targeted, explicit copies (no broad add_local_dir):
#   1) Start from uv's prebuilt image (gives us uv + Python 3.14).
#   2) Copy the Modal-only workspace root pyproject as /workspace/pyproject.toml.
#   3) Copy cloud-poc's own pyproject + Python source under /workspace/cloud-poc/.
#   4) Run `uv sync --no-dev` from /workspace.
#
# Not --frozen: the real repo uv.lock pins 6 members; this stand-in lists only one,
# so we let uv resolve fresh inside the image. cloud-train-poc has no deps, so this
# is instant.
image = (
    modal.Image.from_registry("ghcr.io/astral-sh/uv:python3.14-bookworm-slim")
    .workdir("/workspace")
    .add_local_file(
        CLOUD_POC / "modal_workspace_pyproject.toml",
        "/workspace/pyproject.toml",
        copy=True,
    )
    .add_local_file(
        CLOUD_POC / "pyproject.toml",
        "/workspace/cloud-poc/pyproject.toml",
        copy=True,
    )
    .add_local_dir(
        CLOUD_POC / "cloud_train_poc",
        "/workspace/cloud-poc/cloud_train_poc",
        copy=True,
        ignore=["__pycache__", "*.pyc"],
    )
    .run_commands("uv sync --no-dev")
    .env({"PATH": "/workspace/.venv/bin:/usr/local/bin:/usr/bin:/bin"})
)

app = modal.App("cloud-train-poc", image=image)


@app.function()
def remote_hello() -> dict:
    from cloud_train_poc.work import hello

    return hello()


@app.local_entrypoint()
def main():
    result = remote_hello.remote()
    print("got back from cloud:")
    for k, v in result.items():
        print(f"  {k}: {v}")
