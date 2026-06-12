"""Modal cloud entry point for adhoc eval runs.

Mirrors `run_adhoc.py` in arguments (shared parser in `eval_tools.cli`) and
`run_bc_modal.py` in mechanism, with two cloud-specific translations:

  - Maps must come from a frozen map set on the eval-runs volume (`set:` mode);
    the `random:`/`replay_id:` modes read the local collector DB and can't run
    in a container.
  - Checkpoint paths under data/training/runs-cloud/ are rewritten to the
    training-runs volume mount — the same artifacts, read at the source.

Like training, the remote call is a fire-and-forget `.spawn()` — the run name
is generated locally and printed up front with the pull command, so nothing
depends on staying attached. Run with `--detach` so the spawned eval survives
the local process; without it, Modal stops the ephemeral app when the local
entrypoint returns and cancels the in-flight run. Follow progress with
`uv run modal app logs eval-adhoc` or the dashboard. The full run dir (config,
results.jsonl, per-game artifacts, analysis/) lands on the eval-runs volume;
pull it down with `pull_eval_runs.py <run-name>`.

Run (from repo root):
    uv run modal run --detach packages/eval-tools/scripts/run_eval_modal.py \\
        --policy checkpoint:data/training/runs-cloud/<run>/checkpoints/epoch_005.pt \\
        --policy evalbot \\
        --maps set:eval-map-set-v1:sample=20 --gpu T4
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import modal

from eval_tools.cli import build_arg_parser, spec_from_args
from eval_tools.run_spec import EvalConfigError, EvalRunSpec
from settings import PROJECT_ROOT, RUNS_CLOUD_DIR


EVAL_REQS = PROJECT_ROOT / "packages" / "eval-tools" / "modal_requirements.txt"
SIM_CORE_SRC = PROJECT_ROOT / "sim-core"

# Volume mounts inside the container.
RUNS_MOUNT = Path("/runs")        # training-runs: checkpoints (read)
EVAL_MOUNT = Path("/eval")        # eval-runs: map-sets/ (read) + runs/ (write)

image = (
    modal.Image.debian_slim(python_version="3.14")
    .apt_install("curl", "build-essential")
    .run_commands("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")
    # fjson (the fracturedjson crate's binary): utils.json_io.write_json shells
    # out to it for compact-but-readable formatting; without it every JSON
    # write warns and falls back to verbose output. --root /usr/local puts the
    # binary on PATH for the function runtime (~/.cargo/bin is only on PATH in
    # rustup-aware shells).
    .run_commands(". $HOME/.cargo/env && cargo install fracturedjson --root /usr/local")
    .add_local_dir(str(SIM_CORE_SRC), "/sim-core", copy=True, ignore=["target"])
    .run_commands(". $HOME/.cargo/env && pip install /sim-core")
    .uv_pip_install(requirements=[str(EVAL_REQS)])
    .add_local_python_source(
        "eval_tools", "eval_bot", "game_runner", "self_play", "training",
        "settings", "utils", "game_types",
    )
)

app = modal.App("eval-adhoc", image=image)

training_runs_vol = modal.Volume.from_name("generals-ai.training-runs")
eval_runs_vol = modal.Volume.from_name("generals-ai.eval-runs")


@app.function(
    gpu="T4",  # default; overridden via `with_options(gpu=...)` per-call
    volumes={str(RUNS_MOUNT): training_runs_vol, str(EVAL_MOUNT): eval_runs_vol},
    timeout=60 * 60 * 6,
)
def eval_remote(spec: EvalRunSpec) -> None:
    """Run one eval on a Modal GPU; the run dir lands on the eval-runs volume."""
    import torch

    if torch.cuda.is_available():
        print(f"cuda: {torch.cuda.get_device_name(0)}  "
              f"(region: {os.environ.get('MODAL_REGION')})")

    from eval_tools.runner import run_eval

    run_eval(spec)


def _translate_checkpoint(policy_spec: str) -> str:
    """Rewrite a local runs-cloud checkpoint path to its training-runs volume
    path. Pulled checkpoints are copies of volume files, so the mapping is
    exact; anything outside runs-cloud doesn't exist remotely — reject it."""
    if not policy_spec.startswith("checkpoint:"):
        return policy_spec
    parts = policy_spec.split(":")
    local = Path(parts[1])
    resolved = local if local.is_absolute() else PROJECT_ROOT / local
    try:
        parts[1] = str(RUNS_MOUNT / resolved.relative_to(RUNS_CLOUD_DIR))
    except ValueError:
        raise SystemExit(
            f"checkpoint '{parts[1]}' is not under "
            f"{RUNS_CLOUD_DIR.relative_to(PROJECT_ROOT)}/ — cloud eval loads "
            "checkpoints from the training-runs volume, so only pulled cloud "
            "runs are addressable"
        ) from None
    return ":".join(parts)


@app.local_entrypoint()
def main(*arglist: str) -> None:
    parser = build_arg_parser(description=__doc__)
    parser.add_argument("--gpu", type=str, default="T4",
                        help="Modal GPU class (default T4; eval is host-bound, "
                             "so a cheap card is usually right)")
    args = parser.parse_args(arglist)

    if not args.maps.startswith("set:"):
        raise SystemExit(
            f"--maps {args.maps}: cloud eval requires a frozen map set "
            "('set:<name>[:sample=K]') — random:/replay_id: modes read the "
            "local collector DB, which doesn't exist in the container"
        )

    policies = [_translate_checkpoint(p) for p in args.policy]
    run_name = dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    try:
        spec = spec_from_args(
            args,
            policy_specs=policies,
            out_root=str(EVAL_MOUNT / "runs"),
            run_name=run_name,
            map_sets_root=str(EVAL_MOUNT / "map-sets"),
        )
    except EvalConfigError as e:
        raise SystemExit(f"error: {e}") from None

    print(f"gpu:  {args.gpu}")
    print(f"run:  {run_name}")
    print()
    print("Once the run is done, pull artifacts:")
    print(f"  ./packages/eval-tools/scripts/pull_eval_runs.py {run_name}")

    eval_remote.with_options(gpu=args.gpu).spawn(spec)

    print()
    print("Run spawned.")
