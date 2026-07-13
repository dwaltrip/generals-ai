"""Run-directory lifecycle for BC training.

`initialize_run_dir` creates the run dir and drops provenance before a fresh
run. `RunArtifacts` bundles the per-run runtime resources (file handles via
`RunLogger`, paths, run-start measurements) and owns their open/close
lifecycle as a context manager. The resume helpers (`prepare_resume`,
`check_drift`, `write_args_resume`) plan and validate a resume into an
existing run dir and write the `_resume_NN`-suffixed sibling provenance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
import re

from training.bc.checkpoint import is_legacy_checkpoint
from training.bc.loss import LossConfig
from training.bc.run_logger import RunLogger
from training.bc.train_config import TrainConfig, _extract_loss, json_default
from utils.log import abort


def initialize_run_dir(config: TrainConfig, config_input_text: str | None = None) -> None:
    """Create `config.run_dir` and persist run provenance.

    Mkdirs the run dir with `exist_ok=False` so two runs landing in the
    same wall-clock second collide explicitly. Writes `args.json` (the fully
    resolved `TrainConfig` as JSON) and announces the path. When given, the
    verbatim `--config` input text (possibly partial) is preserved as
    `config.input.json` — provenance distinct from the resolved merge. (Text,
    not a path, so the cloud entry can pass it across the spawn boundary.) Call
    before `bc_run`.

    Split out from `bc_run` so cloud callers can drop sibling provenance
    files (e.g. `args_cloud.json`) into the run dir *before* training
    starts, instead of relying on a try/finally cleanup hook.
    """
    config.run_dir.mkdir(parents=True, exist_ok=False)
    print(f"run dir: {config.run_dir}")
    with (config.run_dir / "args.json").open("x") as fp:
        json.dump(asdict(config), fp, default=json_default, indent=2)
    if config_input_text is not None:
        (config.run_dir / "config.input.json").write_text(config_input_text)


# TODO: fix the naming collision with the unrelated RunArtifacts in perf_report.py.
@dataclass
class RunArtifacts:
    """Per-run runtime: born at run start, released at run end.

    Wraps the file handles (via `RunLogger`), paths, and run-start
    measurements (FLOPs/sample, device peak) that aren't part of the
    serialized training state. Acts as a context manager: `__enter__`
    opens the logger's JSONL writers, `__exit__` closes them.

    `suffix` distinguishes resume segments ("" for the original run,
    "_resume_NN" for resumes) and is the single source for this segment's
    artifact filenames — the `RunLogger`, `log_path`, and `gpu_util_path`
    all derive from it.

    TODO: consider folding the `gpu_util_sidecar` lifecycle in here too —
    it's per-run runtime by the same definition. Currently it's a sibling
    context manager at the call site because it needs `device`.
    """

    run_dir: Path
    ckpt_dir: Path
    flops_per_sample: int
    peak_tflops: float | None
    suffix: str = ""
    resume_segment: int = 0
    logger: RunLogger = field(init=False)

    def __post_init__(self) -> None:
        self.logger = RunLogger(self.run_dir, self.suffix)

    @property
    def log_path(self) -> Path:
        return self.run_dir / f"run{self.suffix}.log"

    @property
    def gpu_util_path(self) -> Path:
        return self.run_dir / f"gpu_util{self.suffix}.jsonl"

    def __enter__(self) -> RunArtifacts:
        self.logger.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.logger.close()


@dataclass
class ResumeInfo:
    """The plan for resuming a run, computed by `prepare_resume`.

    `next_suffix` names this segment's artifacts ("_resume_NN");
    `latest_checkpoint` is the checkpoint to restore; `parent_epoch` is its
    epoch (the last completed); `parent_args_path` is the immediate
    predecessor's args file (for drift comparison + the `_resume_meta` chain);
    `is_legacy_checkpoint` flags a bare-state_dict checkpoint (cold optimizer
    restart — requires `--legacy-lr-warmup-batches`).
    """

    next_suffix: str
    latest_checkpoint: Path
    parent_epoch: int
    parent_args_path: Path
    is_legacy_checkpoint: bool


# Drift buckets (design doc §--force-config-mismatch). Excluded fields are
# always free to change on resume; checkpoint-owned fields can never change
# (the restored weights/optimizer state pin them — not even
# --force-config-mismatch overrides); everything else is trajectory-critical
# and gated by --force-config-mismatch.
#   - `arch` is pinned by the restored weights (you can't reshape them).
#   - `lr`/`weight_decay` are pinned by the restored optimizer state.
#   - `gpu` is recorded provenance only; the card may change freely on resume.
#   - `dump_val_frames` is a diagnostics knob with no trajectory effect; a
#     resume overlay may toggle it (e.g. resume a finished run with dumps on).
_DRIFT_EXCLUDED = frozenset(
    {"epochs", "run_dir", "manifest", "intermediate", "gpu", "dump_val_frames"}
)
_CHECKPOINT_OWNED = frozenset({"lr", "weight_decay", "arch"})

_RESUME_SUFFIX_RE = re.compile(r"_resume_(\d{2})\.")


def prepare_resume(run_dir: Path) -> ResumeInfo:
    """Inspect a run dir to plan a resume. Pure read — no writes.

    Picks the latest checkpoint by epoch number, computes the next `_resume_NN`
    suffix from the resume artifacts already present, and peeks the chosen
    checkpoint's format (combined vs. legacy bare-state_dict). Aborts
    (`SystemExit`) if the run dir or its checkpoints are missing — the operator
    fixes the path and retries.
    """
    if not run_dir.exists():
        abort(f"--resume: run dir not found: {run_dir}")
    ckpt_dir = run_dir / "checkpoints"
    ckpts = list(ckpt_dir.glob("epoch_*.pt")) if ckpt_dir.exists() else []
    if not ckpts:
        abort(f"--resume: no checkpoints found under {ckpt_dir}")

    latest = max(ckpts, key=lambda p: int(p.stem.split("_")[1]))
    parent_epoch = int(latest.stem.split("_")[1])

    existing = [
        int(m.group(1))
        for p in run_dir.glob("*_resume_[0-9][0-9].*")
        if (m := _RESUME_SUFFIX_RE.search(p.name))
    ]
    next_n = max(existing) + 1 if existing else 1
    parent_args = "args.json" if next_n == 1 else f"args_resume_{next_n - 1:02d}.json"

    return ResumeInfo(
        next_suffix=f"_resume_{next_n:02d}",
        latest_checkpoint=latest,
        parent_epoch=parent_epoch,
        parent_args_path=run_dir / parent_args,
        is_legacy_checkpoint=is_legacy_checkpoint(latest),
    )


def load_parent_config(parent_args_path: Path) -> dict:
    """Load a parent run's args file as a `TrainConfig` field dict, dropping
    the `_resume_meta` block (present when the parent is itself a resume) and
    normalizing loss knobs into the current nested `loss` shape.

    Migration shim: a pre-nesting parent recorded loss knobs flat at the top
    level and lacks knobs added since (e.g. `mu_pass`). Folding them under `loss`
    and completing from the `LossConfig` defaults lets both the drift-compare and
    the resume merge see the current shape instead of tripping on the reshape —
    the parent's resolved objective is its old values plus today's defaults for
    knobs it predates, which is exactly what an unchanged resume reconstructs."""
    if not parent_args_path.exists():
        abort(f"--resume: parent args not found: {parent_args_path}")
    data = json.loads(parent_args_path.read_text())
    data.pop("_resume_meta", None)
    data["loss"] = {**asdict(LossConfig()), **_extract_loss(data)}
    return data


def _json_normalize(v: object) -> object:
    """Recursively coerce tuples to lists so a config value compares equal to its
    JSON-round-tripped form. `asdict(config)` keeps tuple-valued fields (e.g.
    `arch.elim_bin_edges`) as tuples, but a parent loaded from `args.json` has
    them as lists — without this, an unchanged resume would spuriously trip the
    arch-drift guard."""
    if isinstance(v, list | tuple):
        return [_json_normalize(x) for x in v]
    if isinstance(v, dict):
        return {k: _json_normalize(x) for k, x in v.items()}
    return v


def check_drift(config: TrainConfig, parent: dict, force_config_mismatch: bool) -> None:
    """Abort on disallowed config drift between a resume config and its parent.

    Checkpoint-owned fields (`arch`, `lr`, `weight_decay`) are pinned by the
    restored weights/optimizer state — a change is silently ineffective and
    always aborts. Other non-excluded fields are trajectory-critical: they abort
    unless `--force-config-mismatch`. Excluded fields (epochs target, paths,
    gpu, dump_val_frames) are free.

    Legacy-parent edge: a pre-`arch` parent config has no `arch` key, so a naive
    compare (`None` vs the default arch dict) would spuriously abort an
    unchanged resume. Skip arch-drift in that case (the bare state_dict
    reconstructs via `LEGACY_ARCH` regardless) and warn.
    """
    cfg = asdict(config)
    locked, trajectory = [], []
    for name, new_val in cfg.items():
        if name in _DRIFT_EXCLUDED or _json_normalize(parent.get(name)) == _json_normalize(new_val):
            continue
        if name == "arch" and "arch" not in parent:
            print("--resume: parent predates the `arch` key; skipping arch-drift "
                  "check (weights reconstruct via LEGACY_ARCH).")
            continue
        bucket = locked if name in _CHECKPOINT_OWNED else trajectory
        bucket.append(f"{name} {parent.get(name)!r}->{new_val!r}")
    if locked:
        abort(
            f"--resume: arch/lr/weight_decay are checkpoint-owned and cannot change "
            f"on resume ({', '.join(locked)}); the restored weights/optimizer state "
            f"pin them."
        )
    if trajectory and not force_config_mismatch:
        abort(
            f"--resume: config drift from parent ({', '.join(trajectory)}); "
            f"pass --force-config-mismatch to override."
        )


def write_args_resume(
    config: TrainConfig,
    info: ResumeInfo,
    *,
    legacy_lr_warmup_batches: int | None,
    force_config_mismatch: bool,
) -> None:
    """Write `args_resume_NN.json`: the resume config plus a `_resume_meta`
    block. Uses exclusive create, so a miscomputed suffix fails loudly.

    `_resume_meta` records the resume-segment facts that aren't part of the
    trajectory config: the predecessor link (for chain-walking), and the
    resume-time directives (`legacy_lr_warmup_batches`, `force_config_mismatch`)
    plus the detected `legacy_checkpoint` flag — so every input to the segment
    is captured even though these directives aren't `TrainConfig` fields."""
    payload = {
        "_resume_meta": {
            "resumed_from_epoch": info.parent_epoch,
            "parent_args": info.parent_args_path.name,
            "resumed_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "legacy_lr_warmup_batches": legacy_lr_warmup_batches,
            "force_config_mismatch": force_config_mismatch,
            "legacy_checkpoint": info.is_legacy_checkpoint,
        },
        **asdict(config),
    }
    with (config.run_dir / f"args{info.next_suffix}.json").open("x") as fp:
        json.dump(payload, fp, default=json_default, indent=2)
