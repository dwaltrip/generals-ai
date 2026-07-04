"""Resume a BC training run from its latest checkpoint.

`bc_resume` is the resume counterpart to `bc.train.bc_run`: it re-derives the
config from the parent run plus operator overrides, gates unsafe resumes,
writes the segment's provenance, and then drives the same `bc.train.run_training`
core as a fresh run. The module exists so `bc.train` stays free of resume logic.

The caller inspects the run dir (`prepare_resume`) and passes the resulting
`ResumeInfo` in. The Modal wrapper relies on this split: it needs the segment
suffix before the run starts, to name its cloud provenance to match.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from training.bc.loss import LossConfig
from training.bc.model_config import build_model_cfg
from training.bc.resume_warmup import WarmupSchedule
from training.bc.run_dir import (
    ResumeInfo,
    check_drift,
    load_parent_config,
    write_args_resume,
)
from training.bc.state import TrainingState
from training.bc.train import run_training
from training.bc.train_config import TrainConfig, _extract_loss
from utils.log import abort


def _resume_config(
    run_dir: Path, parent: dict, overlay: dict, operational: dict[str, Any]
) -> TrainConfig:
    """Build the resume config by merging three layers: the parent's config,
    then the `--config` file, then the explicit CLI flags.

    Later layers take precedence over earlier ones. Defaults are only applied
    to fields not set by either the parent config or the resume overrides.
    A fresh run is different: there, `--config` merges over the defaults.
    """
    parent = dict(parent)
    parent.pop("run_dir", None)
    parent_arch = parent.pop("arch", None)
    # Some older runs recorded `value_head_variant` as a top-level config field.
    # If present, merge it into `arch`, matching the current TrainConfig shape.
    # No other field has ever moved, so this is the only translation needed.
    # NOTE(ckpt-cfg-refactor-note): this would become part of the v0 -> v1 normalizer.
    legacy_variant = parent.pop("value_head_variant", None)
    if parent_arch is None:
        parent_arch = {} if legacy_variant is None else {"value_head_variant": legacy_variant}
    # Loss knobs deep-merge like arch: pull them out of both sides (flat or
    # nested) so the overlay overrides per-knob, not whole-block.
    parent_loss = _extract_loss(parent)
    overlay = dict(overlay)
    overlay_arch = overlay.pop("arch", {})
    overlay_loss = _extract_loss(overlay)

    merged = {**parent, **overlay, **operational}
    merged["manifest"] = Path(merged["manifest"])
    merged["intermediate"] = Path(merged["intermediate"])
    arch = build_model_cfg(**{**parent_arch, **overlay_arch})
    loss = LossConfig(**{**parent_loss, **overlay_loss})
    return TrainConfig(arch=arch, loss=loss, run_dir=run_dir, **merged)


def bc_resume(
    run_dir: Path,
    info: ResumeInfo,
    overlay: dict,
    operational: dict[str, Any],
    force_config_mismatch: bool,
    code_sha: str,
    legacy_lr_warmup_batches: int | None = None,
) -> None:
    """Resume the run at `run_dir` from its latest checkpoint.

    - `info`: the precomputed `prepare_resume` result.
    - `overlay` and `operational`: the parsed `--config` file and the explicit
      CLI flags. `_resume_config` merges them into the effective config.
    - `code_sha`: recorded into this segment's checkpoints.
    - `legacy_lr_warmup_batches`: the legacy cold-restart opt-in. Required for
      a bare-state_dict checkpoint, rejected otherwise.
    """
    parent = load_parent_config(info.parent_args_path)
    effective = _resume_config(run_dir, parent, overlay, operational)
    check_drift(effective, parent, force_config_mismatch)
    if info.parent_epoch >= effective.epochs:
        abort(
            f"--resume: latest checkpoint is epoch {info.parent_epoch}, but epochs "
            f"is {effective.epochs} — nothing to resume (raise epochs in --config to "
            f"continue)."
        )
    if info.is_legacy_checkpoint and legacy_lr_warmup_batches is None:
        abort(
            "--resume: latest checkpoint is a legacy bare-state_dict (no saved "
            "optimizer state). Cold-restarting the optimizer is an explicit opt-in "
            "— pass --legacy-lr-warmup-batches N (e.g. 500) to ramp the LR while "
            "AdamW's variance estimate re-warms."
        )
    if not info.is_legacy_checkpoint and legacy_lr_warmup_batches is not None:
        abort(
            "--resume: --legacy-lr-warmup-batches applies only to legacy "
            "bare-state_dict checkpoints. This is a combined-format checkpoint "
            "with healthy optimizer state — drop the flag to resume normally."
        )

    write_args_resume(
        effective, info,
        legacy_lr_warmup_batches=legacy_lr_warmup_batches,
        force_config_mismatch=force_config_mismatch,
    )
    print(f"resuming {run_dir.name} from epoch {info.parent_epoch} "
          f"(segment {info.next_suffix})")

    def make_state(dev):
        state = TrainingState.from_checkpoint(
            info.latest_checkpoint, effective, dev, code_sha,
            fallback_epoch=info.parent_epoch,
        )
        if legacy_lr_warmup_batches is not None:
            state.warmup = WarmupSchedule(effective.lr, legacy_lr_warmup_batches)
        return state

    run_training(effective, suffix=info.next_suffix, make_state=make_state)
