"""Resume a BC training run from its latest checkpoint.

The resume entry point, sibling to `bc.train.bc_run`. Owns the resume
bookkeeping — overlaying the `--config` file over the parent's resolved config,
gating config drift, gating the legacy cold-restart, writing the
`args_resume_NN.json` provenance — then hands off to the shared
`bc.train.run_training` core. The caller supplies the resume `run_dir` and a
precomputed `ResumeInfo` (from `prepare_resume`); keeping that inspection at the
wrapper boundary lets the Modal wrapper write segment-suffixed provenance with
the same suffix. Keeping resume logic here lets `bc.train` stay ignorant of it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bc.model_config import build_model_cfg
from bc.resume_warmup import WarmupSchedule
from bc.run_dir import (
    ResumeInfo,
    check_drift,
    load_parent_config,
    write_args_resume,
)
from bc.state import TrainingState
from bc.train import run_training
from bc.train_config import TrainConfig
from utils.log import abort


def _resume_config(
    run_dir: Path, parent: dict, overlay: dict, operational: dict[str, Any]
) -> TrainConfig:
    """Build the effective resume config.

    Start from the parent's resolved config (recipe + arch + data paths all
    carry over), overlay the operator's `--config` file (arch + recipe), then the
    explicit operational CLI flags — so an unchanged knob continues the parent's
    value instead of resetting to a default. `run_dir` is the resume target.
    Arch is deep-merged (overlay over parent); any *net* arch change is rejected
    downstream by `check_drift` (arch is checkpoint-owned).

    Note the asymmetry with a fresh run: a fresh `--config` merges over
    *defaults*; resume's merges over the *parent*.
    """
    parent = dict(parent)
    parent.pop("run_dir", None)
    parent_arch = parent.pop("arch", None)
    # A pre-`arch` parent recorded its variant as a top-level field; fold it into
    # the arch so it isn't a stray key (and isn't lost). Other removed/renamed
    # fields don't exist, so this is the only legacy-key translation needed.
    legacy_variant = parent.pop("value_head_variant", None)
    if parent_arch is None:
        parent_arch = {} if legacy_variant is None else {"value_head_variant": legacy_variant}

    overlay = dict(overlay)
    overlay_arch = overlay.pop("arch", {})

    merged = {**parent, **overlay, **operational}
    merged["manifest"] = Path(merged["manifest"])
    merged["intermediate"] = Path(merged["intermediate"])
    arch = build_model_cfg(**{**parent_arch, **overlay_arch})
    return TrainConfig(arch=arch, run_dir=run_dir, **merged)


def bc_resume(
    run_dir: Path,
    info: ResumeInfo,
    overlay: dict,
    operational: dict[str, Any],
    force_config_mismatch: bool,
    legacy_lr_warmup_batches: int | None = None,
) -> None:
    """Resume the run at `run_dir` from its latest checkpoint.

    The effective config is the parent's resolved config overlaid with the
    `--config` file (`overlay`) and the explicit operational CLI flags
    (`operational`). `info` is the precomputed `prepare_resume` result (both
    wrappers compute it; the Modal wrapper additionally writes segment-suffixed
    cloud provenance off the same suffix). `legacy_lr_warmup_batches` is the
    cold-restart opt-in: required to resume a legacy bare-state_dict checkpoint,
    and rejected on a combined one.
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
            info.latest_checkpoint, effective, dev, fallback_epoch=info.parent_epoch
        )
        if legacy_lr_warmup_batches is not None:
            state.warmup = WarmupSchedule(effective.lr, legacy_lr_warmup_batches)
        return state

    run_training(effective, suffix=info.next_suffix, make_state=make_state)
