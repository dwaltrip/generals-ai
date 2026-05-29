"""Resume a BC training run from its latest checkpoint.

The resume entry point, sibling to `bc.train.bc_run`. Owns the resume
bookkeeping — overlaying the parent config, gating config drift, gating the
legacy cold-restart, writing the `args_resume_NN.json` provenance — then hands
off to the shared `bc.train.run_training` core. The caller supplies a
precomputed `ResumeInfo` (from `prepare_resume`); keeping that inspection at the
wrapper boundary lets the Modal wrapper write segment-suffixed provenance with
the same suffix. Keeping resume logic here lets `bc.train` stay ignorant of it.
"""

from __future__ import annotations

from typing import Any

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


# TrainConfig fields supplied by the environment, not carried over from the
# parent run: the resume dir and the (possibly relocated) data paths.
_PATH_FIELDS = ("manifest", "intermediate", "run_dir")


def _resume_config(base: TrainConfig, parent: dict, overrides: dict[str, Any]) -> TrainConfig:
    """Build the effective resume config.

    Training knobs come from the parent run, overlaid with the operator's
    explicitly-passed flags (`overrides`) — so an unchanged flag continues the
    parent's value instead of resetting to a CLI default. Paths and run_dir
    come from `base` (the environment-resolved config). The net effect: only
    the new `epochs` target and any explicit override differ from the parent.
    """
    knobs = {k: v for k, v in parent.items() if k not in _PATH_FIELDS}
    knobs.update(overrides)
    return TrainConfig(
        manifest=base.manifest,
        intermediate=base.intermediate,
        run_dir=base.run_dir,
        **knobs,
    )


def bc_resume(
    config: TrainConfig,
    force_config_mismatch: bool,
    overrides: dict[str, Any],
    info: ResumeInfo,
    legacy_lr_warmup_batches: int | None = None,
) -> None:
    """Resume the run at `config.run_dir` from its latest checkpoint.

    `config` supplies the environment-resolved run_dir + data paths;
    `overrides` is the operator's explicitly-passed training flags. The
    effective config is the parent's knobs overlaid with `overrides`. `info` is
    the precomputed `prepare_resume` result (both wrappers compute it; the Modal
    wrapper additionally writes segment-suffixed cloud provenance off the same
    suffix). `legacy_lr_warmup_batches` is the cold-restart opt-in: required to
    resume a legacy bare-state_dict checkpoint, and rejected on a combined one.
    """
    parent = load_parent_config(info.parent_args_path)
    effective = _resume_config(config, parent, overrides)
    check_drift(effective, parent, force_config_mismatch)
    if info.parent_epoch >= effective.epochs:
        raise SystemExit(
            f"--resume: latest checkpoint is epoch {info.parent_epoch}, but --epochs "
            f"is {effective.epochs} — nothing to resume (raise --epochs to continue)."
        )
    if info.is_legacy_checkpoint and legacy_lr_warmup_batches is None:
        raise SystemExit(
            "--resume: latest checkpoint is a legacy bare-state_dict (no saved "
            "optimizer state). Cold-restarting the optimizer is an explicit opt-in "
            "— pass --legacy-lr-warmup-batches N (e.g. 500) to ramp the LR while "
            "AdamW's variance estimate re-warms."
        )
    if not info.is_legacy_checkpoint and legacy_lr_warmup_batches is not None:
        raise SystemExit(
            "--resume: --legacy-lr-warmup-batches applies only to legacy "
            "bare-state_dict checkpoints. This is a combined-format checkpoint "
            "with healthy optimizer state — drop the flag to resume normally."
        )

    write_args_resume(
        effective, info,
        legacy_lr_warmup_batches=legacy_lr_warmup_batches,
        force_config_mismatch=force_config_mismatch,
    )
    print(f"resuming {config.run_dir.name} from epoch {info.parent_epoch} "
          f"(segment {info.next_suffix})")

    def make_state(dev):
        state = TrainingState.from_checkpoint(
            info.latest_checkpoint, effective, dev, fallback_epoch=info.parent_epoch
        )
        if legacy_lr_warmup_batches is not None:
            state.warmup = WarmupSchedule(effective.lr, legacy_lr_warmup_batches)
        return state

    run_training(effective, suffix=info.next_suffix, make_state=make_state)
