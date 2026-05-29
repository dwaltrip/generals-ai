"""Resume a BC training run from its latest checkpoint.

The resume entry point, sibling to `bc.train.bc_run`. Owns the resume
bookkeeping — inspecting the run dir, overlaying the parent config, gating
config drift, writing the `args_resume_NN.json` provenance — then hands off
to the shared `bc.train.run_training` core. Keeping it here lets `bc.train`
stay ignorant of resumes.
"""

from __future__ import annotations

from typing import Any

from bc.run_dir import (
    ResumeInfo,
    check_drift,
    load_parent_config,
    prepare_resume,
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
    info: ResumeInfo | None = None,
) -> None:
    """Resume the run at `config.run_dir` from its latest checkpoint.

    `config` supplies the environment-resolved run_dir + data paths;
    `overrides` is the operator's explicitly-passed training flags. The
    effective config is the parent's knobs overlaid with `overrides`. `info`
    lets a caller pass a pre-computed `prepare_resume` result (the Modal
    wrapper does, so it can write segment-suffixed provenance with the same
    suffix); when `None` it is computed here.
    """
    if info is None:
        info = prepare_resume(config.run_dir)
    parent = load_parent_config(info.parent_args_path)
    effective = _resume_config(config, parent, overrides)
    check_drift(effective, parent, force_config_mismatch)
    if info.parent_epoch >= effective.epochs:
        raise SystemExit(
            f"--resume: latest checkpoint is epoch {info.parent_epoch}, but --epochs "
            f"is {effective.epochs} — nothing to resume (raise --epochs to continue)."
        )

    write_args_resume(effective, info)
    print(f"resuming {config.run_dir.name} from epoch {info.parent_epoch} "
          f"(segment {info.next_suffix})")
    run_training(
        effective,
        suffix=info.next_suffix,
        make_state=lambda dev: TrainingState.from_checkpoint(info.latest_checkpoint, effective, dev),
    )
