"""EmitSpec: complete contract for what the dataset walk emits.

This is intended to only own knobs that affect the output for a single frame.
Orchestration params (e.g. shuffling, workers) are specifically not included.

The builders below own the canonical derivation from TrainConfig.
"""

# NOTE: This is torch-free, as it is imported by the numpy-only golden tests.

# TODO: Will likely move into a future `bc.walk` module alongside the walk-core
# work (possibly with sample.py, encode_frame.py, and sim_types.py).

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from training.bc.aux_heads.elim_head_meta import does_elim_head_need_alive_mask
from training.bc.config.metrics_config import MetricsConfig, metrics_cfg_from
from training.bc.config.targets_config import TargetsConfig, targets_cfg_from
from training.bc.obs_config import ObsConfig


if TYPE_CHECKING:
    from training.bc.model_config import ModelConfig


@dataclass(frozen=True, kw_only=True)
class EmitSpec:
    obs: ObsConfig
    targets: TargetsConfig
    emit_alive_mask: bool
    emit_frame_info: bool
    # NOTE: Only valid for analysis dataset walks (SimFrame is not collatable).
    attach_sim_frame: bool


@dataclass(frozen=True, kw_only=True)
class PartialEmitSpec:
    """Partial emit spec with `obs` and `emit_frame_info` left open.
    Used for analysis."""

    targets: TargetsConfig
    emit_alive_mask: bool
    attach_sim_frame: bool

    def to_spec(self, obs: ObsConfig, *, emit_frame_info: bool) -> EmitSpec:
        return EmitSpec(
            obs=obs,
            targets=self.targets,
            emit_alive_mask=self.emit_alive_mask,
            emit_frame_info=emit_frame_info,
            attach_sim_frame=self.attach_sim_frame,
        )


def emit_spec_from(
    arch: ModelConfig,
    metrics: MetricsConfig,
    *,
    emit_frame_info: bool,
    attach_sim_frame: bool = False,
) -> EmitSpec:
    return EmitSpec(
        obs=arch.obs,
        targets=targets_cfg_from(arch),
        emit_alive_mask=metrics.include_alive_mask
        or does_elim_head_need_alive_mask(arch.elim_head_variant),
        emit_frame_info=emit_frame_info,
        attach_sim_frame=attach_sim_frame,
    )


def emit_spec_for_model(model_cfg: ModelConfig, *, emit_frame_info: bool) -> EmitSpec:
    """Build a spec for a loaded checkpoint."""
    # TODO(config-bump): read the stored metrics config off the checkpoint instead.
    metrics = metrics_cfg_from(model_cfg)
    return emit_spec_from(model_cfg, metrics, emit_frame_info=emit_frame_info)
