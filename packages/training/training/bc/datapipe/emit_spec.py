"""EmitSpec: complete contract for what the dataset walk emits.

This is intended to only own knobs that affect the output for a single frame.
Orchestration params (e.g. shuffling, workers) are specifically not included.

The builders below own the canonical derivation from the training config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from training.bc.aux_heads.elim_head_meta import does_elim_head_need_alive_mask
from training.bc.config.metrics_config import MetricsConfig, metrics_cfg_from
from training.bc.config.targets_config import (
    TARGETS_CFG_NO_ELIM,
    TargetsConfig,
    targets_cfg_from,
)
from training.bc.obs_config import ObsConfig


if TYPE_CHECKING:
    from training.bc.model_config import ModelConfig


@dataclass(frozen=True, kw_only=True)
class EmitSpec:
    obs: ObsConfig
    targets: TargetsConfig
    emit_alive_mask: bool
    emit_frame_info: bool
    # NOTE: This is NOT safe for train dataloader (SimFrame is not collatable).
    # Only valid for analysis dataset walks.
    attach_sim_frame: bool

    @property
    def partial(self) -> PartialEmitSpec:
        return PartialEmitSpec(targets=self.targets, emit_alive_mask=self.emit_alive_mask)


@dataclass(frozen=True, kw_only=True)
class PartialEmitSpec:
    """The part of the spec the emission tail and the per-game precompute read.
    `obs` and the packing flags are left open."""

    targets: TargetsConfig
    emit_alive_mask: bool

    def to_spec(
        self, obs: ObsConfig, *, emit_frame_info: bool, attach_sim_frame: bool
    ) -> EmitSpec:
        return EmitSpec(
            obs=obs,
            targets=self.targets,
            emit_alive_mask=self.emit_alive_mask,
            emit_frame_info=emit_frame_info,
            attach_sim_frame=attach_sim_frame,
        )


def partial_emit_spec_from(targets: TargetsConfig, metrics: MetricsConfig) -> PartialEmitSpec:
    # TODO: `alive_mask` is emitted when either the loss needs it (time_bin) or
    # metrics request it. If more keys end up wanted by several configs, this
    # "who requires which key" logic needs a proper home rather than an `or` here.
    return PartialEmitSpec(
        targets=targets,
        emit_alive_mask=metrics.include_alive_mask
        or does_elim_head_need_alive_mask(targets.elim_variant),
    )


def base_emit_spec(obs: ObsConfig) -> EmitSpec:
    """Minimal spec: all optional flags off"""
    return EmitSpec(
        obs=obs,
        targets=TARGETS_CFG_NO_ELIM,
        emit_alive_mask=False,
        emit_frame_info=False,
        attach_sim_frame=False,
    )


def emit_spec_from(
    arch: ModelConfig,
    metrics: MetricsConfig,
    *,
    emit_frame_info: bool,
    attach_sim_frame: bool = False,
) -> EmitSpec:
    return partial_emit_spec_from(targets_cfg_from(arch), metrics).to_spec(
        arch.obs, emit_frame_info=emit_frame_info, attach_sim_frame=attach_sim_frame
    )


def emit_spec_for_model(model_cfg: ModelConfig, *, emit_frame_info: bool) -> EmitSpec:
    """Build a spec for a loaded checkpoint."""
    # TODO(config-bump): read the stored metrics config off the checkpoint instead.
    metrics = metrics_cfg_from(model_cfg)
    return emit_spec_from(model_cfg, metrics, emit_frame_info=emit_frame_info)
