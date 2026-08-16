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

from training.bc.config.metrics_config import MetricsConfig
from training.bc.config.targets_config import TargetsConfig, targets_cfg_from
from training.bc.obs_config import ObsConfig


if TYPE_CHECKING:
    from training.bc.model_config import ModelConfig


@dataclass(frozen=True, kw_only=True)
class EmitSpec:
    obs: ObsConfig
    targets: TargetsConfig
    alive_mask: bool    # emit the per-player alive_mask key
    frame_info: bool    # emit the frame-info keys
    sim_frame: bool     # attach the raw sim frame (analysis only)

    def __post_init__(self) -> None:
        # TODO: This validation logic **should not** be owned by EmitSpec
        if self.targets.elim_variant == "time_bin" and not self.alive_mask:
            raise ValueError("time_bin requires alive_mask emission")


def emit_spec_from(
    arch: ModelConfig,
    metrics: MetricsConfig,
    *,
    frame_info: bool,
    sim_frame: bool = False,
) -> EmitSpec:
    return EmitSpec(
        obs=arch.obs,
        targets=targets_cfg_from(arch),
        alive_mask=metrics.include_alive_mask or arch.elim_head_variant == "time_bin",
        frame_info=frame_info,
        sim_frame=sim_frame,
    )


def emit_spec_for_model(model_cfg: ModelConfig, *, frame_info: bool) -> EmitSpec:
    """Build a spec for a loaded checkpoint."""
    # TODO(config-bump): read the stored metrics config off the checkpoint instead.
    metrics = MetricsConfig(include_alive_mask=model_cfg.elim_head_variant is not None)
    return emit_spec_from(model_cfg, metrics, frame_info=frame_info)
