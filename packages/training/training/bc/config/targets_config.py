"""TargetsConfig: the recipe knobs for the emitted training targets."""

# NOTE: This module is torch-free, as it is imported by the numpy-only golden tests.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from training.bc.aux_heads.elim_head_meta import ElimHeadVariant


if TYPE_CHECKING:
    from training.bc.model_config import ModelConfig


@dataclass(frozen=True, kw_only=True)
class TargetsConfig:
    elim_variant: ElimHeadVariant | None
    elim_bin_edges: tuple[int, ...] | None  # time_bin only

    def __post_init__(self) -> None:
        object.__setattr__(self, "elim_variant", ElimHeadVariant.coerce(self.elim_variant))
        if self.elim_bin_edges is not None:
            object.__setattr__(self, "elim_bin_edges", tuple(self.elim_bin_edges))
        if self.elim_variant == ElimHeadVariant.TIME_BIN:
            if self.elim_bin_edges is None:
                raise ValueError('elim head variant "time_bin" requires elim_bin_edges')
        elif self.elim_bin_edges is not None:
            raise ValueError(
                f"elim_bin_edges requires the time_bin variant, got {self.elim_variant}"
            )


TARGETS_CFG_NO_ELIM = TargetsConfig(
    elim_variant=None,
    elim_bin_edges=None,
)
TARGETS_CFG_ELIM_NEXT_DEATH = TargetsConfig(
    elim_variant=ElimHeadVariant.NEXT_DEATH,
    elim_bin_edges=None,
)


def targets_cfg_from(arch: ModelConfig) -> TargetsConfig:
    # TODO(config-bump): becomes a field read once the targets sub-config is stored.
    # At that time, we should re-consider what an appropriate builder function may be.
    variant = ElimHeadVariant.coerce(arch.elim_head_variant)
    return TargetsConfig(
        elim_variant=variant,
        elim_bin_edges=arch.elim_bin_edges if variant == ElimHeadVariant.TIME_BIN else None,
    )
