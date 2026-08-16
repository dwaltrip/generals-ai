"""TargetsConfig: the recipe knobs for the emitted training targets."""

# NOTE: This module is torch-free, as it is imported by the numpy-only golden tests.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from training.bc.model_config import ModelConfig


@dataclass(frozen=True, kw_only=True)
class TargetsConfig:
    elim_variant: str | None  # None | "time_bin" | "next_death"
    elim_bin_edges: tuple[int, ...] | None  # time_bin only

    def __post_init__(self) -> None:
        if self.elim_bin_edges is not None:
            object.__setattr__(self, "elim_bin_edges", tuple(self.elim_bin_edges))
        if self.elim_variant == "time_bin":
            if self.elim_bin_edges is None:
                raise ValueError('elim head variant "time_bin" requires elim_bin_edges')
        elif self.elim_bin_edges is not None:
            raise ValueError(
                f"elim_bin_edges requires the time_bin variant, got {self.elim_variant!r}"
            )


def targets_cfg_from(arch: ModelConfig) -> TargetsConfig:
    # TODO(config-bump): becomes a field read once the targets sub-config is stored.
    # At that time, we should re-consider what an appropriate builder function may be.
    time_bin = arch.elim_head_variant == "time_bin"
    return TargetsConfig(
        elim_variant=arch.elim_head_variant,
        elim_bin_edges=arch.elim_bin_edges if time_bin else None,
    )
