from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TargetsConfig:
    """Config for the emitted training targets.

    TargetConfig knobs should only affect encoding / dataloader code paths.
    Loss calculation codepaths are specified via LossConfig.
    Session note 7.28-2 digs into this seam in more detail.

    This module must be torch-free: it is imported by numpy-only golden tests.
    """
    # Emit the per-player alive_mask, with or without an elim head variant.
    emit_alive_mask: bool

    # Selects the elim head's target keys: None, "time_bin", or "next_death".
    elim_variant: str | None

    # time_bin only: bin edges for elim_bin_target. None for everything else.
    elim_bin_edges: tuple[int, ...] | None

    def __post_init__(self) -> None:
        # TODO(sketch): validate variant / edges / mask consistency.
        raise NotImplementedError('...')
