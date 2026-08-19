"""Spec-driven per-game precompute for the dataset walk."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from training.bc.emit_spec import EmitSpec
from training.bc.player_status import PlayerStatusCtx, precompute_player_status
from training.bc.targets.elim_targets import ElimCtx, make_elim_ctx


@dataclass(frozen=True)
class EmitPrecompute:
    """Per-game precompute backing the spec's emissions. Built once per game,
    consumed per (perspective, frame)."""

    player_status: PlayerStatusCtx | None
    elim: ElimCtx | None


def precompute_for(spec: EmitSpec, sim: dict[str, np.ndarray]) -> EmitPrecompute:
    elim = (
        make_elim_ctx(sim, spec.targets.elim_bin_edges)
        if spec.targets.elim_variant is not None
        else None
    )
    player_status = None
    if spec.emit_alive_mask:
        player_status = (
            elim.player_status if elim is not None else precompute_player_status(sim)
        )
    return EmitPrecompute(player_status=player_status, elim=elim)
