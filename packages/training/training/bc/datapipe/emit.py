from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from training.bc.aux_heads.registry import spec_for
from training.bc.constants import W_PADDED
from training.bc.datapipe.emit_spec import PartialEmitSpec
from training.bc.datapipe.precompute import EmitPrecompute
from training.bc.datapipe.sim_types import GameMeta, PerspectiveMeta
from training.bc.player_status import make_alive_mask
from training.bc.targets.core_targets import policy_pass_target, value_target


# TODO: I don't love the name FrameEmission...
@dataclass(frozen=True)
class FrameEmission:
    action_target: np.ndarray
    is_pass: np.ndarray
    value_target: np.ndarray
    alive_mask: np.ndarray | None
    aux_head_targets: dict[str, np.ndarray] | None

    def to_dict(self) -> dict[str, np.ndarray]:
        out = {
            "action_target": self.action_target,
            "is_pass": self.is_pass,
            "value_target": self.value_target,
        }
        if self.alive_mask is not None:
            out["alive_mask"] = self.alive_mask
        if self.aux_head_targets is not None:
            out.update(self.aux_head_targets)
        return out


def emit_frame(
    sim: dict[str, np.ndarray],
    t: int,
    game: GameMeta,
    perspective: PerspectiveMeta,
    spec: PartialEmitSpec,
    pre: EmitPrecompute,
) -> FrameEmission:
    raw_order = list(perspective.slot_order.order)

    is_pass, flat_idx = policy_pass_target(sim, perspective.slot, t, game.W, W_PADDED)

    alive_mask = None
    if spec.emit_alive_mask:
        assert pre.player_status is not None
        alive_mask = make_alive_mask(pre.player_status, raw_order, t)

    aux_head_targets = None
    aux_spec = spec_for(spec.targets.elim_variant)
    if aux_spec is not None:
        assert pre.elim is not None
        # The aux-head specs return torch tensors. The tail is numpy throughout, so
        # we convert them back here.
        aux_head_targets = {
            key: tensor.numpy()
            for key, tensor in aux_spec.encode_targets(pre.elim, raw_order, t).items()
        }

    return FrameEmission(
        action_target=np.asarray(flat_idx, dtype=np.int64),
        is_pass=np.asarray(is_pass, dtype=np.bool_),
        value_target=np.asarray(value_target(perspective.placement), dtype=np.int64),
        alive_mask=alive_mask,
        aux_head_targets=aux_head_targets,
    )
