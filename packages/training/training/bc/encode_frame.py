from __future__ import annotations

import numpy as np
import torch

from training.bc import bfs
from training.bc.aux_heads.registry import spec_for
from training.bc.constants import H_PADDED, W_PADDED
from training.bc.emit_spec import EmitSpec
from training.bc.mask import build_mask
from training.bc.obs import (
    MemoryState,
    build_obs,
)
from training.bc.player_status import make_alive_mask
from training.bc.sample import FrameMeta, TrainingSample
from training.bc.sim_types import PerspectiveMeta, SimFrame
from training.bc.targets.core_targets import policy_pass_target, value_target
from training.bc.targets.elim_targets import ElimCtx
from training.shared.timing import timer


def encode_frame(
    sim: dict[str, np.ndarray],
    t: int,
    perspective: PerspectiveMeta,
    frame_meta: FrameMeta | None,
    vis: np.ndarray,
    state: MemoryState,
    bfs_cache: bfs.BFSCache,
    spec: EmitSpec,
    elim_ctx: ElimCtx | None = None,
) -> TrainingSample:
    """
    Training samples are single ticks from one player's perspective in a game.
    i.e. A tuple of (game, perspective, timestep) maps directly to one `TrainingSample`.

    Orchestrates:
        - `build_obs`
        - `build_mask` (move legality, stateless)
        - `policy_pass_target` (the action targets, used by the policy and pass heads)
        - `value_target`

    `elim_ctx` is the game-level precompute for the alive mask and elim-head
    targets; required when the spec asks for either.

    For sequentially encoded frames (e.g. during a dataset walk), encode_frame
    assumes that `step_memory` has already been called for this `t`.
    """

    sim_frame = SimFrame(sim=sim, t=t, slot_order=perspective.slot_order)
    perspective_slot = perspective.slot
    H = int(sim["map_height"])
    W = int(sim["map_width"])

    obs_np = build_obs(sim_frame, vis, state, bfs_cache, H, W)
    mask_np = build_mask(sim, t, perspective_slot, H, W)

    # encode_tail: The "rest" of encode_frame after build_obs (build_mask is very minor).
    # Mask + action target + value target + numpy->tensor conversion.
    # Time spent is dominated by the tensor conversions feeding collation.
    with timer.section("encode_tail"):
        # Mask for valid cells: part of this game's actual board (the unpadded part).
        # Needed by the pass head and the value head. Shape: [1, H_PADDED, W_PADDED]
        valid_mask_np = np.zeros((1, H_PADDED, W_PADDED), dtype=np.bool_)
        valid_mask_np[0, :H, :W] = True

        is_pass, flat_idx = policy_pass_target(sim, perspective_slot, t, W, W_PADDED)

        alive_mask = None
        aux_head_targets = None
        if spec.emit_alive_mask or spec.targets.elim_variant is not None:
            assert elim_ctx is not None
            raw_order = list(perspective.slot_order.order)
            if spec.emit_alive_mask:
                alive_mask = torch.from_numpy(make_alive_mask(elim_ctx, raw_order, t))
            aux_spec = spec_for(spec.targets.elim_variant)
            if aux_spec is not None:
                aux_head_targets = aux_spec.encode_targets(elim_ctx, raw_order, t)

        sample = TrainingSample(
            obs=torch.from_numpy(obs_np),
            mask=torch.from_numpy(mask_np),
            valid_mask=torch.from_numpy(valid_mask_np),
            action_target=torch.tensor(flat_idx, dtype=torch.int64),
            is_pass=torch.tensor(is_pass, dtype=torch.bool),
            value_target=torch.tensor(value_target(perspective.placement), dtype=torch.int64),
            frame_meta=frame_meta,
            sim_frame=(sim_frame if spec.attach_sim_frame else None),
            alive_mask=alive_mask,
            aux_head_targets=aux_head_targets,
        )

        return sample
