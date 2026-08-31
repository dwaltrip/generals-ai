from __future__ import annotations

import numpy as np

from training.bc import bfs
from training.bc.datapipe.emit import emit_frame
from training.bc.datapipe.emit_spec import EmitSpec
from training.bc.datapipe.precompute import EmitPrecompute
from training.bc.datapipe.sample import FrameMeta, TrainingSample, pack_sample
from training.bc.datapipe.sim_types import GameMeta, PerspectiveMeta, SimFrame
from training.bc.datapipe.walk import WalkFrame
from training.bc.mask import build_board_mask
from training.bc.obs import MemoryState, build_obs
from training.shared.timing import timer


def encode_frame(
    sim: dict[str, np.ndarray],
    t: int,
    game: GameMeta,
    perspective: PerspectiveMeta,
    frame_meta: FrameMeta | None,
    vis: np.ndarray,
    state: MemoryState,
    bfs_cache: bfs.BFSCache,
    spec: EmitSpec,
    pre: EmitPrecompute,
) -> TrainingSample:
    # NOTE: encode_frame assumes step_memory was already called for the given `t`.
    sim_frame = SimFrame(sim=sim, t=t, slot_order=perspective.slot_order)
    # TODO: Pass `GameMeta` directly into build_obs and build_mask
    H = game.H
    W = game.W

    frame = WalkFrame(
        t=t,
        obs=build_obs(sim_frame, vis, state, bfs_cache, H, W),
        board_mask=build_board_mask(H, W),
    )

    with timer.section("encode_tail"):
        emission = emit_frame(sim, t, game, perspective, spec.partial, pre)
        return pack_sample(
            frame,
            emission,
            frame_meta,
            sim_frame if spec.attach_sim_frame else None,
        )
