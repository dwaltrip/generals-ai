from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from training.bc import bfs
from training.bc.datapipe.sim_types import PerspectiveMeta, SimFrame, SimGame
from training.bc.mask import build_board_mask
from training.bc.obs import build_obs, init_memory, step_memory
from training.bc.obs_config import ObsConfig
from training.bc.slots import SlotOrder
from training.bc.visibility import compute_visibility


@dataclass(frozen=True)
class WalkFrame:
    t: int
    obs: np.ndarray
    board_mask: np.ndarray


class PerspectiveWalk:
    # Frames must be requested in tick order, starting at t=0. Each call to
    # `frame(t)` advances the memory state in place.
    def __init__(
        self,
        obs_cfg: ObsConfig,
        game: SimGame,
        slot_order: SlotOrder,
    ) -> None:
        self._game = game
        self._slot_order = slot_order
        self._state = init_memory(game, slot_order.perspective, game.H, game.W, obs_cfg)
        self._bfs_cache = bfs.init_bfs_cache()
        self._board_mask = build_board_mask(game.H, game.W)
        self._next_t = 0

    def frame(self, t: int) -> WalkFrame:
        assert t == self._next_t, f"out of order: expected t={self._next_t}, got t={t}"
        self._next_t += 1

        game, slot = self._game, self._slot_order.perspective
        H, W = game.H, game.W

        vis = compute_visibility(game.ownership[t], slot, H, W)
        step_memory(self._state, game, t, vis, slot, H, W)
        sim_frame = SimFrame(sim=game, t=t, slot_order=self._slot_order)
        return WalkFrame(
            t=t,
            obs=build_obs(sim_frame, vis, self._state, self._bfs_cache, H, W),
            board_mask=self._board_mask,
        )


def walk(
    game: SimGame,
    perspective: PerspectiveMeta,
    obs_cfg: ObsConfig,
) -> Iterator[WalkFrame]:
    w = PerspectiveWalk(obs_cfg, game, perspective.slot_order)
    for t in range(perspective.end_t):
        yield w.frame(t)
