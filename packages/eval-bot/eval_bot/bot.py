"""EvalBot — hand-coded greedy agent for evaluating NN training progress.

Same duck-typed interface as ModelAgent in self_play.agent:
  - init_for_game(state, static) — called once per game
  - act(view) — called once per tick with the current PlayerView

Milestone 1: ExpandOrExplore only. Future milestones add Defend,
Kill-shot, Attack gates per the spec.
"""

from __future__ import annotations

from typing import Any

import sim_core

from eval_bot.expand import MOVE_EXPAND, MOVE_EXPLORE, pick_expand_or_explore
from eval_bot.plan import Plan
from eval_bot.world_model import PlayerView, WorldModel


class EvalBot:
    def __init__(
        self,
        perspective_slot: int,
        max_ticks_hint: int = 2000,
    ):
        self.perspective_slot = perspective_slot
        self.world = WorldModel(perspective_slot, max_ticks_hint)
        self._active_plan: Plan | None = None

        self.n_passed = 0
        self.n_moved = 0
        self.n_expand = 0
        self.n_explore = 0
        self.n_no_move = 0

    def init_for_game(self, state: sim_core.State, static: Any) -> None:
        self.world.init_for_game(state, static)
        self._active_plan = None

        self.n_passed = 0
        self.n_moved = 0
        self.n_expand = 0
        self.n_explore = 0
        self.n_no_move = 0

    def act(self, view: PlayerView) -> tuple[int, int, int]:
        """Decide one move for the current tick.

        Returns (source, dest, is50) or (-1, -1, -1) for pass.
        """
        result = pick_expand_or_explore(view)

        if result is None:
            self.n_no_move += 1
            return (-1, -1, -1)

        src, dst, is50, mode = result
        self.n_moved += 1
        if mode == MOVE_EXPAND:
            self.n_expand += 1
        elif mode == MOVE_EXPLORE:
            self.n_explore += 1
        return (src, dst, is50)
