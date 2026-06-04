"""EvalBotAgent — Policy-compatible wrapper around EvalBot + WorldModel.

Satisfies the game_runner.Policy protocol so EvalBot can be used with
run_game without changing EvalBot's own interface.
"""

from __future__ import annotations

from eval_bot.bot import EvalBot
from eval_bot.bot_config import BotConfig
from eval_bot.world_model import WorldModel
from game_types import StaticMap
import sim_core


class EvalBotAgent:
    def __init__(
        self,
        perspective_slot: int,
        cfg: BotConfig | None = None,
    ):
        self.perspective_slot = perspective_slot
        self.cfg = cfg or BotConfig()
        self._bot = EvalBot(perspective_slot=perspective_slot, cfg=self.cfg)
        self._world = WorldModel(perspective_slot=perspective_slot, cfg=self.cfg)

    def init_for_game(self, state: sim_core.State, map_data: StaticMap) -> None:
        self._world.init_for_game(state, map_data)
        self._bot.init_for_game(state, map_data)

    def act(self, state: sim_core.State, map_data: StaticMap) -> tuple[int, int, int]:
        view = self._world.update(state, map_data)
        return self._bot.act(view)

    @property
    def n_moved(self) -> int:
        return self._bot.n_moved

    @property
    def n_passed(self) -> int:
        return self._bot.n_passed
