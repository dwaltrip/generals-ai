"""MetricsCollector — on_tick callback that accumulates per-game metrics.

Wired into run_game via the on_tick callback. Collects:
  - Land/army curves sampled every N ticks
  - Move analysis via MoveAnalyzer
  - Policy-specific diagnostics (via duck-typed get_diagnostics)
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from eval.move_analyzer import MoveAnalyzer
from game_runner.policy import GameResult, Policy
import sim_core


class MetricsCollector:
    def __init__(self, num_players: int, sample_interval: int = 25):
        self.num_players = num_players
        self.sample_interval = sample_interval
        self.move_analyzer = MoveAnalyzer(num_players)

        # sampled curves: list of (timestep, per-player-land, per-player-army)
        self._land_curves: list[list[int]] = []
        self._army_curves: list[list[int]] = []
        self._curve_timesteps: list[int] = []

    def on_tick(
        self,
        state: sim_core.State,
        moves: list[tuple[int, int, int, int]],
        policies: Sequence[Policy],
    ) -> None:
        self.move_analyzer.record_moves(state, moves)

        if self.sample_interval > 0 and state.timestep % self.sample_interval == 0:
            self._sample_curves(state)

    def _sample_curves(self, state: sim_core.State) -> None:
        own = np.asarray(state.ownership)
        arm = np.asarray(state.armies)
        land_row = []
        army_row = []
        for p in range(self.num_players):
            mask = own == p
            land_row.append(int(mask.sum()))
            army_row.append(int((arm * mask).sum()))
        self._curve_timesteps.append(state.timestep)
        self._land_curves.append(land_row)
        self._army_curves.append(army_row)

    def finalize(
        self, result: GameResult, policies: Sequence[Policy],
    ) -> dict:
        metrics: dict = {}

        # move analysis
        metrics.update(self.move_analyzer.summary())

        # land/army curves
        metrics["curves"] = {
            "timesteps": self._curve_timesteps,
            "land": self._land_curves,
            "army": self._army_curves,
        }

        # policy diagnostics (duck-typed)
        policy_diags = []
        for p, policy in enumerate(policies):
            get_diag = getattr(policy, "get_diagnostics", None)
            if get_diag is not None:
                policy_diags.append({"player": p, **get_diag()})
        if policy_diags:
            metrics["policy_diagnostics"] = policy_diags

        return metrics
