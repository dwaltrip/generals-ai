"""run_game — drive N policies through a full generals.io game.

Accepts any objects satisfying the Policy protocol (init_for_game + act).
The game loop is map-agnostic and player-count-agnostic.

Termination conditions:
  - alive_count <= 1 (winner declared)
  - timestep >= max_turns (caller-imposed safety cap; the sim has its
    own MAX_GAME_TIMESTEPS and MAX_ALL_AFK_TIMESTEPS internal caps,
    but those are very loose — 50k and 2k respectively)

A move with source == -1 means the agent passes; we filter those out
of the per-tick moves list rather than submitting them to step_tick.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from game_runner.policy import GameResult, PlayerStats, Policy
import sim_core


def run_game(
    policies: Sequence[Policy],
    static: Any,
    max_turns: int = 2000,
    progress_interval: int = 0,
) -> GameResult:
    state = sim_core.new_state(static)
    num_players = state.num_players
    if len(policies) != num_players:
        raise ValueError(
            f"expected {num_players} policies for this map, got {len(policies)}"
        )

    for policy in policies:
        policy.init_for_game(state, static)

    while state.alive_count > 1 and state.timestep < max_turns:
        moves: list[tuple[int, int, int, int]] = []
        for p, policy in enumerate(policies):
            if not state.alive[p]:
                continue
            src, dst, is50 = policy.act(state, static)
            if src == -1:
                continue
            moves.append((p, src, dst, is50))
        state.step_tick(moves=moves, afks=[])

        if progress_interval > 0 and state.timestep % progress_interval == 0:
            _print_progress(state, policies)

    return _build_result(state, policies)


def _build_result(
    state: sim_core.State, policies: Sequence[Policy],
) -> GameResult:
    own = np.asarray(state.ownership)
    arm = np.asarray(state.armies)

    winner: int | None = None
    if state.alive_count == 1:
        for p in range(state.num_players):
            if state.alive[p]:
                winner = p
                break

    player_stats = []
    for p in range(state.num_players):
        mask = own == p
        stats = PlayerStats(
            land=int(mask.sum()),
            army=int((arm * mask).sum()),
            n_moved=getattr(policies[p], "n_moved", 0),
            n_passed=getattr(policies[p], "n_passed", 0),
        )
        player_stats.append(stats)

    return GameResult(
        winner=winner,
        game_length=state.timestep,
        player_stats=player_stats,
        state=state,
    )


def _print_progress(state: sim_core.State, policies: Sequence[Policy]) -> None:
    own = np.asarray(state.ownership)
    arm = np.asarray(state.armies)
    alive_chr = "".join("T" if a else "F" for a in state.alive)
    parts = []
    for p in range(state.num_players):
        mask = own == p
        land = int(mask.sum())
        army = int((arm * mask).sum())
        parts.append(f"p{p}: land={land:3d} army={army:4d}")
    print(f"[t={state.timestep:4d}] alive=[{alive_chr}] {' | '.join(parts)}")
