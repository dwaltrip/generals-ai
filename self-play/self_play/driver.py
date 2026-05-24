"""play_game — drive two ModelAgents through a full 2-player self-play game.

Both agents share the same nn.Module (same weights → self-play of a
single policy from two perspectives), each carrying its own per-
perspective MemoryState / BFSCache / sim buffers.

Termination conditions:
  - alive_count <= 1 (winner declared)
  - timestep >= max_turns (caller-imposed safety cap; the sim has its
    own MAX_GAME_TIMESTEPS and MAX_ALL_AFK_TIMESTEPS internal caps,
    but those are very loose — 50k and 2k respectively)

A move with source == -1 means the agent passes; we filter those out
of the per-tick moves list rather than submitting them to step_tick
(which would do nothing with them anyway, but the empty submission
form is clearer).
"""

from __future__ import annotations

from typing import Any

import torch

import sim_core

from self_play.agent import ModelAgent
from bc.model import BCModel


def play_game(
    model: BCModel,
    static: Any,
    device: torch.device,
    max_turns: int = 2000,
    max_turns_hint: int = 1000,
) -> sim_core.State:
    """Play one 2-player self-play game to termination. Returns the final
    State (with full snapshot history, event lists, etc. ready for the
    viewer).
    """
    state = sim_core.new_state(static)
    if state.num_players != 2:
        raise ValueError(
            f"play_game is 2-player only; static has {state.num_players} players"
        )

    agents = [
        ModelAgent(model, perspective_slot=p, device=device,
                   max_turns_hint=max_turns_hint)
        for p in (0, 1)
    ]
    for ag in agents:
        ag.init_for_game(state, static)

    while state.alive_count > 1 and state.timestep < max_turns:
        moves: list[tuple[int, int, int, int]] = []
        for p, ag in enumerate(agents):
            if not state.alive[p]:
                continue
            src, dst, is50 = ag.act(state, static)
            if src == -1:
                continue  # pass
            moves.append((p, src, dst, is50))
        state.step_tick(moves=moves, afks=[])

    return state
