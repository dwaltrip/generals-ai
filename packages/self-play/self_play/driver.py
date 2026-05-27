"""Self-play convenience wrapper: two ModelAgents (same weights) on a 2-player map."""

from __future__ import annotations

from typing import Any

from bc.model import BCModel
import torch

from game_runner.runner import run_game
from self_play.agent import ModelAgent
import sim_core


def play_game(
    model: BCModel,
    static: Any,
    device: torch.device,
    max_turns: int = 2000,
    force_move: bool = False,
    sample: bool = False,
    temperature: float = 1.0,
    progress_interval: int = 50,
) -> tuple[sim_core.State, list[ModelAgent]]:
    """Play one 2-player self-play game. Returns (State, agents) for
    backward compatibility with the CLI and tests.
    """
    agents: list[ModelAgent] = [
        ModelAgent(
            model,
            perspective_slot=p,
            device=device,
            force_move=force_move,
            sample=sample,
            temperature=temperature,
        )
        for p in (0, 1)
    ]
    result = run_game(
        agents, static,
        max_turns=max_turns,
        progress_interval=progress_interval,
    )
    assert result.state is not None
    return result.state, agents
