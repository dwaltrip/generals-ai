"""Acceptance test for the model-config refactor.

The no-op fingerprints (smoke loss, parity) only prove behavior is *unchanged*.
This proves the capability the refactor exists for: two checkpoints at different
trunk widths load from their self-describing `arch` keys and play head-to-head in
one process — the batched runner groups by `model_key` and runs each distinct
architecture through its own forward. With width as a global constant this was
impossible; that's what blocked the width sweep.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from bc.model import BCModel
from bc.model_config import ModelConfig


def _save_arch_checkpoint(path: Path, cfg: ModelConfig) -> None:
    """Write a combined checkpoint (random init) that self-describes `cfg`."""
    model = BCModel(cfg)
    torch.save({"model": model.state_dict(), "arch": asdict(cfg), "epoch": 1}, path)


def _corpus_2p_replay_id() -> str | None:
    try:
        from game_runner.seed_map import list_replay_ids_by_player_count

        ids = list_replay_ids_by_player_count(2)
    except Exception:
        return None
    return ids[0] if ids else None


def test_two_widths_coexist_and_play_head_to_head(tmp_path: Path) -> None:
    replay_id = _corpus_2p_replay_id()
    if replay_id is None:
        pytest.skip("replay corpus not available")
    from bc.inference import BCModelHandle
    from eval_tools.policy_spec import parse_policy_spec
    from game_runner.batched import PendingGame, run_batched
    from game_runner.seed_map import load_static_from_db
    from self_play.nn_agent import NNAgent

    device = torch.device("cpu")
    wide = ModelConfig()  # default 128/128/160
    narrow = ModelConfig(outer_width=64, middle_width=64, inner_width=96)
    wide_ckpt, narrow_ckpt = tmp_path / "wide.pt", tmp_path / "narrow.pt"
    _save_arch_checkpoint(wide_ckpt, wide)
    _save_arch_checkpoint(narrow_ckpt, narrow)

    # Both load in one process, each reconstructing its own architecture.
    p0 = parse_policy_spec(f"checkpoint:{wide_ckpt}:force_move=true", slot=0, device=device)
    p1 = parse_policy_spec(f"checkpoint:{narrow_ckpt}:force_move=true", slot=1, device=device)
    # parse_policy_spec is typed to return the Policy protocol; narrow to the
    # concrete NN types so the .model_handle.model.cfg chain type-checks (and to
    # verify the reconstruction actually produced them).
    assert isinstance(p0, NNAgent) and isinstance(p1, NNAgent)
    h0, h1 = p0.model_handle, p1.model_handle
    assert isinstance(h0, BCModelHandle) and isinstance(h1, BCModelHandle)
    assert h0.model.cfg == wide
    assert h1.model.cfg == narrow
    # Distinct arch-bearing checkpoints → distinct model_keys → the runner runs
    # each through its own forward rather than collapsing them into one.
    assert h0.model_key != h1.model_key

    static = load_static_from_db(replay_id)
    pending = [PendingGame(game_id=1, map_data=static, policies=[p0, p1])]
    finished = list(run_batched(iter(pending), pool_size=1, max_turns=40))

    assert len(finished) == 1
    result = finished[0].result
    assert result.game_length > 0
    assert result.winner in (0, 1, None)
