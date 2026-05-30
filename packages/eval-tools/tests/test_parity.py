"""Behavior-preservation oracle for the ModelAgent -> bc.inference refactor.

Runs a fully deterministic NN-vs-NN game (CPU, argmax, force_move) through the
standard construction path (`parse_policy_spec`) and compares a fingerprint —
the full submitted-move sequence plus final winner / length / per-player
land+army — against a committed snapshot.

The fingerprint is captured on the *pre-refactor* code and frozen. Because the
agents are built via `parse_policy_spec`, the refactor can swap ModelAgent for
NNAgent under the factory without touching this test: the game must stay
byte-identical.

CPU + sample=False is deliberate: CPU conv is run-to-run deterministic (unlike
MPS), and argmax removes RNG, so any divergence is a real behavior change.

Regenerate the snapshot (only when an intentional behavior change is expected):
    uv run python packages/eval-tools/tests/test_parity.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch


_REPO_ROOT = Path(__file__).resolve().parents[3]
_CKPT = _REPO_ROOT / "training/data/runs-cloud/2026-05-28T08-21-33Z/checkpoints/epoch_005.pt"
_FINGERPRINT = Path(__file__).parent / "parity_fingerprint.json"

# force_move so the agents actually play (deterministic argmax moves), pyramid
# value head to match the checkpoint this fingerprint was captured with.
_SPEC = f"checkpoint:{_CKPT}:force_move=true,value_head_variant=pyramid"
_MAX_TURNS = 150


def _corpus_replay_id() -> str | None:
    """First 2-player replay id in the local corpus, or None if unavailable."""
    try:
        from game_runner.seed_map import list_replay_ids_by_player_count

        ids = list_replay_ids_by_player_count(2)
    except Exception:
        return None
    return ids[0] if ids else None


def _compute_fingerprint(replay_id: str) -> dict:
    from eval_tools.policy_spec import parse_policy_spec
    from game_runner.runner import run_game
    from game_runner.seed_map import load_static_from_db

    torch.manual_seed(0)
    device = torch.device("cpu")
    static = load_static_from_db(replay_id)
    policies = [parse_policy_spec(_SPEC, slot=s, device=device) for s in range(2)]

    moves_log: list[list[list[int]]] = []

    def on_tick(state, moves, policies) -> None:  # noqa: ARG001
        moves_log.append([[int(x) for x in m] for m in moves])

    result = run_game(policies, static, max_turns=_MAX_TURNS, on_tick=on_tick)
    return {
        "replay_id": replay_id,
        "max_turns": _MAX_TURNS,
        "winner": result.winner,
        "game_length": result.game_length,
        "player_stats": [[ps.land, ps.army] for ps in result.player_stats],
        "moves": moves_log,
    }


@pytest.mark.skipif(not _CKPT.exists(), reason="parity checkpoint not present")
def test_game_is_byte_identical_to_fingerprint() -> None:
    if not _FINGERPRINT.exists():
        pytest.skip("no committed fingerprint; run this file as __main__ to capture")
    expected = json.loads(_FINGERPRINT.read_text())
    replay_id = expected["replay_id"]
    if _corpus_replay_id() is None:
        pytest.skip("replay corpus not available")
    got = _compute_fingerprint(replay_id)
    assert got == expected


if __name__ == "__main__":
    rid = _corpus_replay_id()
    if rid is None or not _CKPT.exists():
        raise SystemExit("cannot capture: checkpoint or corpus unavailable")
    fp = _compute_fingerprint(rid)
    _FINGERPRINT.write_text(json.dumps(fp, indent=2))
    print(f"wrote {_FINGERPRINT}  (replay_id={rid}, "
          f"winner={fp['winner']}, length={fp['game_length']}, "
          f"ticks_logged={len(fp['moves'])})")
