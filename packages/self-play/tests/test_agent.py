"""Smoke test: load the BC checkpoint and run a single inference tick.

Builds a ModelAgent, seeds a 2-player game, then either passes or
sends moves through enough step_tick calls that armies grow to >= 2
(generals have army=1 at start, so no move is legal until production
fires). Then calls agent.act() and verifies the returned wire move
is either a pass or is legal under the mask.

The checkpoint path is wired to the run the user pointed at:
training/data/runs-cloud/2026-05-23T23-40-14Z/checkpoints/epoch_005.pt
"""

from pathlib import Path

import pytest
import torch

import sim_core

from replay_collector.config import DB_PATH
from game_runner import seed_map
from self_play import agent


# Repo-rooted absolute path so the test works from any CWD.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHECKPOINT = _REPO_ROOT / "training" / "data" / "runs-cloud" \
    / "2026-05-23T23-40-14Z" / "checkpoints" / "epoch_005.pt"


@pytest.fixture(scope="module")
def loaded_model():
    if not _CHECKPOINT.exists():
        pytest.skip(f"checkpoint not present at {_CHECKPOINT}")
    device = agent.default_device()
    model = agent.load_checkpoint(_CHECKPOINT, device)
    return model, device


@pytest.fixture(scope="module")
def seeded_game():
    if not DB_PATH.exists():
        pytest.skip(f"replay DB not present at {DB_PATH}")
    ids = seed_map.list_two_player_replay_ids(limit=1)
    if not ids:
        pytest.skip("no 2-player replays with wire_data in corpus")
    static = seed_map.load_static_from_db(ids[0])
    state = sim_core.new_state(static)
    return state, static


def test_load_checkpoint_smoke(loaded_model):
    model, device = loaded_model
    assert isinstance(model, torch.nn.Module)
    # Sanity: forward a zero obs to confirm the loaded weights produce
    # the expected output shapes (size of the policy/pass/value heads).
    from bc.constants import OBS_CHANNELS, H_PADDED, W_PADDED
    obs = torch.zeros((1, OBS_CHANNELS, H_PADDED, W_PADDED), device=device)
    valid_mask = torch.ones((1, 1, H_PADDED, W_PADDED), dtype=torch.bool, device=device)
    with torch.no_grad():
        out = model(obs, valid_mask)
    assert out["policy_logits"].shape == (1, 8, H_PADDED, W_PADDED)
    assert out["pass_logit"].shape == (1,)
    assert out["value_logits"].shape == (1, 8)


def test_single_tick_inference(loaded_model, seeded_game):
    model, device = loaded_model
    state, static = seeded_game

    ag = agent.ModelAgent(model, perspective_slot=0, device=device)
    ag.init_for_game(state, static)

    # Advance with no moves until at least one player has a tile with
    # army >= 2 (production fires every 2 ticks for generals/cities, so
    # by t=2 the general has army=2 and a move becomes legal).
    for _ in range(5):
        state.step_tick(moves=[], afks=[])

    src, dst, is50 = ag.act(state, static)

    if src == -1:
        # The pass head fired, or no legal moves existed. The mask check
        # below would have surfaced "no legal moves" via len(static.mountains)
        # type assertions; here we just record the pass and exit.
        assert (dst, is50) == (-1, -1)
        return

    # Returned move must be legal under the mask we'd build for this tick.
    from bc import mask as bc_mask
    from bc.actions import encode
    from bc.constants import W_PADDED
    sim_dict = ag._sim  # internal — fine for a smoke test
    m = bc_mask.build_mask(sim_dict, state.timestep, 0, static.map_height, static.map_width)
    is_pass, flat_idx = encode(src, dst, is50, static.map_width, W_PADDED)
    assert not is_pass
    # Verify the chosen flat_idx is True in the mask.
    # mask is [H_PAD, W_PAD, 8] — reshape to [H_PAD*W_PAD*8] flat (cell-major)
    flat_mask = m.reshape(-1)
    assert flat_mask[flat_idx], (
        f"agent returned move (src={src}, dst={dst}, is50={is50}) "
        f"→ flat_idx={flat_idx} which is False in the legality mask"
    )
