"""End-to-end smoke test: drive a full self-play game.

Plays a capped-length game with the BC checkpoint and verifies the
loop terminates cleanly with a renderable State. Aggressive max_turns
cap (200) because the BC-only checkpoint may not produce decisive
play — we mainly want to confirm the driver doesn't deadlock and the
viewer renders the output.

Also writes the rendered HTML to tmp/ so it can be opened in a
browser for manual visual check.
"""

from pathlib import Path

from bc.checkpoint import load_bc_model
import pytest

from game_runner import seed_map, viewer
from replay_collector.config import DB_PATH
from self_play import agent, driver


# packages/self-play/tests/ → repo root is 3 levels up.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CHECKPOINT = _REPO_ROOT / "training" / "data" / "runs-cloud" \
    / "2026-05-23T23-40-14Z" / "checkpoints" / "epoch_005.pt"
_TMP_DIR = _REPO_ROOT / "packages" / "self-play" / "tmp"


@pytest.fixture(scope="module")
def loaded_model():
    if not _CHECKPOINT.exists():
        pytest.skip(f"checkpoint not present at {_CHECKPOINT}")
    device = agent.default_device()
    model = load_bc_model(_CHECKPOINT, device)
    return model, device


def test_play_game_terminates_and_renders(loaded_model):
    model, device = loaded_model
    if not DB_PATH.exists():
        pytest.skip(f"replay DB not present at {DB_PATH}")
    ids = seed_map.list_two_player_replay_ids(limit=1)
    if not ids:
        pytest.skip("no 2-player replays with wire_data in corpus")
    static = seed_map.load_static_from_db(ids[0])

    state, agents = driver.play_game(
        model, static, device=device, max_turns=200, force_move=True,
    )

    # Termination: either one player won or we hit max_turns.
    assert state.alive_count <= 2
    assert state.timestep > 0
    assert state.timestep <= 200
    # snapshots_len = initial snapshot + one per step_tick call
    assert state.snapshots_len == state.timestep + 1
    # force_move=True means each agent should have actually moved.
    for ag in agents:
        assert ag.n_moved > 0, "force_move agent should have produced moves"

    # Renderable: build_html_from_state shouldn't raise, output well-formed.
    html = viewer.build_html_from_state("selfplay-smoke-driver", state, static)
    assert "$REPLAY_ID" not in html
    assert "selfplay-smoke-driver" in html

    _TMP_DIR.mkdir(exist_ok=True)
    out = _TMP_DIR / "selfplay-driver-smoke.html"
    out.write_text(html)
    print(f"\nwrote: {out}")
