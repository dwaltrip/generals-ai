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

import pytest

from game_runner import seed_map, viewer
from self_play import driver
from settings import DB_PATH, RUNS_CLOUD_DIR
from training.bc.inference import BCModelHandle, default_device


_CHECKPOINT = RUNS_CLOUD_DIR / "2026-05-23T23-40-14Z" / "checkpoints" / "epoch_005.pt"
_TMP_DIR = Path(__file__).resolve().parent.parent / "tmp"


@pytest.fixture(scope="module")
def loaded_handle():
    if not _CHECKPOINT.exists():
        pytest.skip(f"checkpoint not present at {_CHECKPOINT}")
    return BCModelHandle.load(_CHECKPOINT, default_device())


def test_play_game_terminates_and_renders(loaded_handle):
    if not DB_PATH.exists():
        pytest.skip(f"replay DB not present at {DB_PATH}")
    ids = seed_map.list_two_player_replay_ids(limit=1)
    if not ids:
        pytest.skip("no 2-player replays with wire_data in corpus")
    static = seed_map.load_static_from_db(ids[0])

    state, agents = driver.play_game(
        loaded_handle, static, max_turns=200, force_move=True,
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
