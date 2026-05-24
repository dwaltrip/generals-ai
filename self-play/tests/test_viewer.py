"""Smoke test: drive a few step_tick calls with pass-only moves, then
render the resulting State through the viewer.

Validates the corpus → static → new_state → step_tick → build_html
pipeline end-to-end with zero model dependency. If this passes, the
viewer can render any game our self-play loop produces.
"""

import pytest

import sim_core

from replay_collector.config import DB_PATH
from self_play import seed_map, viewer


@pytest.fixture(scope="module")
def static_and_state():
    if not DB_PATH.exists():
        pytest.skip(f"replay DB not present at {DB_PATH}")
    ids = seed_map.list_two_player_replay_ids(limit=1)
    if not ids:
        pytest.skip("no 2-player replays with wire_data in corpus")
    static = seed_map.load_static_from_db(ids[0])
    state = sim_core.new_state(static)
    # 10 pass-only ticks — production still fires (land tick at 50 is the
    # only meaningful event, and we don't get there here), but army
    # counts stay essentially fixed and ownership doesn't change.
    for _ in range(10):
        ran = state.step_tick(moves=[], afks=[])
        assert ran
    return static, state


def test_build_html_from_state_smoke(static_and_state):
    static, state = static_and_state

    html = viewer.build_html_from_state("selfplay-smoke", state, static)

    assert state.timestep == 10
    assert state.snapshots_len == 11  # initial + 10 step_tick calls
    assert state.alive_count == 2

    # All template placeholders resolved — any unsubstituted `$IDENT`
    # would mean the rendered HTML is broken.
    assert "$REPLAY_ID" not in html
    assert "$DATA_JSON" not in html
    assert "$SLOT_COLOR_RULES" not in html

    # Sanity-check the embedded payload identifies our run + lists the
    # right player count.
    assert "selfplay-smoke" in html
    assert '"num_timesteps":11' in html
