"""Smoke test for the per-game intermediate output writers.

This is a smoke test, not an integration test — we assert structural
properties (keys, shapes, dtypes) + a few load-bearing spot checks, but
do NOT pin the writer's byte output the way `test_sim_integration` pins
the sim's. The proper fixture-regression test for `write_sim_output` is
deferred until the writer's output has been hand-validated on a few
replays.

Run with `-s` to print on-disk byte size per sample (handoff §7).
"""
from pathlib import Path

import numpy as np
import pytest

from replay_parser.metadata import build_metadata
from replay_parser.output import write_metadata, write_sim_output
from replay_parser.parser import parse_replay

from _fixture_lib import FIXTURES_DIR


SMOKE_FIXTURES = [
    "multi_surrender_a",           # neutralize events (non-initial cities_present_at)
    "capture_during_surrender_a",  # surrender-then-captured (dead-then-captured)
    "run_of_the_mill_a",           # interleaved pass frames in actions
]

SIM_KEYS = {
    "replay_id", "version", "map_width", "map_height",
    "mountains",
    "initial_cities", "initial_city_armies",
    "initial_neutrals", "initial_neutral_armies",
    "initial_generals",
    "ownership", "armies",
    "cities", "cities_present_at",
    "death_events", "capture_events", "neutralize_events",
    "actions_source", "actions_dest", "actions_is50",
}

META_KEYS = {
    "replay_id", "sim_core_version",
    "perspective_player_ids", "perspective_usernames",
    "placement", "stars_at_start",
    "elim_timestep",
    "rolling_1st_rate", "rolling_top3_rate", "prior_games_count",
}


@pytest.mark.parametrize("name", SMOKE_FIXTURES)
def test_writer_smoke(name: str, tmp_path: Path):
    wire_path = FIXTURES_DIR / name / "wire.bin"
    if not wire_path.exists():
        pytest.skip(f"missing wire.bin: {wire_path}")

    state, replay = parse_replay(wire_path.read_bytes())

    sim_path = tmp_path / f"{replay.static.id}.npz"
    meta_path = tmp_path / f"{replay.static.id}.meta.npz"

    write_sim_output(state, replay, sim_path)

    # All players in this game serve as "perspectives" for the smoke test —
    # the real perspective resolution lives in the deferred corpus driver.
    P = state.num_players
    perspective_ids = list(range(P))
    placement = list(range(1, P + 1))  # placeholder; the real builder reads DB
    meta = build_metadata(
        state, replay, perspective_ids, placement,
        sim_core_version="smoke-test",
    )
    write_metadata(meta, meta_path)

    sim = dict(np.load(sim_path))
    meta_loaded = dict(np.load(meta_path))

    _check_sim_output(name, state, replay, sim)
    _check_meta(meta_loaded, replay, P)

    # Sibling-file replay_id cross-check (the join key from §1).
    assert str(sim["replay_id"]) == str(meta_loaded["replay_id"]) == replay.static.id

    print(
        f"\n[{name}] replay_id={replay.static.id} "
        f"sim={sim_path.stat().st_size:,}B "
        f"meta={meta_path.stat().st_size:,}B"
    )


def _check_sim_output(name: str, state, replay, sim: dict) -> None:
    actual_keys = set(sim)
    assert actual_keys == SIM_KEYS, (
        f"[{name}] key drift: extra={actual_keys - SIM_KEYS} "
        f"missing={SIM_KEYS - actual_keys}"
    )

    P = state.num_players
    T = state.snapshots_len
    HW = replay.static.map_width * replay.static.map_height
    C_final = len(state.cities)
    C0 = len(replay.static.initial_cities)

    assert sim["ownership"].shape == (T, HW)
    assert sim["ownership"].dtype == np.int8
    assert sim["armies"].shape == (T, HW)
    assert sim["armies"].dtype == np.int16

    assert sim["cities"].shape == (C_final,)
    assert sim["cities_present_at"].shape == (C_final,)
    # Initial cities are present from snapshot 0.
    assert np.all(sim["cities_present_at"][:C0] == 0), (
        f"[{name}] expected first {C0} cities_present_at == 0"
    )
    # All entries fall within [0, T-1] (snapshot index, present at <= t).
    assert sim["cities_present_at"].min() >= 0
    assert sim["cities_present_at"].max() <= T - 1

    assert sim["actions_source"].shape == (P, T - 1)
    assert sim["actions_dest"].shape == (P, T - 1)
    assert sim["actions_is50"].shape == (P, T - 1)
    assert sim["actions_source"].dtype == np.int16
    assert sim["actions_dest"].dtype == np.int16
    assert sim["actions_is50"].dtype == np.int8

    # is50: -1 for pass; 0/1 for recorded move. No other values.
    is50 = sim["actions_is50"]
    assert np.isin(is50, np.array([-1, 0, 1], dtype=np.int8)).all(), (
        f"[{name}] unexpected actions_is50 values: {np.unique(is50)}"
    )

    # Pass / non-pass consistency across the three action arrays: a pass on
    # one means a pass on all three.
    pass_mask = sim["actions_source"] == -1
    assert np.array_equal(pass_mask, sim["actions_dest"] == -1)
    assert np.array_equal(pass_mask, sim["actions_is50"] == -1)

    # Recorded sources/dests are within valid tile range.
    nonpass = ~pass_mask
    if nonpass.any():
        assert (sim["actions_source"][nonpass] >= 0).all()
        assert (sim["actions_source"][nonpass] < HW).all()
        assert (sim["actions_dest"][nonpass] >= 0).all()
        assert (sim["actions_dest"][nonpass] < HW).all()

    # Scenario-specific spot checks.
    if name == "multi_surrender_a":
        # Neutralize event ⇒ at least one city added mid-game.
        assert (sim["cities_present_at"] > 0).any(), (
            "expected at least one non-initial city (neutralize)"
        )
        assert sim["neutralize_events"].shape[0] >= 1

    if name == "capture_during_surrender_a":
        # A captured player whose death precedes capture (alive=False
        # before being captured). death.timestep <= capture.timestep for
        # the same player.
        deaths = {int(t): int(p) for t, p in sim["death_events"]}
        found = False
        for t_cap, _captor, captured in sim["capture_events"]:
            for t_death, dead_p in deaths.items():
                if dead_p == int(captured) and t_death <= int(t_cap):
                    found = True
                    break
            if found:
                break
        assert found, "expected dead-then-captured player in this fixture"

    if name == "run_of_the_mill_a":
        # Interleaved pass + recorded moves exist somewhere in the matrix.
        assert pass_mask.any(), "expected at least one pass frame"
        assert nonpass.any(), "expected at least one recorded move"


def _check_meta(meta: dict, replay, P: int) -> None:
    actual_keys = set(meta)
    assert actual_keys == META_KEYS, (
        f"meta key drift: extra={actual_keys - META_KEYS} "
        f"missing={META_KEYS - actual_keys}"
    )

    K = P  # smoke uses all players as perspectives
    assert meta["perspective_player_ids"].shape == (K,)
    assert meta["perspective_usernames"].shape == (K,)
    assert meta["perspective_usernames"].dtype.kind == "U"
    # Usernames round-trip from wire.static.usernames via perspective_player_ids.
    expected_names = [replay.static.usernames[p] for p in meta["perspective_player_ids"].tolist()]
    assert meta["perspective_usernames"].tolist() == expected_names
    assert meta["placement"].shape == (K,)
    assert meta["stars_at_start"].shape == (K,)
    assert meta["elim_timestep"].shape == (K,)
    assert meta["rolling_1st_rate"].shape == (K,)
    assert meta["rolling_top3_rate"].shape == (K,)
    assert meta["prior_games_count"].shape == (K,)

    # Stubbed sentinels for the deferred DB-walk fields.
    assert np.all(meta["rolling_1st_rate"] == -1.0)
    assert np.all(meta["rolling_top3_rate"] == -1.0)
    assert np.all(meta["prior_games_count"] == -1)

    # stars_at_start round-trips from wire.
    expected_stars = np.asarray(replay.static.stars, dtype=np.float32)
    np.testing.assert_array_equal(meta["stars_at_start"], expected_stars)
