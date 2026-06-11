"""run_analysis tests: span/placement edge cases that can silently lie, plus
an end-to-end smoke on a synthetic two-game run dir.

The synthetic game is a 3-player 2x3 map small enough to hand-compute every
share and rank; the smoke asserts spot-check values, not just artifact
existence.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from eval_tools.run_analysis import extract, loader


SPEC_A = "checkpoint:data/training/runs-cloud/2026-06-07T08-41-55Z/checkpoints/epoch_004.pt:sample"
SPEC_B = "checkpoint:data/training/runs-cloud/2026-06-10T09-20-33Z/checkpoints/epoch_006.pt:sample"


def _rec(
    death_events: list[list[int]],
    winner_seat: int | None,
    game_length: int,
    num_players: int = 3,
    ownership: np.ndarray | None = None,
    armies: np.ndarray | None = None,
    capture_events: list[list[int]] | None = None,
    game_index: int = 1,
    slot_map: list[int] | None = None,
) -> loader.GameRecord:
    T1 = game_length + 1
    hw = 6  # 2x3 map
    if ownership is None:
        ownership = np.zeros((T1, hw), dtype=np.int8)
        for p in range(num_players):
            ownership[:, p] = p
    if armies is None:
        armies = np.ones_like(ownership, dtype=np.int16)
    return loader.GameRecord(
        game_index=game_index,
        game_id=f"game_{game_index:03d}",
        replay_id="map_x",
        slot_map=slot_map or list(range(num_players)),
        winner_seat=winner_seat,
        game_length=game_length,
        player_stats=[{"land": 1, "army": 1, "n_moved": 0, "n_passed": 0}] * num_players,
        metrics={},
        map_width=3,
        map_height=2,
        ownership=ownership,
        armies=armies,
        cities=np.array([], dtype=np.int32),
        cities_present_at=np.array([], dtype=np.int32),
        death_events=np.array(death_events, dtype=np.int32).reshape(-1, 2),
        capture_events=np.array(capture_events or [], dtype=np.int32).reshape(-1, 3),
    )


def test_alive_spans_basic():
    rec = _rec(death_events=[[40, 2], [90, 1]], winner_seat=0, game_length=90)
    spans = extract.alive_spans(rec)
    assert [(s.stage, s.t_start, s.t_end) for s in spans] == [
        (3, 0, 40),
        (2, 41, 90),
    ]


def test_alive_spans_same_tick_deaths_skip_zero_length():
    rec = _rec(
        death_events=[[40, 3], [40, 2], [70, 1]],
        winner_seat=0,
        game_length=70,
        num_players=4,
    )
    spans = extract.alive_spans(rec)
    # The 3-alive stage never had a snapshot: both deaths landed on tick 40.
    assert [(s.stage, s.t_start, s.t_end) for s in spans] == [
        (4, 0, 40),
        (2, 41, 70),
    ]


def test_alive_spans_draw_has_open_final_span():
    rec = _rec(death_events=[[40, 2]], winner_seat=None, game_length=100)
    spans = extract.alive_spans(rec)
    assert spans[-1].stage == 2 and spans[-1].t_end == 100


def test_placements_death_order_and_draw_ties():
    rec = _rec(death_events=[[40, 2], [90, 1]], winner_seat=0, game_length=90)
    assert extract.placements(rec) == [1.0, 2.0, 3.0]

    draw = _rec(death_events=[[40, 2]], winner_seat=None, game_length=100)
    # seats 0 and 1 survive a draw → tie-share places 1 and 2
    assert extract.placements(draw) == [1.5, 1.5, 3.0]


def test_leader_flag_requires_strict_max():
    rec = _rec(death_events=[[40, 2], [90, 1]], winner_seat=0, game_length=90)
    # armies: seat 0 strictly leads everywhere; land is tied 2-2-2 → no land leader
    own = np.zeros((91, 6), dtype=np.int8)
    own[:, 0:2] = 0
    own[:, 2:4] = 1
    own[:, 4:6] = 2
    arm = np.ones_like(own, dtype=np.int16)
    arm[:, 0] = 5
    rec.ownership, rec.armies = own, arm
    groups = loader.derive_groups([SPEC_A, SPEC_B, SPEC_B])
    _, stage_rows = extract.extract_game(rec, groups, early_grid=range(0))
    first = [r for r in stage_rows if r["stage"] == 3]
    leaders = {r["seat"]: r["army_leader_mid"] for r in first}
    assert leaders == {0: 1, 1: 0, 2: 0}
    assert all(r["land_leader_mid"] == 0 for r in first)
    s0 = next(r for r in first if r["seat"] == 0)
    assert s0["army_share_mid"] == pytest.approx(6 / 10)  # tiles 0,1: armies 5+1 of 10
    assert s0["eventual_win"] == 1


def _write_synthetic_run(run_dir: Path) -> None:
    games_dir = run_dir / "games"
    games_dir.mkdir(parents=True)
    specs = [SPEC_A, SPEC_B]
    (run_dir / "config.json").write_text(json.dumps(
        {"policy_specs": specs, "maps_arg": "random:1", "max_turns": 100}
    ))
    lines = []
    for idx, (slot_map, winner) in enumerate([([0, 1], 0), ([1, 0], 1)], start=1):
        T = 50
        own = np.zeros((T + 1, 6), dtype=np.int8)
        own[:, 0:3] = winner
        own[:, 3:6] = 1 - winner
        own[31:, :] = winner  # loser eliminated at tick 30
        arm = np.ones_like(own, dtype=np.int16)
        empty_i32 = np.array([], dtype=np.int32)
        np.savez(
            games_dir / f"game_{idx:03d}.npz",
            map_width=3, map_height=2,
            mountains=empty_i32,
            initial_cities=empty_i32,
            initial_city_armies=empty_i32,
            initial_neutrals=empty_i32,
            initial_neutral_armies=empty_i32,
            initial_generals=np.array([0, 3], dtype=np.int32),
            ownership=own, armies=arm,
            cities=empty_i32,
            cities_present_at=empty_i32,
            death_events=np.array([[30, 1 - winner]], dtype=np.int32),
            capture_events=np.array([[30, winner, 1 - winner]], dtype=np.int32),
            neutralize_events=np.zeros((0, 2), dtype=np.int32),
            actions_source=empty_i32,
            actions_dest=empty_i32,
            actions_is50=np.array([], dtype=np.int8),
        )
        metrics = {
            "move_analysis": [
                {"total_moves": 10, "total_passes": 5, "ones_traversed": 2,
                 "attack_move_count": 1, "attack_chain_count": 1}
            ] * 2,
            "policy_diagnostics": [],
        }
        (games_dir / f"game_{idx:03d}.metrics.json").write_text(json.dumps(metrics))
        lines.append(json.dumps({
            "game_index": idx,
            "replay_id": "map_x",
            "slot_map": slot_map,
            "winner": winner,
            "game_length": T,
            "player_stats": [{"land": 3, "army": 3, "n_moved": 10, "n_passed": 5}] * 2,
        }))
    (run_dir / "results.jsonl").write_text("\n".join(lines) + "\n")


def test_end_to_end_smoke(tmp_path):
    run_dir = tmp_path / "run"
    _write_synthetic_run(run_dir)
    script = Path(__file__).parents[1] / "scripts" / "analyze_eval.py"
    subprocess.run(
        [sys.executable, str(script), str(run_dir), "--no-plots"],
        check=True,
    )
    out = run_dir / "analysis"
    agg = json.loads((out / "metrics.json").read_text())
    # In both games, the seat playing SPEC_A's policy (slot 0) won.
    hl = agg["headline"]["groups"]
    assert hl["0607-ep4"]["wins"] == 2
    assert hl["0610-ep6"]["wins"] == 0
    assert agg["paired_by_map"]["sweeps"]["0607-ep4"] == 1
    assert (out / "report.md").exists()
    assert (out / "distributions.md").exists()
    assert (out / "player_games.csv").exists()


def test_groups_derivation_labels():
    groups = loader.derive_groups([SPEC_A, SPEC_A, SPEC_B])
    assert [g.label for g in groups] == ["0607-ep4", "0610-ep6"]
    assert groups[0].policy_indices == [0, 1]
