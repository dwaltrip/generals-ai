"""Parity test for `sim_core.State.step_tick` against `sim_core.simulate`.

`simulate` runs the full game in one Rust call by walking a global moves
array via persistent cursors. `step_tick` is the stepwise companion for
self-play / eval / RL rollouts: caller hands it only the moves + AFKs
that apply at the current tick, the sim packs them and advances one tick.

This test replays each fixture through a step_tick loop and byte-compares
the resulting State against `simulate(replay)`. Comparison covers the
finalized board (ownership / armies / cities_mask), per-player liveness,
timestep, the full snapshot history, and the three event lists. If
step_tick produces a different result than simulate on the same input,
the sim_core delta is broken.
"""

from collections import defaultdict

import numpy as np
import pytest

from replay_parser.parser import parse_replay
import sim_core

from _fixture_lib import FIXTURES, FIXTURES_DIR


def _group_moves_by_tick(
    timestep: np.ndarray,
    index: np.ndarray,
    source: np.ndarray,
    dest: np.ndarray,
    is50: np.ndarray,
) -> dict[int, list[tuple[int, int, int, int]]]:
    by_t: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
    for i in range(len(timestep)):
        by_t[int(timestep[i])].append(
            (int(index[i]), int(source[i]), int(dest[i]), int(is50[i]))
        )
    return by_t


def _group_afks_by_tick(
    timestep: np.ndarray, index: np.ndarray
) -> dict[int, list[int]]:
    by_t: dict[int, list[int]] = defaultdict(list)
    for i in range(len(timestep)):
        by_t[int(timestep[i])].append(int(index[i]))
    return by_t


@pytest.mark.parametrize("spec", FIXTURES, ids=[f.name for f in FIXTURES])
def test_step_tick_matches_simulate(spec):
    wire_path = FIXTURES_DIR / spec.name / "wire.bin"
    if not wire_path.exists():
        pytest.skip(f"missing wire.bin: {wire_path}")

    _state, replay = parse_replay(wire_path.read_bytes())
    canonical = sim_core.simulate(replay)

    moves_by_tick = _group_moves_by_tick(
        replay.moves.timestep,
        replay.moves.index,
        replay.moves.source,
        replay.moves.dest,
        replay.moves.is50,
    )
    afks_by_tick = _group_afks_by_tick(replay.afks.timestep, replay.afks.index)

    live = sim_core.new_state(replay.static.map)
    # Hard cap as a runaway-loop guard; canonical.timestep is the real ceiling
    # but we don't want to trust it for the termination condition.
    max_iters = canonical.timestep + 10
    iters = 0
    while live.alive_count > 1:
        t = live.timestep
        ran = live.step_tick(
            moves_by_tick.get(t, []),
            afks_by_tick.get(t, []),
        )
        assert ran, f"step_tick returned False mid-game at t={t}"
        assert live.timestep == t + 1, (
            f"step_tick did not advance: t was {t}, now {live.timestep}"
        )
        iters += 1
        if iters > max_iters:
            pytest.fail(f"step_tick loop exceeded {max_iters} iterations")

    # Finalized board
    np.testing.assert_array_equal(live.ownership, canonical.ownership)
    np.testing.assert_array_equal(live.armies, canonical.armies)
    np.testing.assert_array_equal(live.cities_mask, canonical.cities_mask)

    # Per-player liveness + game clock
    assert list(live.alive) == list(canonical.alive)
    assert live.alive_count == canonical.alive_count
    assert live.timestep == canonical.timestep

    # Snapshot history — same length, byte-identical at every tick
    assert live.snapshots_len == canonical.snapshots_len
    for t in range(canonical.snapshots_len):
        np.testing.assert_array_equal(
            live.snapshots_ownership[t],
            canonical.snapshots_ownership[t],
            err_msg=f"snapshots_ownership diverged at t={t}",
        )
        np.testing.assert_array_equal(
            live.snapshots_armies[t],
            canonical.snapshots_armies[t],
            err_msg=f"snapshots_armies diverged at t={t}",
        )
        np.testing.assert_array_equal(
            live.snapshots_cities_mask[t],
            canonical.snapshots_cities_mask[t],
            err_msg=f"snapshots_cities_mask diverged at t={t}",
        )

    # Event lists — order matters (sim emits in canonical order)
    assert len(live.death_events) == len(canonical.death_events)
    for a, b in zip(live.death_events, canonical.death_events, strict=True):
        assert (a.timestep, a.player) == (b.timestep, b.player)

    assert len(live.capture_events) == len(canonical.capture_events)
    for a, b in zip(live.capture_events, canonical.capture_events, strict=True):
        assert (a.timestep, a.captor, a.captured) == (b.timestep, b.captor, b.captured)

    assert len(live.neutralize_events) == len(canonical.neutralize_events)
    for a, b in zip(live.neutralize_events, canonical.neutralize_events, strict=True):
        assert (a.timestep, a.player) == (b.timestep, b.player)
