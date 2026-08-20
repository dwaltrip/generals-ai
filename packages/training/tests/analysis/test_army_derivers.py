"""Smoke coverage for the `captured` deriver (§6 of the surrender design).

Two halves: a unit check of `captured_per_tick` (column order + latching, on a
synthetic sim) and a corpus check that the table-attached column upholds the two
relationships the `capture_status` encoding rests on — `captured ⟹ army==0` and
`captured ⟹ ~alive`. The army-zero implication is the load-bearing one: its
contrapositive (army>0 ⟹ not-captured) is exactly the blind spot that makes a
surrendered-present slot invisible to this channel.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from training.analysis.families.army_derivers import (
    ARMY_SIM,
    CAPTURED,
    captured_per_tick,
    kth_lowest_alive,
)
from training.analysis.fq.frame_table import (
    GROUND_TRUTH_OBS_CFG,
    FrameTableSpec,
    build_frame_table,
)
from training.bc.config.targets_config import TargetsConfig
from training.bc.emit_spec import PartialEmitSpec


def test_kth_lowest_alive_order_and_guard() -> None:
    # Two frames, 8 channels. Dead channels (alive=False) must never be picked,
    # even when they hold the smallest value.
    values = np.array(
        [
            [50, 10, 30, 20, 40, 99, 99, 99],   # alive 0..4; ranks: ch1<ch3<ch2<ch4<ch0
            [5, 7, 99, 99, 99, 99, 99, 99],     # only ch0, ch1 alive
        ],
        dtype=np.float64,
    )
    alive = np.array(
        [
            [True, True, True, True, True, False, False, False],
            [True, True, False, False, False, False, False, False],
        ],
        dtype=bool,
    )
    # k=0 lowest, k=1 2nd, k=2 3rd.
    np.testing.assert_array_equal(kth_lowest_alive(values, alive, 0), [1, 0])
    np.testing.assert_array_equal(kth_lowest_alive(values, alive, 1), [3, 1])
    # Frame 1 has only 2 alive -> 3rd-lowest is undefined -> sentinel -1.
    np.testing.assert_array_equal(kth_lowest_alive(values, alive, 2), [2, -1])

    # A dead channel holding the global min is never picked.
    v2 = np.array([[0.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]])
    a2 = np.array([[False, True, True, True, True, True, True, True]])
    assert kth_lowest_alive(v2, a2, 0)[0] == 1


def test_captured_per_tick_order_and_latch() -> None:
    # ownership only supplies T (= shape[0]); content is irrelevant to the walk.
    sim = {
        "ownership": np.zeros((6, 4), dtype=np.int64),
        "capture_events": np.array([[2, 3, 1], [4, 3, 5]], dtype=np.int32),
    }
    flags = captured_per_tick(sim)
    assert flags.shape == (6, 8) and flags.dtype == bool

    # victim is column 2 of (tick, captor, victim); the flag latches at tick+1
    # (the board transfers tiles one tick after the capture event — the obs/alive
    # tick seam, 6.18-6).
    assert not flags[:3, 1].any() and flags[3:, 1].all()   # capture event t=2 -> latch t=3
    assert not flags[:5, 5].any() and flags[5:, 5].all()   # capture event t=4 -> latch t=5
    # captor (slot 3) and untouched slots are never flagged.
    assert not flags[:, 3].any()
    assert not flags[:, [0, 2, 4, 6, 7]].any()
    # monotonic per slot: once True, stays True (prev True ⟹ next True).
    assert np.all(~flags[:-1] | flags[1:])


def test_captured_column_implications(samples: list[tuple[Path, int]]) -> None:
    if not samples:
        pytest.skip("no eligible samples in fixture corpus")

    spec = FrameTableSpec(
        name="captured_smoke",
        emit=PartialEmitSpec(
            targets=TargetsConfig(elim_variant=None, elim_bin_edges=None),
            emit_alive_mask=True,
            attach_sim_frame=True,
        ),
        emit_cols={"alive_mask": "alive"},
        derivers=[ARMY_SIM, CAPTURED],
        derived_cols={},
    )
    t = build_frame_table(spec, samples, GROUND_TRUTH_OBS_CFG, max_games=3)

    n = t.frame_t.size
    captured = t.cols["captured"]
    army = t.cols["army_sim"]
    alive = t.cols["alive"].astype(bool)
    assert captured.shape == (n, 8) and captured.dtype == bool
    assert army.shape == (n, 8)

    # FFA games reliably eliminate players, so the implications below aren't vacuous.
    assert captured.sum() > 0, "no captured slots in fixture — implications untested"

    # The two load-bearing relationships. `captured ⟹ army==0` is what lets a
    # `none` (army-only) model exclude captured players via the zero-notch, and its
    # contrapositive is the surrender blind spot (army>0 ⟹ not-captured).
    assert np.all(army[captured] == 0), "a captured slot still holds army"
    assert not np.any(alive[captured]), "a captured slot still reads alive"
