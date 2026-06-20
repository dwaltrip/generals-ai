"""Smoke coverage for the fq toolkit: build a small elim_head_debug table off the
real corpus and assert the load-bearing contracts — the army-from-obs vs
army-from-sim parity (the single-source claim) and the rule-as-column.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from training.analysis.families.elim_head_debug import ELIM_HEAD_DEBUG
from training.analysis.fq.derivers import Frame, windowed_per_game
from training.analysis.fq.frame_table import GROUND_TRUTH_OBS_CFG, build_frame_table, select


def _frame(sim: dict, t: int) -> Frame:
    """A minimal Frame for exercising a per-game deriver's fn directly: only
    `sim`, `t`, `raw_order`, `game_id` are read (identity raw_order keeps channels
    == slots). The obs/valid/alive fields are unused by these derivers."""
    return Frame(
        obs=None,  # type: ignore[arg-type]
        valid_mask=None,  # type: ignore[arg-type]
        alive=np.ones(8, dtype=bool),
        sim=sim,
        t=t,
        raw_order=list(range(8)),
        game_id=0,
    )


def test_windowed_per_game_nan_padding_and_reduction() -> None:
    # A toy [T, 8] signal: slot s = s + 10 * tick, so trailing windows are easy to
    # reason about. `series` ignores `sim`'s contents and just returns this array.
    T, K = 5, 3
    S = (np.arange(8)[None, :] + 10 * np.arange(T)[:, None]).astype(np.float64)
    sim = {"ownership": np.zeros((T, 1), dtype=np.int64)}  # only T (shape[0]) matters

    der = windowed_per_game(
        "wmin", per_player=True, series=lambda _sim: S, reduce=lambda W: np.nanmin(W, axis=1), K=K
    )

    # t=0: window is only tick 0 (two NaN pads dropped by nanmin) -> the t=0 row.
    np.testing.assert_array_equal(der.fn(_frame(sim, 0)), S[0])
    # t=1: window {tick0, tick1}, one pad -> min is tick0 (smaller).
    np.testing.assert_array_equal(der.fn(_frame(sim, 1)), S[0])
    # t=4: full K-window {tick2,3,4} -> min is tick2.
    np.testing.assert_array_equal(der.fn(_frame(sim, 4)), S[2])

    # A different NaN-aware reduce (mean) over a known full window: ticks 1,2,3.
    der_mean = windowed_per_game(
        "wmean", per_player=True, series=lambda _sim: S, reduce=lambda W: np.nanmean(W, axis=1), K=K
    )
    np.testing.assert_allclose(der_mean.fn(_frame(sim, 3)), S[1:4].mean(0))
    # Early tick t=1 reduces over only its real rows (ticks 0,1), not the pad.
    np.testing.assert_allclose(der_mean.fn(_frame(sim, 1)), S[0:2].mean(0))


def test_fq_build_parity_and_rule(samples: list[tuple[Path, int]]) -> None:
    if not samples:
        pytest.skip("no eligible samples in fixture corpus")

    # GROUND_TRUTH_OBS_CFG is fp32, so the parity assert isn't masked by fp16 noise.
    t = build_frame_table(ELIM_HEAD_DEBUG, samples, GROUND_TRUTH_OBS_CFG, max_games=3)

    assert t.n_games >= 1
    assert {"alive", "victim", "dt", "army_obs", "army_sim"} <= set(t.cols)
    n = t.frame_t.size
    assert t.cols["army_sim"].shape == (n, 8)        # per-player, axis 0 == N
    assert t.cols["victim"].shape == (n,)            # per-frame

    # Single-source: army-from-obs == army-from-sim (exact modulo float) on alive.
    alive = t.cols["alive"]
    parity = float(np.abs(t.cols["army_obs"] - t.cols["army_sim"])[alive].max())
    assert parity < 2.0, f"army_obs vs army_sim diverge by {parity}"

    # build_frame_table attaches the family's derived columns, so the rule-as-column
    # (computed on the table's exact frames) is present without the consumer re-running it.
    assert {"margin", "low_army_victim"} <= set(t.cols)
    assert t.cols["low_army_victim"].shape == (n,)
    assert np.all((t.cols["low_army_victim"] >= 0) & (t.cols["low_army_victim"] < 8))

    # select masks every column at once and keeps the axis-0 invariant.
    imm = select(t, (t.cols["dt"] >= 0) & (t.cols["dt"] < 10))
    assert all(v.shape[0] == imm.frame_t.size for v in imm.cols.values())


@pytest.mark.skip(reason="stub — join_dump not built yet; see TODO below")
def test_fq_join_dump_against_real_dump() -> None:
    """STUB — the real-dump validation for `join_dump` (design 6.15-3 Appendix A).

    TODO(join_dump): build this once `join_dump` + the `persp_val_index` remap
    land. The point of this test is the cross-check the design insists on: join an
    fq table against an ACTUAL model dump and assert the shared truth columns agree
    on the overlap. A self round-trip is explicitly insufficient — it uses fq's key
    on both sides and passes even when fq's `persp_val_index` disagrees with real
    dumps, which is the one thing that can be wrong.

    Shape when built:
      - Build ELIM_HEAD_DEBUG on the SAME manifest/split the dump's run used, so
        fq's full-split positions line up with the dump's `persp_val_index`.
        (`samples_for_split` reads `manifest["val"]` in order; the dataset numbers
        `sample_idx` over that order; the dump remaps `sample_idx -> persp_val_index`.
        Same manifest => aligned keys.)
      - `j = join_dump(t, dump_path, ELIM_HEAD_DEBUG.truth_map)`  # inner-join on
        (persp_val_index, frame_t).
      - Assert the truth_map columns agree on the overlap:
        victim == next_elim_target, dt == next_elim_dt, alive == next_elim_alive_mask.
      - Assert (near-)full coverage of the table's rows (a capped fq table is a
        strict subset of the dump's frames), and that the head's prediction columns
        (next_elim_probs / next_elim_ce) came through attached.

    OPEN — the fixture question (design Q4, DEFERRED for a fresh-morning decision):
    the natural target is the yolo-11k epoch-5 dump
    (`data/training/sweeps/2026-06-14-yolo-11k-elim-head-v2/runs/2026-06-14T07-47-22Z/
    analysis/stratified_val_epoch_005.npz`), whose run used `cloud_24k_sub_11k.json`
    — already the fq CLI default manifest. But sweeps/ is gitignored, so this dump
    is NOT present in CI. Decide: (a) skip-when-absent local validation, (b) commit a
    small fixture dump, or (c) keep this a manual local check and leave the suite to
    the round-trip-free pieces above. That choice determines whether this is a
    standing automated test or a documented manual one.
    """
    raise NotImplementedError
