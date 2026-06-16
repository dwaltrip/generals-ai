"""Smoke coverage for the fq toolkit: build a small elim_head_debug table off the
real corpus and assert the load-bearing contracts — the army-from-obs vs
army-from-sim parity (the single-source claim) and the rule-as-column.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from training.analysis.fq.families import ELIM_HEAD_DEBUG, bottom_two_margin, lowest_army_victim
from training.analysis.fq.frame_table import build_frame_table, select
from training.bc.obs_config import OBS_CONFIG_DEFAULTS, ObsConfig


def test_fq_build_parity_and_rule(samples: list[tuple[Path, int]]) -> None:
    if not samples:
        pytest.skip("no eligible samples in fixture corpus")

    # fp32 obs so the parity assert isn't masked by fp16 storage quantization.
    obs_cfg = ObsConfig(dense_history_n=OBS_CONFIG_DEFAULTS.dense_history_n, obs_dtype="fp32")
    t = build_frame_table(ELIM_HEAD_DEBUG, samples, obs_cfg, max_games=3)

    assert t.n_games >= 1
    assert {"alive", "victim", "dt", "army_obs", "army_sim"} <= set(t.cols)
    n = t.frame_t.size
    assert t.cols["army_sim"].shape == (n, 8)        # per-player, axis 0 == N
    assert t.cols["victim"].shape == (n,)            # per-frame

    # Single-source: army-from-obs == army-from-sim (exact modulo float) on alive.
    alive = t.cols["alive"]
    parity = float(np.abs(t.cols["army_obs"] - t.cols["army_sim"])[alive].max())
    assert parity < 2.0, f"army_obs vs army_sim diverge by {parity}"

    # Rule-as-column computes on the table's exact frames and yields valid victims.
    t.cols["margin"] = bottom_two_margin(t)
    t.cols["low_army_victim"] = lowest_army_victim(t)
    assert t.cols["low_army_victim"].shape == (n,)
    assert np.all((t.cols["low_army_victim"] >= 0) & (t.cols["low_army_victim"] < 8))

    # select masks every column at once and keeps the axis-0 invariant.
    imm = select(t, (t.cols["dt"] >= 0) & (t.cols["dt"] < 10))
    assert all(v.shape[0] == imm.frame_t.size for v in imm.cols.values())
