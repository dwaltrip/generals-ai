"""Army derivers: per-player total army from the raw sim (ground truth) and from
the encoded obs (what the model sees). Both read the same quantity, so analyses
built on either agree.
"""

from __future__ import annotations

import numpy as np

from training.analysis.fq.derivers import Deriver, per_game
from training.analysis.obs_utils import army_totals_ch_indices, per_player_army
from training.bc.obs_config import OBS_CONFIG_DEFAULTS


# Army-total obs channels — `dense_history_n`-independent, so resolving them once
# from the config defaults is safe for ground-truth tables. A model-joined table
# would instead pin obs_cfg to the producing checkpoint.
ARMY_CH = army_totals_ch_indices(OBS_CONFIG_DEFAULTS.dense_history_n)


def army_totals_per_tick(sim: dict[str, np.ndarray]) -> np.ndarray:
    """`[T, 8]` total army per slot per tick (sum of armies over owned cells).

    The canonical army-from-sim ground truth — mirrors the leaderboard army column:
    neutral/mountain cells excluded by the `ownership == slot` mask, owned cities
    contribute their garrison. The one source for sim army totals, so analyses agree.
    """
    own, arm = sim["ownership"], sim["armies"]  # [T, H, W]
    T = own.shape[0]
    tot = np.zeros((T, 8), dtype=np.int64)
    for s in range(8):
        tot[:, s] = np.where(own == s, arm, 0).reshape(T, -1).sum(1)
    return tot


# army from the encoded obs — what the MODEL sees (carries fp16 storage noise
# when obs_dtype is fp16). Scaled back to raw-army units to match army_sim.
ARMY_OBS = Deriver(
    "army_obs",
    per_player=True,
    fn=lambda f: per_player_army(f.obs, f.valid_mask, ARMY_CH).numpy().astype(float) * 1000.0,
)

# army from raw sim — GROUND TRUTH. Whole-game totals, indexed per frame.
ARMY_SIM = per_game(
    "army_sim",
    per_player=True,
    prepare=army_totals_per_tick,                              # sim -> [T, 8]
    index=lambda totals, f: totals[f.t, f.raw_order].astype(float),
)
