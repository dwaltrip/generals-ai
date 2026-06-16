"""Canonical single-source derivers for fq.

A `Deriver` owns the *one* way to compute a named per-frame quantity, so every
analysis reads the same value (the fix for the army-divergence class of bug,
6.14-1). A deriver maps one `Frame` (the unified per-frame view) to that frame's
value of its column — `[8]` when `per_player`, scalar otherwise; `build_frame_table`
stacks the per-frame results into the `[N, 8]` / `[N]` column.

Two scopes:
  - per-frame (row-local): read straight off the `Frame` (e.g. `army_obs`).
  - per-game: `per_game(prepare, index)` runs `prepare(sim)` once per game and
    indexes this frame's slice out — for whole-game quantities (e.g. `army_sim`,
    and later army-delta over a window).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from torch import Tensor

from training.analysis.obs_utils import army_totals_ch_indices, per_player_army
from training.bc.obs_config import OBS_CONFIG_DEFAULTS


# Army-total obs channels. These are `dense_history_n`-independent, so resolving
# them once from the config defaults is safe for ground-truth tables (Q5: a
# model-joined table would instead pin obs_cfg to the producing checkpoint).
ARMY_CH = army_totals_ch_indices(OBS_CONFIG_DEFAULTS.dense_history_n)


@dataclass
class Frame:
    """The deriver-facing view of one (game, perspective, tick): the dataset's
    encoded outputs (`obs`, `valid_mask`, `alive`) unified with the raw sim it
    attached via the `unsafe_attach_sim_frame` seam (`sim`, `t`, `raw_order`).

    `raw_order` is the canonical `[perspective_slot, *opp_slots]` channel→raw-slot
    map. `game_id` is the walk-local per-game token used as the `per_game` cache
    key (monotonic over the contiguous walk — collision-free by construction,
    unlike `id(sim)`).
    """

    obs: Tensor
    valid_mask: Tensor
    alive: np.ndarray          # [8] bool
    sim: dict[str, np.ndarray]  # raw sim, shared by reference across the game
    t: int
    raw_order: list[int]       # canonical [perspective, *opp] slots, len 8
    game_id: int


@dataclass
class Deriver:
    name: str
    per_player: bool                       # [N, 8] vs [N]
    fn: Callable[[Frame], np.ndarray]


def per_game(
    name: str,
    per_player: bool,
    prepare: Callable[[dict], np.ndarray],
    index: Callable[[np.ndarray, Frame], np.ndarray],
) -> Deriver:
    """A per-game deriver: `prepare(sim)` runs once per game, cached on the
    frame's `game_id`; `index(state, frame)` reads this frame's value out. Valid
    only on the single contiguous, tick-ordered walk `build_frame_table` runs.
    """
    cache: dict = {"gid": None, "state": None}

    def fn(f: Frame) -> np.ndarray:
        if f.game_id != cache["gid"]:
            cache["state"] = prepare(f.sim)
            cache["gid"] = f.game_id
        return index(cache["state"], f)

    return Deriver(name, per_player, fn)


def army_totals_per_tick(sim: dict[str, np.ndarray]) -> np.ndarray:
    """`[T, 8]` total army per slot per tick (sum of armies over owned cells).

    The canonical army-from-sim ground truth: mirrors the leaderboard's army
    column — neutral / mountain cells are excluded by the `ownership == slot`
    mask, owned cities contribute their garrison. Single source: both `army_sim`
    here and `who_dies_next_baselines` read it, so the two never diverge.
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
