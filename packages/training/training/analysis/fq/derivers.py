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
