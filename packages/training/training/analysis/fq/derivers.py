"""Canonical single-source derivers for fq.

A `Deriver` owns the one way to compute a named per-frame quantity, so every
analysis reads the same value. It maps one `Frame` to that frame's value of its
column (`[8]` when `per_player`, scalar otherwise); `build_frame_table` stacks the
per-frame results into the `[N, 8]` / `[N]` column. Two scopes:
  - per-frame (row-local): read straight off the `Frame`.
  - per-game: `per_game(prepare, index)` runs `prepare(sim)` once per game and
    indexes this frame's slice out (whole-game quantities, e.g. army totals).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from torch import Tensor


@dataclass
class Frame:
    """The deriver-facing view of one (game, perspective, tick): the dataset's
    encoded outputs (`obs`, `valid_mask`, `alive`) unified with the raw `sim` it
    attached via the `unsafe_attach_sim_frame` seam (`sim`, `t`, `raw_order`).

    `raw_order` is the canonical `[perspective_slot, *opp_slots]` channel→raw-slot
    map. `game_id` is a walk-local per-game token (the `per_game` cache key):
    monotonic over the contiguous walk, collision-free unlike `id(sim)`.
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


def windowed_per_game(
    name: str,
    per_player: bool,
    series: Callable[[dict], np.ndarray],
    reduce: Callable[[np.ndarray], np.ndarray],
    K: int,
) -> Deriver:
    """A per-game deriver that reduces each tick's trailing-K window to one value.

    `series(sim) -> [T, 8]` is the per-tick signal (e.g. army deltas). For each tick
    `t` we form the window of the `K` ticks ending at `t` and apply `reduce` to it,
    yielding one `[T, 8]` column of reduced values (computed once per game, then
    indexed per frame like `per_game`).

    The trailing-window tensor handed to `reduce` is `[T, K, 8]`: row `t` holds the
    `K` rows `series[t-K+1 .. t]`. Early ticks have fewer than `K` real values, so
    the window is **NaN-padded at the front** — `W[t, :K-1-t]` is NaN for `t < K-1`.

    CONTRACT: `reduce` must be NaN-aware (`np.nanmin`, `np.nanpercentile`,
    `np.nanmean`, ...), so a short start-of-game window reduces over only its real
    values rather than over the padding. A non-NaN-aware reduce would propagate the
    pad and corrupt every early tick.

    Like `per_game`, this is a deriver constructor — the heavy work runs once per
    game in `prepare`, not as a groupby/agg over a built `FrameTable`.
    """

    def prepare(sim: dict) -> np.ndarray:
        S = series(sim)                                  # [T, 8]
        T, P = S.shape
        # W[t] = the K rows ending at t, NaN-padded where t - K + 1 < 0.
        W = np.full((T, K, P), np.nan, dtype=np.float64)
        for j in range(K):                               # j = offset back from t
            lag = K - 1 - j                              # how many ticks before t
            W[lag:, j, :] = S[: T - lag if lag else T]
        # An all-NaN window is in-contract (a fully-padded early row when the series
        # itself starts with NaN); the nan-reduce returns NaN for it, so silence the
        # benign "All-NaN slice" / "empty slice" warnings the reduce raises there.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"(All-NaN|Mean of empty) slice", RuntimeWarning)
            return reduce(W)                             # [T, 8]

    def index(state: np.ndarray, f: Frame) -> np.ndarray:
        return state[f.t, f.raw_order]

    return per_game(name, per_player, prepare, index)
