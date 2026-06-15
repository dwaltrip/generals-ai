"""Obs-introspection and board-geometry helpers for analysis leaf code.

These read the unpadded board or specific channels out of a padded obs tensor.
They target analysis / probe leaf code, not hot training paths: `crop_to_board`
forces a host sync (it reads the board extent back to Python to slice), which is
fine off the training loop and the reason these don't live in `bc/`.
"""

from __future__ import annotations

from torch import Tensor

from training.bc.constants import get_obs_channel_indices


N_PLAYERS = 8  # the per-player axis: alive_mask, the army-total planes, the elim head


def crop_to_board(x: Tensor, board_mask: Tensor) -> Tensor:
    """View of `x` with canvas padding removed — the canonical unpadded-board
    accessor for leaf code, and the one place the "real board is a rectangle at
    the top-left of the padded canvas" assumption lives. The extent is derived
    from `board_mask`, so callers never hardcode board dims.

    `x`: `[..., H, W]`, `board_mask`: `[..., H, W]` bool → `[..., H_b, W_b]`, a
    view (basic slicing, zero copy).
    """
    h = int(board_mask.any(dim=-1).sum())   # rows with any real cell
    w = int(board_mask.any(dim=-2).sum())   # cols with any real cell
    return x[..., :h, :w]


def army_totals_ch_indices(dense_history_n: int) -> list[int]:
    """Obs channel indices of the per-player total-army planes, canonical player
    order (0 = self, 1..7 = opponents in `opp_slots` order) — aligned with the
    elim head's player axis and the frame's `alive_mask`. These planes are
    broadcast scalars (`army / 1000`), fog-independent, in the fixed base
    channels, so the indices are independent of `dense_history_n`.
    """
    # TODO: this is a bit brittle
    wanted = ["self_army_count"] + [f"opp_{i}_army_count" for i in range(1, N_PLAYERS)]
    return get_obs_channel_indices(dense_history_n, wanted)


def per_player_army(obs: Tensor, board_mask: Tensor, army_ch: list[int]) -> Tensor:
    """The per-player total-army scalars (canonical player order) read from the
    obs army-total planes. Each plane is constant over the real board, so we read
    the cropped board's corner — one real cell, no averaging.

    `obs`: `[C, H, W]`, `board_mask`: `[1, H, W]` bool, `army_ch`: the army-total
    channel indices (see `army_totals_ch_indices`) → `[len(army_ch)]`.
    """
    return crop_to_board(obs, board_mask)[army_ch, 0, 0]
