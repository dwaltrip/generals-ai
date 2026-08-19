"""Elimination-head targets: the per-game precompute and per-frame target builders.

`make_elim_ctx` resolves the per-game precompute once; its result (`ElimCtx`) is
the natural per-game cache seam for downstream analysis — the fq toolkit keys on
it rather than recomputing per frame.

This module is the intended shared home for the elim target logic that the probe
/ fq consolidation converges on: today both `bc.encode_frame` and
`who_dies_next_baselines` build their targets here off the one precompute.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from training.bc.player_status import (
    PlayerStatusCtx,
    make_alive_mask,
    precompute_player_status,
    present_mask,
)


__all__ = [
    "ElimCtx",
    "make_elim_ctx",
    "next_death_target",
    "time_bin_targets",
]


@dataclass(frozen=True)
class ElimCtx:
    """Per-game precompute backing the elimination heads' targets: the shared
    player-status precompute, with an elim-sized sentinel, plus the time_bin
    head's bin edges.

    The two status events back different elim targets: `death_by_slot` (first
    DeathEvent — surrender or capture) drives the time_bin head's `alive` domain;
    `removal_by_slot` (board-removal — capture/neutralize that clears the slot's
    tiles) drives the next_death head's `present` domain. See `bc.player_status`
    for the death-vs-removal distinction (the surrender window is the gap).
    """

    player_status: PlayerStatusCtx
    edges: np.ndarray | None  # [n_bins - 1] strictly-increasing bin edges; None for next_death


def make_elim_ctx(
    sim: dict[str, np.ndarray],
    edges: tuple[int, ...] | None
) -> ElimCtx:
    edges_arr = np.asarray(edges, dtype=np.int64) if edges is not None else None
    # Sentinel sizing: any value past the last real event tick keeps the winner
    # out of the soonest-event argmin, but the time_bin head also digitizes the
    # winner's Δ = sentinel − t, so it needs T + max_edge to land the winner in
    # the top "never" bin. next_death passes no edges and gets T + 1.
    T = sim["ownership"].shape[0]
    sentinel = T + (int(edges_arr[-1]) if edges_arr is not None else 1)
    return ElimCtx(
        player_status=precompute_player_status(sim, sentinel=sentinel),
        edges=edges_arr,
    )


def time_bin_targets(
    elim: ElimCtx, raw_order: list[int], t: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel `(elim_bin_target[8], alive_mask[8])` at frame `t`.

    `raw_order` is the canonical channel→raw-slot map (`[perspective_slot,
    *opp_slots]`). A channel is alive iff its slot is real AND not yet
    eliminated (`death >= t` — a player eliminated at `t` reads alive at frame
    `t`, matching the obs's pre-event board snapshot; see `bc.player_status`).
    Live `Δ = death − t` digitizes into a bin; the winner's large sentinel Δ
    lands in the top bin (merged with "never"). Dead and phantom channels get
    bin 0, masked out by `alive`.
    """
    assert elim.edges is not None, "time_bin targets require bin edges"
    status = elim.player_status
    raw = np.asarray(raw_order, dtype=np.intp)
    death_ch = status.death_by_slot[raw]
    alive = make_alive_mask(status, raw_order, t)
    delta = death_ch - t
    bins = np.where(alive, np.digitize(delta, elim.edges, right=False), 0)
    return bins.astype(np.int64), alive


def next_death_target(
    elim: ElimCtx, raw_order: list[int], t: int
) -> tuple[int, np.ndarray, int, np.ndarray]:
    """Per-frame `(next_victim_channel, present_mask[8], dt, removal_dt[8])` for who-dies-next.

    The event is **board-removal** (capture/neutralize that clears a player's
    tiles), not "stops playing" (the DeathEvent the time_bin head uses). Removal
    is the more value-relevant "the player is gone" signal — a surrendered player
    still on the board hasn't changed the survivors' position yet; the tile
    transfer at removal is what does.

    `raw_order` is the canonical channel→raw-slot map (`[perspective_slot,
    *opp_slots]`). A channel is *present* iff its slot is real AND not yet removed
    (`removal >= t`); this is the domain of the cross-player softmax. Event and
    domain move together: a surrendered-but-present player can be the next removal
    and must stay in the domain to be nameable. The next victim is the present
    channel with the soonest removal tick; the winner's `sentinel` removal is
    larger than any real removal, so it only wins this argmin when it is the lone
    survivor.

    Returns the next-victim channel index, the present mask, `dt = removal − t`
    for the next victim (the horizon, dumped for offline confidence-ramp /
    horizon-stratified reads), and the per-channel `removal_dt = removal − t` over
    all 8 channels (the soft target's input — masked to `present` by the loss; the
    winner's huge `dt` underflows to ~0 target mass). The removal frame itself
    reads `dt = 0` (the player is present-and-the-victim at frame `removal`).
    Frames with no real future removal — the winner's tail, where only the winner
    remains present — get `target = -1` (a CE `ignore_index` sentinel, masked from
    the loss) and `dt = -1`. Ties (two removals at the same tick) resolve to the
    lowest canonical channel index via `argmin`'s first-min rule — a fixed,
    deterministic convention; the soft target instead splits mass evenly across a
    tie.
    """
    status = elim.player_status
    raw = np.asarray(raw_order, dtype=np.intp)
    removal_ch = status.removal_by_slot[raw]                 # [8]
    present = present_mask(status, raw_order, t)             # [8]
    removal_dt = (removal_ch - t).astype(np.int64)           # [8] per-channel horizon
    # Soonest removal among present channels; non-present channels can't be the victim.
    cand = np.where(present, removal_ch, np.iinfo(np.int64).max)
    nxt = int(cand.argmin())
    if cand[nxt] >= status.sentinel:
        # The soonest "removal" is the winner sentinel (or no present channel) →
        # no real next removal from here. Mask the frame out.
        return -1, present, -1, removal_dt
    return nxt, present, int(cand[nxt] - t), removal_dt
