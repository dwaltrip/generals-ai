"""Elimination-head targets: the per-game precompute and per-frame target builders.

`precompute_elim` resolves the winner-vs-phantom split once per game; its result
(`ElimCtx`) is the natural per-game cache seam for downstream analysis — the fq
toolkit keys on it rather than recomputing per frame.

This module is the intended shared home for the elim target logic that the probe
/ fq consolidation converges on: today both the dataset's `encode_frame` and
`who_dies_next_baselines` build their targets here off the one precompute.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# `alive_mask` is single-sourced in `bc.player_status` (shared with the obs
# encoder); re-exported here so `elim_targets` callers keep their import path.
# `ElimCtx` carries the `is_real`/`death_by_slot` fields it reads, so it satisfies
# the mask's structural type without an import cycle.
from training.bc.player_status import alive_mask, precompute_player_status

__all__ = [
    "ElimCtx",
    "alive_mask",
    "next_death_target",
    "precompute_elim",
    "time_bin_targets",
]


@dataclass(frozen=True)
class ElimCtx:
    """Per-game precompute backing the elimination heads' targets.

    `edges` is the dataset-constant bin-edge array (time_bin head only);
    `death_by_slot` and `is_real` are per-game [8]-arrays indexed by *raw* slot
    id; `sentinel` is the winner's stand-in death tick. Built once per game in
    `_walk` (when an elim head is enabled) and threaded into every `encode_frame`
    call for that game. Both elim variants read off this one precompute.
    """

    edges: np.ndarray | None   # [n_bins - 1] strictly-increasing bin edges; None for next_death
    death_by_slot: np.ndarray  # [8] int64 — elim timestep; winner sentinel; -1 = phantom
    is_real: np.ndarray        # [8] bool — slot actually played this game
    sentinel: int              # winner's stand-in death tick (> any real death)


def precompute_elim(
    sim: dict[str, np.ndarray], edges: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, int]:
    """Per-game `(death_by_slot, is_real, sentinel)` for the elim targets.

    `death_by_slot[s]` is slot `s`'s elimination timestep, the large finite
    winner `sentinel` (integer-typed, not `np.inf`) for the one real slot absent
    from `death_events`, or `-1` for a phantom slot that never played.
    `is_real[s]` marks the slots present at t=0. The winner-vs-phantom split is
    what `real_slots` resolves: the winner is real-and-absent-from-deaths, a
    phantom is not-real — and only the latter must be masked out.

    The sentinel only has to exceed every real death tick (real deaths are ≤
    T−1) so the winner never wins the soonest-death argmin unless it is the lone
    survivor. For the time_bin head it is additionally sized as `T + max_edge`
    so the winner's `Δ = sentinel − t` lands in the top "never" bin; next_death
    passes no edges and uses `T + 1`.

    The obs encoder always runs at P=8 and `opp_slots` always yields 7 ids from
    `range(8)`, but FFA games can start with <8 players (~7% of the corpus, see
    `6.13-6`). Without `is_real`, those phantom channels would inherit the winner
    sentinel and train as "alive, never dies" on every frame.

    The `(death_by_slot, is_real)` computation is single-sourced in
    `bc.player_status.precompute_player_status` (shared with the obs encoder);
    this wrapper sizes the elim-specific `sentinel` and returns the tuple the
    `ElimCtx(edges, *precompute_elim(...))` construction expects.
    """
    T = sim["ownership"].shape[0]
    sentinel = T + (int(edges[-1]) if edges is not None else 1)
    status = precompute_player_status(sim, sentinel=sentinel)
    return status.death_by_slot, status.is_real, status.sentinel


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
    raw = np.asarray(raw_order, dtype=np.intp)
    death_ch = elim.death_by_slot[raw]
    alive = alive_mask(elim, raw_order, t)
    delta = death_ch - t
    bins = np.where(alive, np.digitize(delta, elim.edges, right=False), 0)
    return bins.astype(np.int64), alive


def next_death_target(
    elim: ElimCtx, raw_order: list[int], t: int
) -> tuple[int, np.ndarray, int]:
    """Per-frame `(next_victim_channel, alive_mask[8], dt)` for who-dies-next.

    `raw_order` is the canonical channel→raw-slot map (`[perspective_slot,
    *opp_slots]`). A channel is alive iff its slot is real AND not yet eliminated
    (`death >= t`) — the same convention as the time_bin head; this is the domain
    of the cross-player softmax. The next victim is the alive channel with the
    soonest death tick; the winner's `sentinel` death is larger than any real
    death, so it only wins this argmin when it is the lone survivor.

    Returns the next-victim channel index, the alive mask, and `dt = death − t`
    (ticks until that death — the horizon, dumped for offline confidence-ramp /
    horizon-stratified reads). The death frame itself reads `dt = 0` (the dying
    player is alive-and-the-victim at frame `death`). Frames with no real future death — the winner's
    tail, where only the winner remains alive — get `target = -1` (a CE
    `ignore_index` sentinel, masked from the loss) and `dt = -1`. Ties (two
    deaths at the same tick) resolve to the lowest canonical channel index via
    `argmin`'s first-min rule — a fixed, deterministic convention.
    """
    raw = np.asarray(raw_order, dtype=np.intp)
    death_ch = elim.death_by_slot[raw]                       # [8]
    alive = alive_mask(elim, raw_order, t)                   # [8]
    # Soonest death among alive channels; non-alive channels can't be the victim.
    cand = np.where(alive, death_ch, np.iinfo(np.int64).max)
    nxt = int(cand.argmin())
    if cand[nxt] >= elim.sentinel:
        # The soonest "death" is the winner sentinel (or no alive channel) → no
        # real next elimination from here. Mask the frame out.
        return -1, alive, -1
    return nxt, alive, int(cand[nxt] - t)
