"""Per-player game status: the per-game precompute + the alive/present masks.

A neutral, torch-free home shared by the obs encoder (`bc.obs`) and the elim
targets (`bc.targets.elim_targets`), so the "who is alive / on the board at tick
t" logic lives in exactly one place rather than being recomputed (and risking
divergence) on each side.

Two per-slot tick markers, both following the sim-core event convention (events
fire pre-increment and become visible at `snapshot[e+1]`, so `snapshot[e]` is
pre-event — see `sim-core/README.md`):

  - `death_by_slot` — the first `DeathEvent` (surrender, or capture-while-alive).
    Drives `alive`: a player stops *playing* here.
  - `removal_by_slot` — the board-removal event (capture or neutralize) that
    transfers/clears the player's tiles. Drives `present`: the player's army
    leaves the board here.

The surrender window is exactly the gap between the two (`~alive & present`).
Both masks use `tick >= t` (alive/present through `snapshot[e]`, gone from
`e+1`), which is the correct test against the obs's pre-event board snapshot.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class PlayerStatusCtx:
    """Per-game per-slot status, indexed by *raw* slot id `0..7`.

    `death_by_slot` / `removal_by_slot` hold the event tick, the winner
    `sentinel` (a finite integer > any real tick) for a real slot that never
    dies / is never removed, or `-1` for a phantom slot that never played.
    `is_real` marks the slots present at `t=0` (FFA games can start with <8
    players, ~7% of the corpus); phantom slots must be masked out.
    """

    is_real: np.ndarray          # [8] bool
    death_by_slot: np.ndarray    # [8] int64 — first DeathEvent tick; sentinel; -1 phantom
    removal_by_slot: np.ndarray  # [8] int64 — capture/neutralize tick; sentinel; -1 phantom
    sentinel: int


class _HasDeathFields(Protocol):
    """Structural type for `alive_mask` — anything carrying the death markers.

    Both `PlayerStatusCtx` and `targets.elim_targets.ElimCtx` satisfy it, so the
    one `alive_mask` implementation serves both without an import cycle. Declared
    as read-only properties so frozen-dataclass fields match.
    """

    @property
    def is_real(self) -> np.ndarray: ...
    @property
    def death_by_slot(self) -> np.ndarray: ...


def precompute_player_status(
    sim: Mapping[str, Any], sentinel: int | None = None
) -> PlayerStatusCtx:
    """Per-game `PlayerStatusCtx` from the sim's event lists.

    `sentinel` defaults to `T + 1` (just past the last tick — enough for the
    masks); callers that digitize a `Δ = sentinel − t` into bins (the elim
    time_bin head) pass a larger value so the winner lands in the top bin.
    """
    ownership = sim["ownership"]
    # `len` (not `.shape[0]`) so the live inference path can pass a growing list
    # of per-tick rows; `sentinel` only has to exceed the current tick.
    T = len(ownership)
    sentinel = int(T + 1) if sentinel is None else int(sentinel)

    real = np.unique(ownership[0])
    real = real[real >= 0].astype(np.intp)
    is_real = np.zeros(8, dtype=np.bool_)
    is_real[real] = True

    # death: surrender or capture-while-alive (the first DeathEvent per slot).
    death_by_slot = np.full(8, -1, dtype=np.int64)
    death_by_slot[real] = sentinel  # winner default; dead slots overwritten below
    deaths = sim["death_events"]  # [K, 2] = (tick, player)
    if deaths.size:
        death_by_slot[deaths[:, 1].astype(np.intp)] = deaths[:, 0]

    # removal: the board-removal event that clears the slot's tile ownership.
    # A slot is removed at most once, via exactly one of capture / neutralize.
    # This is the event that makes a surrendered-player no longer capturable.
    removal_by_slot = np.full(8, -1, dtype=np.int64)
    removal_by_slot[real] = sentinel  # winner default; removed slots overwritten below
    captures = sim["capture_events"]  # [K, 3] = (tick, captor, captured)
    if captures.size:
        removal_by_slot[captures[:, 2].astype(np.intp)] = captures[:, 0]
    neutralizes = sim["neutralize_events"]  # [K, 2] = (tick, player)
    if neutralizes.size:
        removal_by_slot[neutralizes[:, 1].astype(np.intp)] = neutralizes[:, 0]

    return PlayerStatusCtx(is_real, death_by_slot, removal_by_slot, int(sentinel))


def alive_mask(ctx: _HasDeathFields, raw_order: list[int], t: int) -> np.ndarray:
    """Per-channel alive mask `[8]`: real and not yet eliminated.

    `death_by_slot >= t` — a player whose first DeathEvent fires at `t` reads
    *alive* at frame `t` (the obs's pre-event board still shows them) and dead
    from `t+1`. This is the cross-player softmax / regression domain.
    """
    raw = np.asarray(raw_order, dtype=np.intp)
    return ctx.is_real[raw] & (ctx.death_by_slot[raw] >= t)


def present_mask(ctx: PlayerStatusCtx, raw_order: list[int], t: int) -> np.ndarray:
    """Per-channel present mask `[8]`: real and still on the board.

    `removal_by_slot >= t` — the player's tiles are still on the board through
    frame `t`, gone from `t+1`. A surrendered-but-not-yet-removed player is
    `present & ~alive`; the army==0-but-alive transient (general zeroed, not
    captured) is `present & alive` (no removal event fired).
    """
    raw = np.asarray(raw_order, dtype=np.intp)
    return ctx.is_real[raw] & (ctx.removal_by_slot[raw] >= t)
