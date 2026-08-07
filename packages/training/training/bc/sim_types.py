"""Domain objects over the raw sim/meta dicts.

They represent the main nested layers or views of a game. Listed outermost first:

- Game level: GameMeta
- Player perspective level (for the given game): PerspectiveMeta
- Frame level (one frame in the game, for the given perspective): SimFrame

The construction inputs differ:

`GameMeta` and `PerspectiveMeta` need the full replay-parser npz pair (sim + meta),
so only code reading parser output can build them. The live path (`inference.py`)
does not load or construct a meta dict.

`SimFrame` only needs a sim dict, and accepts either of its two shapes:
    - replay-parser output npz files: These are completed games. The sim dict has
      the "full" set of keys, and all values are `ndarray`.
    - The live path's hand-built dict: These games are still in progress. This sim
      dict has a subset of the keys, and the time-indexed history fields are Python
      lists of `ndarray` snapshots, appended per tick, as they occur. This is in 
      contrast to their npz counterparts, which are single pre-stacked `ndarray`s.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from training.bc.slots import SlotOrder


@dataclass(frozen=True)
class PerspectiveMeta:
    k: int              # perspective_k
    slot_order: SlotOrder
    placement: int      # 1..P (1-indexed)
    elim_t: int         # -1 if survived
    # Exclusive bound of this perspective's walkable frames. Frames stop at the
    # player's elimination, and always exclude the terminal snapshot at T - 1,
    # which has no paired action to use as a target.
    end_t: int

    @property
    def slot(self) -> int:
        return self.slot_order.perspective

    # TODO: add a small synthetic test (tests/test_sim_types.py) pinning
    # `end_t`'s three cases (survivor, early elim, clamp) and the
    # `count_players_alive_at` death-tick boundary.
    @classmethod
    def from_meta(cls, meta: dict[str, np.ndarray], k: int, T: int) -> PerspectiveMeta:
        slot = int(meta["perspective_player_ids"][k])
        elim_t = int(meta["elim_timestep"][k])
        end_t = T - 1 if elim_t == -1 else min(T - 1, elim_t)
        return cls(
            k=k,
            slot_order=SlotOrder.for_perspective(slot),
            placement=int(meta["placement"][k]),
            elim_t=elim_t,
            end_t=end_t,
        )


@dataclass(frozen=True)
class GameMeta:
    # tick count — length of the sim arrays' time axis
    T: int
    H: int
    W: int
    # number of starting players
    p_start: int
    # sorted death timesteps
    deaths: np.ndarray
    # every perspective recorded in the meta npz, keyed by `perspective_k`
    perspectives: dict[int, PerspectiveMeta]

    @classmethod
    def from_npz(cls, sim: dict[str, np.ndarray], meta: dict[str, np.ndarray]) -> GameMeta:
        T = sim["ownership"].shape[0]
        H = int(sim["map_height"])
        W = int(sim["map_width"])

        # Death ticks for every eliminated player (rows are (t, slot)).
        # We need the full sim data, as the meta only stores `elim_timestep`
        # for curated perspectives.
        deaths = (
            np.sort(sim["death_events"][:, 0])
            if sim["death_events"].size
            else np.zeros(0, dtype=np.int64)
        )
        # The meta npz doesn't report player count, so we derive it from board
        # presence on game tick 0 (all players start with a single tile).
        # TODO: Fix this, we shouldn't have to do this. meta.npz should have player count.
        p_start = int((np.unique(sim["ownership"][0]) >= 0).sum())

        num_recorded = len(meta["perspective_player_ids"])
        perspectives_by_k = {
            k: PerspectiveMeta.from_meta(meta, k, T) for k in range(num_recorded)
        }
        return GameMeta(
            T=T,
            H=H,
            W=W,
            p_start=p_start,
            deaths=deaths,
            perspectives=perspectives_by_k,
        )

    def count_players_alive_at(self, t: int) -> int:
        # A player eliminated at frame `t` is counted as dead in that same frame.
        # This is why we pin to the "right" side.
        return self.p_start - int(np.searchsorted(self.deaths, t, side="right"))


@dataclass(frozen=True)
class SimFrame:
    """Raw sim-domain context for a single frame.

    Holds the per-game `sim` dict, which is shared by reference across frames
    and should be treated as read-only. Also holds this frame's indexing.
    """
    # The value type is `Any` because two different sim-dict shapes are valid here.
    # See "construction inputs" above, in the module docstring.
    sim: Mapping[str, Any]
    # `t` is not entirely perspective-specific. But in general, it doesn't make
    # sense to construct a `SimFrame` for ticks where the player is eliminated.
    t: int
    # This is perspective-specific.
    slot_order: SlotOrder
