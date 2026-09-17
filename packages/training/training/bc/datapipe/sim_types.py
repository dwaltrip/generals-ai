"""Domain objects for one recorded game.

Listed outermost first:

- `CorpusGame`: a parsed corpus game: the sim output plus its curated perspectives.
- `SimGame`: the replay parser's sim output (`<id>.npz`), one field per array.
- `PerspectiveMeta`: one curated player perspective of the game (from `<id>.meta.npz`).
- `SimFrame`: one frame of the game, seen through a perspective's slot order.

Producers (the walk core, the emission tail, the per-game precompute) take a
`SimGame` and a `PerspectiveMeta`. Only `CorpusGame` knows the curated list.

The kernels under `bc.obs`, `bc.mask`, and `bc.targets` are shared with live
inference, which builds its own sim dict as the game progresses (a subset of
the keys, with the per-tick fields as growing lists). So the kernels take a
`Mapping[str, ...]`, which both `SimGame` and the live dict satisfy.
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from training.bc.slots import SlotOrder
from training.bc.utils import meta_path_for


@dataclass(frozen=True, eq=False)
class SimGame(Mapping[str, np.ndarray]):
    # The parser's output schema, in the order `write_sim_output` writes it
    # (replay-parser/replay_parser/output.py). Values are kept as loaded.
    replay_id: np.ndarray
    version: np.ndarray
    map_width: np.ndarray
    map_height: np.ndarray
    mountains: np.ndarray
    initial_cities: np.ndarray
    initial_city_armies: np.ndarray
    initial_neutrals: np.ndarray
    initial_neutral_armies: np.ndarray
    initial_generals: np.ndarray
    ownership: np.ndarray             # [T, H*W] int8
    armies: np.ndarray                # [T, H*W] int16
    cities: np.ndarray
    cities_present_at: np.ndarray
    death_events: np.ndarray          # [n, 2] (t, slot)
    capture_events: np.ndarray        # [n, 3] (t, captor, captured)
    neutralize_events: np.ndarray     # [n, 2] (t, slot)
    actions_source: np.ndarray
    actions_dest: np.ndarray
    actions_is50: np.ndarray

    @classmethod
    def from_npz(cls, path: Path) -> SimGame:
        names = _sim_field_names()
        with np.load(path) as z:
            assert set(z.files) == set(names), (
                f"{path}: keys {sorted(set(z.files) ^ set(names))} differ from SimGame's fields"
            )
            return cls(**{name: z[name] for name in names})

    @property
    def T(self) -> int:
        return self.ownership.shape[0]

    @property
    def H(self) -> int:
        return int(self.map_height)

    @property
    def W(self) -> int:
        return int(self.map_width)

    @cached_property
    def deaths(self) -> np.ndarray:
        # Sorted death ticks for every eliminated player.
        if self.death_events.size == 0:
            return np.zeros(0, dtype=np.int64)
        return np.sort(self.death_events[:, 0])

    @cached_property
    def p_start(self) -> int:
        # The meta npz doesn't report player count, so we derive it from board
        # presence on game tick 0 (all players start with a single tile).
        # TODO: Fix this, we shouldn't have to do this. meta.npz should have player count.
        return int((np.unique(self.ownership[0]) >= 0).sum())

    def count_players_alive_at(self, t: int) -> int:
        # A player eliminated at frame `t` is counted as dead in that same frame.
        # This is why we pin to the "right" side.
        return self.p_start - int(np.searchsorted(self.deaths, t, side="right"))

    # TODO(sim-typing): temporary Mapping shim. The kernels still read
    # `sim["key"]`, and live inference passes its own dict. Remove when the
    # kernels switch to attribute access (planned with the inference refactor).
    def __getitem__(self, key: str) -> np.ndarray:
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(_sim_field_names())

    def __len__(self) -> int:
        return len(_sim_field_names())


def _sim_field_names() -> tuple[str, ...]:
    return tuple(f.name for f in fields(SimGame))


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
class CorpusGame:
    sim: SimGame
    # Every perspective recorded in the meta npz, in recorded order.
    _perspectives: tuple[PerspectiveMeta, ...]

    @classmethod
    def load(cls, sim_path: Path) -> CorpusGame:
        sim = SimGame.from_npz(sim_path)
        with np.load(meta_path_for(sim_path)) as z:
            meta = {key: z[key] for key in z.files}
        num_recorded = len(meta["perspective_player_ids"])
        perspectives = tuple(PerspectiveMeta.from_meta(meta, k, sim.T) for k in range(num_recorded))
        return cls(sim=sim, _perspectives=perspectives)

    def perspective(self, k: int) -> PerspectiveMeta:
        # `k` is a position in the parser's recorded list (`perspective_k` in
        # the manifest), which the meta npz arrays follow.
        return self._perspectives[k]

    def perspective_for_slot(self, slot: int) -> PerspectiveMeta:
        matches = [p for p in self._perspectives if p.slot == slot]
        assert len(matches) == 1, (
            f"slot {slot}: expected one recorded perspective, found {len(matches)}"
        )
        return matches[0]


@dataclass(frozen=True)
class SimFrame:
    """Raw sim-domain context for a single frame.

    Holds the per-game sim mapping, which is shared by reference across frames
    and should be treated as read-only. Also holds this frame's indexing.
    """
    # `Any` because two sim shapes are valid here: `SimGame`, and the live
    # path's hand-built dict (see the module docstring).
    sim: Mapping[str, Any]
    # `t` is not entirely perspective-specific. But in general, it doesn't make
    # sense to construct a `SimFrame` for ticks where the player is eliminated.
    t: int
    # This is perspective-specific.
    slot_order: SlotOrder
