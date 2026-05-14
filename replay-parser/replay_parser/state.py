from collections import deque
from dataclasses import dataclass, field

import numpy as np

from replay_parser.decode import ReplayData
from replay_parser.errors import ArmyOverflowError
from replay_parser.types import (
    CaptureEvent,
    DeathEvent,
    MoveRowIndex,
    NeutralizeEvent,
    PerspectiveIndex,
    PlayerIndex,
    TileIndex,
    Timestep,
)

# `state.armies` is int32 during simulation, narrowed to int16 at snapshot time
# (the per-tick output dtype per design doc §3 / §44). Stacks beyond int16
# don't occur in competitive FFA; the snapshot narrow is the overflow gate —
# any tile > _ARMY_MAX raises ArmyOverflowError and the caller skips the game.
_ARMY_MAX = 32767


# All-AFK fallback can fire at most 2000 timesteps after the last move; pad a little
# more so action-stream arrays never need to grow.
_ACTION_BUFFER_HEADROOM = 2100


@dataclass(slots=True)
class SnapshotBuffer:
    ownership: list[np.ndarray] = field(default_factory=list)
    armies: list[np.ndarray] = field(default_factory=list)   # int16 (narrowed from int32 working state)
    cities_mask: list[np.ndarray] = field(default_factory=list)

    def append(self, ownership: np.ndarray, armies: np.ndarray, cities_mask: np.ndarray, timestep: int = -1) -> None:
        if armies.max(initial=0) > _ARMY_MAX:
            raise ArmyOverflowError(f"army > {_ARMY_MAX} at t={timestep}")
        self.ownership.append(ownership.copy())
        self.armies.append(armies.astype(np.int16))   # copies and narrows in one pass
        self.cities_mask.append(cities_mask.copy())

    def __len__(self) -> int:
        return len(self.ownership)


@dataclass(slots=True)
class State:
    # Grid (mutated in place)
    ownership: np.ndarray     # int8[H*W];  -2=mountain, -1=neutral, p>=0 owned
    armies: np.ndarray        # int32[H*W] (snapshots narrow to int16 — see SnapshotBuffer.append)
    cities_mask: np.ndarray   # bool[H*W];  in lockstep with `cities`

    # Structures (lists)
    cities: list[TileIndex]
    generals: list[TileIndex]

    # Per-player flags
    alive: list[bool]
    has_kill: list[bool]
    input_buffer: list[deque[MoveRowIndex]]

    # Game-level scalars
    timestep: Timestep
    num_players: int
    alive_count: int
    updates_since_move: int
    afks_cursor: int
    moves_cursor: int

    # Event lists (game-level output)
    death_events: list[DeathEvent]
    capture_events: list[CaptureEvent]
    neutralize_events: list[NeutralizeEvent]

    # Four variants of the "army of player j destroyed by player i" matrix,
    # used to investigate which definition matches the server's surrender-
    # bonus kill-credit rule. All shape int32[num_players, num_players].
    #
    #   sym vs off : do we credit both sides of a combat (defender also
    #                destroyed some of attacker's army) or only the attacker?
    #   all vs pre : do we count damage to player j throughout the game, or
    #                only while alive[j] is still True (i.e., before they
    #                surrendered / were captured)?
    damage_sym_all: np.ndarray
    damage_sym_pre: np.ndarray
    damage_off_all: np.ndarray
    damage_off_pre: np.ndarray

    # Per-perspective output (curated players only)
    perspective_indices: dict[PlayerIndex, PerspectiveIndex]
    actions_source: np.ndarray   # int16[K, T_max];  -1 = pass sentinel
    actions_dest: np.ndarray     # int16[K, T_max];  -1 = pass sentinel
    actions_is50: np.ndarray     # uint8[K, T_max]

    # Per-tick snapshot buffer
    snapshots: SnapshotBuffer


def build_initial_state(
    replay: ReplayData,
    perspective_player_ids: tuple[PlayerIndex, ...] = (),
) -> State:
    static = replay.static
    n_cells = static.map_width * static.map_height
    num_players = len(static.usernames)

    ownership = np.full(n_cells, -1, dtype=np.int8)
    armies = np.zeros(n_cells, dtype=np.int32)
    cities_mask = np.zeros(n_cells, dtype=bool)

    if static.mountains:
        ownership[static.mountains] = -2

    cities: list[TileIndex] = []
    for idx, army in zip(static.initial_cities, static.initial_city_armies, strict=True):
        cities.append(int(idx))
        cities_mask[idx] = True
        armies[idx] = army

    generals: list[TileIndex] = []
    alive = [True] * num_players
    alive_count = num_players
    for p, gen in enumerate(static.initial_generals):
        if gen >= 0:
            ownership[gen] = p
            armies[gen] = 1
            generals.append(int(gen))
        else:
            generals.append(-1)
            alive[p] = False
            alive_count -= 1

    for idx, army in zip(static.initial_neutrals, static.initial_neutral_armies, strict=True):
        armies[idx] = army

    t_last_move = int(replay.moves.timestep.max()) if len(replay.moves.timestep) else 0
    action_len = t_last_move + _ACTION_BUFFER_HEADROOM
    k = len(perspective_player_ids)
    perspective_indices = {p: i for i, p in enumerate(perspective_player_ids)}

    return State(
        ownership=ownership,
        armies=armies,
        cities_mask=cities_mask,
        cities=cities,
        generals=generals,
        alive=alive,
        has_kill=[False] * num_players,
        input_buffer=[deque() for _ in range(num_players)],
        timestep=0,
        num_players=num_players,
        alive_count=alive_count,
        updates_since_move=0,
        afks_cursor=0,
        moves_cursor=0,
        death_events=[],
        capture_events=[],
        neutralize_events=[],
        damage_sym_all=np.zeros((num_players, num_players), dtype=np.int32),
        damage_sym_pre=np.zeros((num_players, num_players), dtype=np.int32),
        damage_off_all=np.zeros((num_players, num_players), dtype=np.int32),
        damage_off_pre=np.zeros((num_players, num_players), dtype=np.int32),
        perspective_indices=perspective_indices,
        actions_source=np.full((k, action_len), -1, dtype=np.int16),
        actions_dest=np.full((k, action_len), -1, dtype=np.int16),
        actions_is50=np.zeros((k, action_len), dtype=np.uint8),
        snapshots=SnapshotBuffer(),
    )
