import sim_core

from replay_parser.decode import ReplayData, decode_wire
from replay_parser.state import State, build_initial_state
from replay_parser.step import step
from replay_parser.types import PlayerIndex


def parse_replay(
    raw: bytes,
    perspective_player_ids: tuple[PlayerIndex, ...] = (),
) -> tuple[State, ReplayData]:
    sim_core.ping()
    replay = decode_wire(raw)
    state = build_initial_state(replay, perspective_player_ids)
    state.snapshots.append(state.ownership, state.armies, state.cities_mask, state.timestep)
    while step(state, replay):
        pass
    return state, replay
