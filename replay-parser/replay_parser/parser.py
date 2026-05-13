from replay_parser.decode import ReplayData, decode_wire
from replay_parser.state import State, build_initial_state
from replay_parser.step import step
from replay_parser.types import PlayerIndex


def parse_replay(
    raw: bytes,
    perspective_player_ids: tuple[PlayerIndex, ...] = (),
) -> tuple[State, ReplayData]:
    replay = decode_wire(raw)
    state = build_initial_state(replay, perspective_player_ids)
    state.snapshots.append(state.ownership, state.armies, state.cities_mask)
    while step(state, replay):
        pass
    return state, replay
