import sim_core

from replay_parser.decode import ReplayData, decode_wire
from replay_parser.types import PlayerIndex


def parse_replay(
    raw: bytes,
    perspective_player_ids: tuple[PlayerIndex, ...] = (),
) -> tuple[sim_core.State, ReplayData]:
    replay = decode_wire(raw)
    state = sim_core.simulate(replay, list(perspective_player_ids))
    return state, replay
