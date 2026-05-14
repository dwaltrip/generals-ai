import sim_core

from replay_parser.decode import ReplayData, decode_wire


def parse_replay(raw: bytes) -> tuple[sim_core.State, ReplayData]:
    replay = decode_wire(raw)
    state = sim_core.simulate(replay)
    return state, replay
