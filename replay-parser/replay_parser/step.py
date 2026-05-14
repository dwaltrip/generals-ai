from replay_parser.combat import (
    execute_attack,
    kill_all_but_strongest,
    kill_player,
    try_neutralize_player,
)
from replay_parser.decode import ReplayData
from replay_parser.moves import (
    dependency_loop,
    is_valid,
    priority_sort,
    record_action,
)
from replay_parser.state import State
from replay_parser.types import MoveRowIndex


MAX_ALL_AFK_TIMESTEPS = 2000      # bundle line 67932
MAX_GAME_TIMESTEPS = 50000        # bundle line 67930


def step(state: State, replay: ReplayData) -> bool:
    """Advance one timestep. Returns False when the game has ended."""
    if state.alive_count <= 1:
        return False

    process_pending_afks(state, replay)
    buffer_pending_moves(state, replay)
    candidates = select_candidates(state, replay)

    for i in candidates:                                        # intent recorded pre-resolution
        record_action(state, i, replay.moves)

    any_ran = resolve_and_execute(state, replay, candidates)

    state.updates_since_move = 0 if any_ran else state.updates_since_move + 1
    if state.updates_since_move > MAX_ALL_AFK_TIMESTEPS or state.timestep > MAX_GAME_TIMESTEPS:
        kill_all_but_strongest(state)

    state.timestep += 1
    apply_production(state)
    state.snapshots.append(state.ownership, state.armies, state.cities_mask, state.timestep)
    return True


def process_pending_afks(state: State, replay: ReplayData) -> None:
    afks = replay.afks
    while state.afks_cursor < len(afks.timestep) and afks.timestep[state.afks_cursor] <= state.timestep:
        p = int(afks.index[state.afks_cursor])
        if state.alive[p]:
            kill_player(state, p)
        else:
            try_neutralize_player(state, p)
        state.afks_cursor += 1
        if state.alive_count <= 1:
            break


def buffer_pending_moves(state: State, replay: ReplayData) -> None:
    moves = replay.moves
    while state.moves_cursor < len(moves.timestep) and moves.timestep[state.moves_cursor] <= state.timestep:
        p = int(moves.index[state.moves_cursor])
        state.input_buffer[p].append(state.moves_cursor)
        state.moves_cursor += 1


def select_candidates(state: State, replay: ReplayData) -> list[MoveRowIndex]:
    candidates: list[MoveRowIndex] = []
    for p in range(state.num_players):
        while state.input_buffer[p]:
            i = state.input_buffer[p].popleft()
            if is_valid(state, i, replay.moves):
                candidates.append(i)
                break
    return candidates


def resolve_and_execute(state: State, replay: ReplayData, candidates: list[MoveRowIndex]) -> bool:
    ordered = dependency_loop(
        priority_sort(candidates, state, replay.moves),
        state, replay.moves,
    )
    any_ran = False
    for i in ordered:
        if is_valid(state, i, replay.moves):           # defensive re-check (type-B invalidation)
            execute_attack(state, i, replay.moves)
            any_ran = True
    return any_ran


def apply_production(state: State) -> None:
    if state.timestep % 2 == 0:
        for gen in state.generals:
            if gen >= 0:
                state.armies[gen] += 1
        for c in state.cities:
            if state.ownership[c] >= 0:
                state.armies[c] += 1
    if state.timestep % 50 == 0:
        state.armies[state.ownership >= 0] += 1
