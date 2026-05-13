from replay_parser.decode import Moves
from replay_parser.state import State
from replay_parser.types import MoveRowIndex


def is_valid(state: State, move_idx: MoveRowIndex, moves: Moves) -> bool:
    """Predicate used both at buffer-pop (type-A) and at execution time (type-B)."""
    source = int(moves.source[move_idx])
    dest = int(moves.dest[move_idx])
    mover = int(moves.index[move_idx])
    return (
        state.ownership[source] == mover
        and state.ownership[dest] != -2
        and state.armies[source] >= 2
    )


def record_action(state: State, move_idx: MoveRowIndex, moves: Moves) -> None:
    p = int(moves.index[move_idx])
    ps = state.perspective_indices.get(p)
    if ps is None:
        return
    t = state.timestep
    state.actions_source[ps, t] = moves.source[move_idx]
    state.actions_dest[ps, t] = moves.dest[move_idx]
    state.actions_is50[ps, t] = moves.is50[move_idx]


def priority_sort(
    candidates: list[MoveRowIndex],
    state: State,
    moves: Moves,
) -> list[MoveRowIndex]:
    """V3 sort key (5.11-2 §2.4):
        (defensive-first, general-attacks-last, larger-source-army-first, player-index-tiebreak)
    """
    def key(i: MoveRowIndex) -> tuple[int, int, int, int]:
        p = int(moves.index[i])
        dest = int(moves.dest[i])
        source = int(moves.source[i])
        return (
            0 if state.ownership[dest] == p else 1,
            0 if not _is_general_attack(state, dest, p) else 1,
            -int(state.armies[source]),
            p,
        )

    return sorted(candidates, key=key)


def dependency_loop(
    sorted_candidates: list[MoveRowIndex],
    state: State,
    moves: Moves,
) -> list[MoveRowIndex]:
    """Inward-first ordering: a move whose source is another candidate's dest is
    deferred until that other move resolves. Cycle fallback at line 67415–67418:
    when no non-deferred move exists, take the highest-priority remaining.
    """
    remaining = list(sorted_candidates)
    result: list[MoveRowIndex] = []
    while remaining:
        for j, m in enumerate(remaining):
            src = moves.source[m]
            blocked = any(moves.dest[other] == src for other in remaining if other is not m)
            if not blocked:
                result.append(remaining.pop(j))
                break
        else:
            result.append(remaining.pop(0))
    return result


def _is_general_attack(state: State, dest: int, mover: int) -> bool:
    for p, g in enumerate(state.generals):
        if g == dest:
            return p != mover
    return False
