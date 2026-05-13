import numpy as np

from replay_parser.decode import Moves
from replay_parser.state import State, set_army
from replay_parser.types import CaptureEvent, DeathEvent, MoveRowIndex, PlayerIndex


def attack(state: State, move_idx: MoveRowIndex, moves: Moves) -> None:
    source = int(moves.source[move_idx])
    dest = int(moves.dest[move_idx])
    mover = int(moves.index[move_idx])
    is50 = bool(moves.is50[move_idx])

    src_army = int(state.armies[source])
    move_reserve = (src_army + 1) // 2 if is50 else 1
    incoming = src_army - move_reserve

    set_army(state, source, src_army - incoming)  # subtracted regardless of outcome

    dest_owner = int(state.ownership[dest])
    dest_army = int(state.armies[dest])

    if dest_owner == mover:
        # Only growth path in combat: same-owner accumulation. The other
        # branches produce values <= incoming, which already fits in int16.
        set_army(state, dest, dest_army + incoming)
    elif dest_army >= incoming:
        # Defender holds (includes the equal-armies tie — defender's advantage).
        set_army(state, dest, dest_army - incoming)
    else:
        set_army(state, dest, incoming - dest_army)
        state.ownership[dest] = mover


def execute_attack(state: State, move_idx: MoveRowIndex, moves: Moves) -> None:
    dest = int(moves.dest[move_idx])
    old_owner = int(state.ownership[dest])

    attack(state, move_idx, moves)

    new_owner = int(state.ownership[dest])
    if old_owner != new_owner and old_owner >= 0 and state.generals[old_owner] == dest:
        execute_player_capture(state, captured=old_owner, captor=new_owner)


def execute_player_capture(state: State, captured: PlayerIndex, captor: PlayerIndex) -> None:
    general_tile = state.generals[captured]

    # Combat already flipped the general tile to captor, so the mask naturally
    # excludes it — only the captured player's *other* tiles get halved.
    mask = state.ownership == captured
    state.ownership[mask] = captor
    set_army(state, mask, (state.armies[mask].astype(np.int32) + 1) // 2)  # halved, rounded upward

    state.has_kill[captor] = True

    if state.alive[captured]:
        kill_player(state, captured)

    state.cities.append(general_tile)
    state.cities_mask[general_tile] = True
    state.generals[captured] = -1

    state.capture_events.append(
        CaptureEvent(timestep=state.timestep, captor=captor, captured=captured)
    )


def try_neutralize_player(state: State, p: PlayerIndex) -> None:
    general_tile = state.generals[p]

    mask = state.ownership == p
    state.ownership[mask] = -1  # no halving — armies preserved

    state.cities.append(general_tile)
    state.cities_mask[general_tile] = True
    # Without nulling generals[p], the per-turn general-tick keeps incrementing
    # a now-neutral tile every 2 timesteps.
    state.generals[p] = -1


def kill_player(state: State, p: PlayerIndex) -> None:
    state.alive[p] = False
    state.alive_count -= 1
    state.death_events.append(DeathEvent(timestep=state.timestep, player=p))
    state.input_buffer[p].clear()
    # generals[p] stays — tile remains owned, continues producing during the
    # surrender countdown (5.11-2 §3.4).


def kill_all_but_strongest(state: State) -> None:
    living = [p for p in range(state.num_players) if state.alive[p]]
    if len(living) <= 1:
        return

    owned_mask = state.ownership >= 0
    owners_owned = state.ownership[owned_mask]
    armies_per_p = np.bincount(
        owners_owned,
        weights=state.armies[owned_mask].astype(np.int64),
        minlength=state.num_players,
    )
    tiles_per_p = np.bincount(owners_owned, minlength=state.num_players)

    living.sort(key=lambda p: (int(armies_per_p[p]), int(tiles_per_p[p]), p))
    for p in living[:-1]:
        kill_player(state, p)
