"""Unit tests for `bc.player_status`: the per-game precompute and the
alive/present masks, with the four player fates (winner, captured-while-alive,
surrender→neutralize, surrender→capture) plus phantom slots, and the timing
edges the obs/target seam fix turns on (`>= t`, the surrender window, the
capture transient with no spurious surrendered frame).
"""

from __future__ import annotations

import numpy as np

from training.bc.player_status import (
    alive_mask,
    precompute_player_status,
    present_mask,
)


def _sim(T: int = 40):
    """A 4-real-player game (slots 0-3; 4-7 phantom) exercising every fate:

      - slot 0: winner (no death / removal).
      - slot 1: captured-while-alive at t=10 (death + capture same tick).
      - slot 2: surrenders at t=5, neutralized at t=15 (removal via neutralize).
      - slot 3: surrenders at t=8, captured at t=12 (removal via capture).
    """
    own = np.full((T, 1, 8), -1, dtype=np.int64)
    own[0, 0, :4] = [0, 1, 2, 3]
    return {
        "ownership": own,
        "death_events": np.array(
            [(10, 1), (5, 2), (8, 3)], dtype=np.int32
        ),  # (tick, player)
        "capture_events": np.array(
            [(10, 0, 1), (12, 0, 3)], dtype=np.int32
        ),  # (tick, captor, captured)
        "neutralize_events": np.array([(15, 2)], dtype=np.int32),  # (tick, player)
    }


def test_precompute_marks_deaths_removals_and_phantoms() -> None:
    ctx = precompute_player_status(_sim(T=40))
    s = ctx.sentinel
    assert s == 41  # T + 1
    assert list(ctx.is_real) == [True] * 4 + [False] * 4
    # death = first DeathEvent (surrender or capture-while-alive); winner=sentinel.
    assert list(ctx.death_by_slot) == [s, 10, 5, 8, -1, -1, -1, -1]
    # removal = board-removal tick; slot 2 via neutralize, slot 3 via capture.
    assert list(ctx.removal_by_slot) == [s, 10, 15, 12, -1, -1, -1, -1]


_RAW = [0, 1, 2, 3, 4, 5, 6, 7]


def test_winner_always_alive_and_present_phantoms_never() -> None:
    ctx = precompute_player_status(_sim(T=40))
    for t in (0, 20, 39):
        assert alive_mask(ctx, _RAW, t)[0]  # winner alive throughout
        assert present_mask(ctx, _RAW, t)[0]  # ...and on the board
        assert not alive_mask(ctx, _RAW, t)[4:].any()  # phantoms masked
        assert not present_mask(ctx, _RAW, t)[4:].any()


def test_alive_uses_ge_t_at_the_death_frame() -> None:
    # slot 1 dies at t=10: alive THROUGH frame 10 (the obs's pre-event board
    # still shows it), dead from 11.
    ctx = precompute_player_status(_sim(T=40))
    assert alive_mask(ctx, _RAW, 10)[1]
    assert not alive_mask(ctx, _RAW, 11)[1]


def test_surrender_window_is_present_and_not_alive() -> None:
    # slot 2: surrenders t=5 (alive through 5), neutralized t=15 (present
    # through 15). Window [6, 15] reads present & ~alive.
    ctx = precompute_player_status(_sim(T=40))
    for t in (6, 10, 15):
        assert present_mask(ctx, _RAW, t)[2]
        assert not alive_mask(ctx, _RAW, t)[2]
    # before surrender: alive & present; after removal: neither.
    assert alive_mask(ctx, _RAW, 5)[2] and present_mask(ctx, _RAW, 5)[2]
    assert not alive_mask(ctx, _RAW, 16)[2] and not present_mask(ctx, _RAW, 16)[2]


def test_capture_while_alive_has_no_spurious_surrender_frame() -> None:
    # slot 1: death and removal both at t=10, so alive & present flip together
    # — alive→eliminated with no intervening present-&-~alive (surrender) frame.
    ctx = precompute_player_status(_sim(T=40))
    for t in range(40):
        surrendered = present_mask(ctx, _RAW, t)[1] and not alive_mask(ctx, _RAW, t)[1]
        assert not surrendered, f"slot 1 read surrendered at t={t}"
    # the transient: alive+present at 10, eliminated at 11.
    assert alive_mask(ctx, _RAW, 10)[1] and present_mask(ctx, _RAW, 10)[1]
    assert not alive_mask(ctx, _RAW, 11)[1] and not present_mask(ctx, _RAW, 11)[1]


def test_army_zero_but_alive_edge_reads_present_and_alive() -> None:
    # A real slot with no death/removal event (e.g. a general zeroed to 0 army
    # but not captured) has no event, so present_mask stays True regardless of
    # army — the reason `is_present` is event-derived, not army==0. The winner
    # (slot 0) is exactly this: no events, present & alive at every tick.
    ctx = precompute_player_status(_sim(T=40))
    assert ctx.removal_by_slot[0] == ctx.sentinel  # no removal event
    assert present_mask(ctx, _RAW, 25)[0] and alive_mask(ctx, _RAW, 25)[0]
