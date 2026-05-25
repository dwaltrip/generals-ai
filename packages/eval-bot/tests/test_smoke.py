"""Smoke test: run a short game with EvalBots and verify they expand."""

from self_play.seed_map import list_two_player_replay_ids, load_static_from_db

from eval_bot.driver import play_game


def test_bots_expand():
    """Two EvalBots play 200 ticks on a real map. Both should expand
    beyond their starting general (land > 1)."""
    replay_ids = list_two_player_replay_ids(limit=1)
    assert replay_ids, "need at least one 2-player replay in the corpus"
    static = load_static_from_db(replay_ids[0])

    state, bots = play_game(
        static, max_ticks=200, progress_interval=0,
    )

    own = state.ownership
    for p in range(state.num_players):
        if state.alive[p]:
            land = int((own == p).sum())
            assert land > 1, (
                f"player {p} should have expanded beyond the general "
                f"(land={land} after {state.timestep} ticks)"
            )

    for bot in bots:
        assert bot.n_moved > 0, "bot should have made at least one move"
