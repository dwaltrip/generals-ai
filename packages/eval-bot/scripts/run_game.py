#!/usr/bin/env -S uv run python
"""Run an eval-bot game and render an HTML replay.

Usage (from repo root):

    ./packages/eval-bot/scripts/run_game.py
    ./packages/eval-bot/scripts/run_game.py --seed 42
    ./packages/eval-bot/scripts/run_game.py --max-ticks 800 --replay-id abc123
    ./packages/eval-bot/scripts/run_game.py --out /tmp/my-game.html
"""

from __future__ import annotations

import argparse
import datetime as dt
import random
from pathlib import Path

from self_play.seed_map import list_two_player_replay_ids, load_static_from_db
from self_play.viewer import build_html_from_state

from eval_bot.driver import play_game


_DEFAULT_TMP = Path(__file__).resolve().parents[1] / "tmp"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replay-id", type=str, default=None,
                        help="Seed-map replay ID. Mutually exclusive with --seed.")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed to randomly pick a 2p replay from the corpus.")
    parser.add_argument("--max-ticks", type=int, default=400,
                        help="Game length cap in timesteps.")
    parser.add_argument("--progress-interval", type=int, default=50,
                        help="Print status every N ticks (0 to disable).")
    parser.add_argument("--out", type=str, default=None,
                        help="HTML output path. Default: tmp/evalbot-<timestamp>.html.")
    args = parser.parse_args()

    if args.replay_id:
        replay_id = args.replay_id
    elif args.seed is not None:
        rng = random.Random(args.seed)
        candidates = list_two_player_replay_ids(limit=100)
        replay_id = rng.choice(candidates)
        print(f"seed={args.seed} picked replay {replay_id}")
    else:
        replay_id = list_two_player_replay_ids(limit=1)[0]
    static = load_static_from_db(replay_id)
    print(f"seed: replay_id={replay_id} map={static.map_width}x{static.map_height}")

    state, bots = play_game(
        static,
        max_ticks=args.max_ticks,
        progress_interval=args.progress_interval,
    )

    print()
    print(f"ticks={state.timestep} alive={list(state.alive)}")
    for i, b in enumerate(bots):
        own = state.ownership
        arm = state.armies
        mask = own == i
        land = int(mask.sum())
        army = int((arm * mask).sum())
        print(f"  p{i}: land={land} army={army} "
              f"expand={b.n_expand} explore={b.n_explore} no_move={b.n_no_move}")

    ts = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    label = f"evalbot-{ts}"
    out_path = Path(args.out) if args.out else _DEFAULT_TMP / f"{label}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_html_from_state(label, state, static))
    print(f"\nwrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
