#!/usr/bin/env -S uv run python
"""POC eval script: run a model checkpoint against EvalBot N times.

Usage (from repo root):

    ./scripts/eval_poc.py --checkpoint path/to/epoch_005.pt
    ./scripts/eval_poc.py --checkpoint path/to/epoch_005.pt --games 20
    ./scripts/eval_poc.py --checkpoint path/to/epoch_005.pt --games 10 --force-move --sample
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import time

from training.bc.inference import BCModelHandle, BCPerspective, default_device
from eval_bot.bot_config import BotConfig
from eval_bot.eval_bot_agent import EvalBotAgent
from game_runner.policy import GameResult
from game_runner.runner import run_game
from game_runner.seed_map import list_two_player_replay_ids, load_static_from_db
from self_play.nn_agent import NNAgent


def run_eval(
    checkpoint: Path,
    num_games: int,
    max_turns: int,
    force_move: bool,
    sample: bool,
    temperature: float,
    device_name: str,
    seed: int | None,
) -> None:
    device = default_device() if device_name == "auto" else __import__("torch").device(device_name)
    handle = BCModelHandle.load(checkpoint, device)
    print(f"checkpoint: {checkpoint}")
    print(f"device: {device}")
    print(f"mode: force_move={force_move} sample={sample} temperature={temperature}")
    print(f"games: {num_games}  max_turns: {max_turns}")

    rng = random.Random(seed)
    replay_ids = list_two_player_replay_ids(limit=200)
    print(f"map pool: {len(replay_ids)} candidates (seed={seed})")
    print()

    results: list[GameResult] = []
    t0 = time.perf_counter()

    for i in range(num_games):
        replay_id = rng.choice(replay_ids)
        static = load_static_from_db(replay_id)

        # Model plays slot 0, EvalBot plays slot 1.
        policies = [
            NNAgent(handle, BCPerspective(0, device, handle.model.cfg.obs,
                                          force_move=force_move,
                                          sample=sample, temperature=temperature)),
            EvalBotAgent(perspective_slot=1, cfg=BotConfig()),
        ]

        result = run_game(policies, static, max_turns=max_turns)
        results.append(result)

        winner_str = f"p{result.winner}" if result.winner is not None else "draw"
        model_won = result.winner == 0
        print(f"  game {i+1:3d}/{num_games}: {winner_str:5s}  "
              f"len={result.game_length:4d}  "
              f"model_land={result.player_stats[0].land:3d}  "
              f"bot_land={result.player_stats[1].land:3d}"
              f"{'  ✓' if model_won else ''}")

    elapsed = time.perf_counter() - t0

    # Summary
    model_wins = sum(1 for r in results if r.winner == 0)
    bot_wins = sum(1 for r in results if r.winner == 1)
    draws = sum(1 for r in results if r.winner is None)
    avg_length = sum(r.game_length for r in results) / len(results)

    print()
    print(f"results ({num_games} games, {elapsed:.1f}s):")
    print(f"  model wins: {model_wins:3d}  ({model_wins/num_games:.0%})")
    print(f"  bot wins:   {bot_wins:3d}  ({bot_wins/num_games:.0%})")
    print(f"  draws:      {draws:3d}  ({draws/num_games:.0%})")
    print(f"  avg length: {avg_length:.0f} ticks")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--max-turns", type=int, default=1000)
    parser.add_argument("--force-move", action="store_true")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_eval(
        checkpoint=args.checkpoint,
        num_games=args.games,
        max_turns=args.max_turns,
        force_move=args.force_move,
        sample=args.sample,
        temperature=args.temperature,
        device_name=args.device,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
