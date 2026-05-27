"""CLI runner: play one self-play game and render the result.

Usage (from repo root):

    uv run python -m self_play.cli \\
        --checkpoint training/data/runs-cloud/2026-05-23T23-40-14Z/checkpoints/epoch_005.pt \\
        --sample --temperature 1.0

If --replay-id is omitted, picks the longest 2-player replay in the
corpus by default. Output HTML defaults to self-play/tmp/selfplay-<ts>.html.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import torch

from game_runner import seed_map, viewer
from self_play import agent, driver


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_TMP = _REPO_ROOT / "self-play" / "tmp"


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return agent.default_device()
    return torch.device(name)


def _resolve_out_path(arg_out: str | None) -> Path:
    if arg_out:
        return Path(arg_out)
    ts = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    return _DEFAULT_TMP / f"selfplay-{ts}.html"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to a BC model state_dict (.pt).")
    parser.add_argument("--replay-id", type=str, default=None,
                        help="Seed-map replay ID. Default: longest 2p replay in corpus.")
    parser.add_argument("--max-turns", type=int, default=1000,
                        help="Hard cap on game length (driver-level).")
    parser.add_argument("--sample", action="store_true",
                        help="Bernoulli-sample pass + softmax-sample move (vs argmax).")
    parser.add_argument("--force-move", action="store_true",
                        help="Never pass when a legal move exists (overrides pass head).")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Softmax temperature for move sampling (--sample only).")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "mps", "cuda"],
                        help="Inference device.")
    parser.add_argument("--progress-interval", type=int, default=50,
                        help="Print a status line every N ticks (0 to disable).")
    parser.add_argument("--out", type=str, default=None,
                        help="HTML output path. Default: tmp/selfplay-<timestamp>.html.")
    parser.add_argument("--replay-id-label", type=str, default=None,
                        help="Override the in-HTML replay label (default: 'selfplay-<ts>').")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    print(f"loading checkpoint: {args.checkpoint}")
    model = agent.load_checkpoint(args.checkpoint, device)
    print(f"device: {device}")

    replay_id = args.replay_id or seed_map.list_two_player_replay_ids(limit=1)[0]
    static = seed_map.load_static_from_db(replay_id)
    print(f"seed: replay_id={replay_id} map={static.map_width}x{static.map_height}")
    print(f"mode: sample={args.sample} force_move={args.force_move} "
          f"temperature={args.temperature} max_turns={args.max_turns}")

    state, agents = driver.play_game(
        model, static,
        device=device,
        max_turns=args.max_turns,
        sample=args.sample,
        force_move=args.force_move,
        temperature=args.temperature,
        progress_interval=args.progress_interval,
    )

    print()
    print(f"ticks={state.timestep} alive={list(state.alive)} "
          f"alive_count={state.alive_count}")
    for i, ag in enumerate(agents):
        print(f"  p{i}: pass={ag.n_passed:4d} move={ag.n_moved:4d} "
              f"no_legal={ag.n_no_legal:4d}")

    label = args.replay_id_label or f"selfplay-{dt.datetime.now():%Y%m%dT%H%M%S}"
    out_path = _resolve_out_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(viewer.build_html_from_state(label, state, static))
    print(f"\nwrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
