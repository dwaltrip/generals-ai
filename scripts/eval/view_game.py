#!/usr/bin/env -S uv run python
"""Render a saved eval game (.npz) as a timestep-viewer HTML file.

Usage:
    ./scripts/eval/view_game.py data/eval-runs/.../games/game_001.npz
    ./scripts/eval/view_game.py data/eval-runs/.../games/game_001.npz --out /tmp/game.html
    ./scripts/eval/view_game.py data/eval-runs/.../games/game_001.npz --open

Reads the compressed .npz saved by write_eval_game and produces a
self-contained HTML viewer — no sim_core replay needed since the .npz
contains the full ownership/armies snapshots at every timestep.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


# Add scripts/ to path so `eval` package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from game_runner.viewer import render_viewer_html


# TODO: Replace SimpleNamespace mocks with proper dataclasses
# (SavedGameState, SavedGameStatic, CaptureEvent, DeathEvent, etc.).
# These should live in the eval package (or game-runner) once this
# moves out of scripts/. The current duck-typed approach works but
# is untyped and fragile — field name typos are silent.

def _load_usernames(npz_path: Path, num_players: int) -> list[str]:
    """Derive display names from the sibling .meta.json if it exists."""
    meta_path = npz_path.with_suffix(".meta.json")
    if not meta_path.exists():
        return [f"Player {i+1}" for i in range(num_players)]

    import json

    from eval.policy_spec import build_policy_names

    meta = json.loads(meta_path.read_text())
    specs: list[str] = meta.get("policy_specs", [])
    if not specs:
        return [f"Player {i+1}" for i in range(num_players)]

    names = build_policy_names(specs)
    slot_map = meta.get("slot_map")
    if slot_map:
        names = [names[slot_map[s]] for s in range(len(slot_map))]
    while len(names) < num_players:
        names.append(f"Player {len(names)+1}")
    return names[:num_players]


def _load_game(npz_path: Path) -> tuple[SimpleNamespace, SimpleNamespace]:
    """Load a .npz and return (mock_state, mock_static) objects that
    satisfy the viewer's read surface."""
    data = np.load(npz_path)

    map_width = int(data["map_width"])
    map_height = int(data["map_height"])
    ownership = data["ownership"]   # [T+1, H*W] int8
    armies = data["armies"]         # [T+1, H*W] int16
    num_timesteps = ownership.shape[0]
    num_players = int(ownership[0].max()) + 1

    # Reconstruct event objects from packed arrays
    capture_events = [
        SimpleNamespace(timestep=int(row[0]), captor=int(row[1]), captured=int(row[2]))
        for row in data["capture_events"]
    ]
    death_events = [
        SimpleNamespace(timestep=int(row[0]), player=int(row[1]))
        for row in data["death_events"]
    ]
    neutralize_events = [
        SimpleNamespace(timestep=int(row[0]), player=int(row[1]))
        for row in data["neutralize_events"]
    ]

    # Determine alive status at game end from final ownership snapshot
    final_own = ownership[-1]
    alive = [bool((final_own == p).any()) for p in range(num_players)]

    # The last step_tick call is timestep = num_timesteps - 1
    # (snapshots_len = initial + one per step_tick)
    timestep = num_timesteps - 1

    state = SimpleNamespace(
        capture_events=capture_events,
        death_events=death_events,
        neutralize_events=neutralize_events,
        num_players=num_players,
        alive=alive,
        timestep=timestep,
        snapshots_len=num_timesteps,
        snapshots_ownership=[ownership[t] for t in range(num_timesteps)],
        snapshots_armies=[armies[t] for t in range(num_timesteps)],
    )

    initial_generals = data["initial_generals"].tolist()
    static = SimpleNamespace(
        map_width=map_width,
        map_height=map_height,
        usernames=_load_usernames(npz_path, num_players),
        mountains=data["mountains"].tolist(),
        initial_cities=data["initial_cities"].tolist(),
        initial_city_armies=data["initial_city_armies"].tolist(),
        initial_generals=initial_generals,
    )

    return state, static


def render_html(npz_path: Path, label: str | None = None) -> str:
    state, static = _load_game(npz_path)
    if label is None:
        label = npz_path.stem
    return render_viewer_html(label, state, static)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("npz_path", type=Path, help="Path to a game .npz file")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output HTML path (default: same dir, .html extension)")
    parser.add_argument("--open", action="store_true",
                        help="Open the HTML file in the default browser after writing")
    parser.add_argument("--label", type=str, default=None,
                        help="Display label for the viewer (default: filename stem)")
    args = parser.parse_args()

    if not args.npz_path.exists():
        print(f"error: file not found: {args.npz_path}", file=sys.stderr)
        return 1

    html = render_html(args.npz_path, label=args.label)

    out_path = args.out or args.npz_path.with_suffix(".html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    print(f"wrote: {out_path}")

    if args.open:
        subprocess.run(["open", str(out_path)])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
