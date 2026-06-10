"""Load a finished eval run directory into per-game records.

An eval run dir (produced by scripts/run_adhoc.py) contains:

    config.json     — run-level settings, incl. the seat→policy-spec list
    results.jsonl   — one line per game: outcome, per-seat stats, collected metrics
    games/game_NNN.npz       — full state evolution (see game_runner.save)

GameRecord joins results.jsonl with the game's npz arrays. iter_games yields
records one at a time so callers can stream large runs without holding every
npz in memory.

Snapshot indexing convention (verified against sim_core): ownership/armies
have game_length+1 rows; a death recorded at tick d means the player still
owns tiles in snapshot d and is gone from snapshot d+1.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np


@dataclass
class GameRecord:
    game_index: int
    game_id: str  # "game_001"
    replay_id: str  # map identity; the pairing key across rotations
    slot_map: list[int]  # seat -> index into the run's policy_specs
    winner_seat: int | None  # None for draws
    game_length: int
    player_stats: list[dict]  # final land/army/n_moved/n_passed per seat
    metrics: dict  # move_analysis / curves / policy_diagnostics per seat

    # npz arrays
    map_width: int
    map_height: int
    ownership: np.ndarray  # [T+1, H*W] int8; -1 neutral, -2 mountain
    armies: np.ndarray  # [T+1, H*W] int16
    cities: np.ndarray  # [C] tile indices, incl. generals converted on capture
    cities_present_at: np.ndarray  # [C] first snapshot index where city exists
    death_events: np.ndarray  # [D, 2] (tick, seat), in death order
    capture_events: np.ndarray  # [K, 3] (tick, captor_seat, victim_seat)

    @property
    def num_players(self) -> int:
        return len(self.slot_map)


@dataclass
class Group:
    """Seats sharing one policy spec — the unit of comparison in the report."""

    label: str
    spec: str
    policy_indices: list[int]  # indices into the run's policy_specs

    def seats_in(self, slot_map: list[int]) -> list[int]:
        return [s for s, p in enumerate(slot_map) if p in self.policy_indices]


def load_config(run_dir: Path) -> dict:
    return json.loads((run_dir / "config.json").read_text())


def iter_games(run_dir: Path) -> Iterator[GameRecord]:
    games_dir = run_dir / "games"
    with open(run_dir / "results.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            idx = rec["game_index"]
            game_id = f"game_{idx:03d}"
            with np.load(games_dir / f"{game_id}.npz") as z:
                yield GameRecord(
                    game_index=idx,
                    game_id=game_id,
                    replay_id=rec["replay_id"],
                    slot_map=rec["slot_map"],
                    winner_seat=rec["winner"],
                    game_length=rec["game_length"],
                    player_stats=rec["player_stats"],
                    metrics=rec["metrics"],
                    map_width=int(z["map_width"]),
                    map_height=int(z["map_height"]),
                    ownership=z["ownership"],
                    armies=z["armies"],
                    cities=z["cities"],
                    cities_present_at=z["cities_present_at"],
                    death_events=z["death_events"].reshape(-1, 2),
                    capture_events=z["capture_events"].reshape(-1, 3),
                )


_CHECKPOINT_RE = re.compile(
    r"runs-[^/]+/(\d{4})-(\d{2})-(\d{2})[^/]*/checkpoints/epoch_(\d+)"
)


def derive_groups(
    policy_specs: list[str], labels: list[str] | None = None
) -> list[Group]:
    """Group policy indices by identical spec string, with readable labels.

    Auto-labels checkpoint specs as e.g. "0607-ep4" (run date + epoch); other
    spec kinds fall back to "policy-N". Pass `labels` (one per distinct spec,
    in first-appearance order) to override.
    """
    groups: list[Group] = []
    by_spec: dict[str, Group] = {}
    for i, spec in enumerate(policy_specs):
        if spec in by_spec:
            by_spec[spec].policy_indices.append(i)
            continue
        g = Group(label="", spec=spec, policy_indices=[i])
        by_spec[spec] = g
        groups.append(g)

    for n, g in enumerate(groups):
        if labels is not None:
            g.label = labels[n]
            continue
        m = _CHECKPOINT_RE.search(g.spec)
        g.label = f"{m.group(2)}{m.group(3)}-ep{int(m.group(4))}" if m else f"policy-{n}"

    # Disambiguate label collisions (e.g. same run dir, same epoch, different opts)
    seen: dict[str, int] = {}
    for g in groups:
        seen[g.label] = seen.get(g.label, 0) + 1
        if seen[g.label] > 1:
            g.label = f"{g.label}-{chr(ord('a') + seen[g.label] - 1)}"
    return groups
