"""Defines what an eval run *is*, independent of how it was requested.

`EvalRunSpec` is the contract between CLI entry points (e.g.
scripts/run_adhoc.py) and the runner: a frozen, serializable description of
one eval run, with no argparse coupling. The run dir's config.json is
`asdict(spec)` plus a timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass


class EvalConfigError(ValueError):
    """A spec that can't be run: bad policy spec, rotation collision,
    map/player-count mismatch, etc. Raised by the runner's up-front
    validation; CLI entry points catch it and report without a traceback."""


@dataclass(frozen=True)
class EvalRunSpec:
    """One eval run: who plays (`policy_specs`, one per slot), on which maps,
    how many games, and the runtime knobs. String fields (`maps`, `device`)
    keep their compact serialized form — the runner interprets them."""

    policy_specs: list[str]      # one per player slot
    maps: str = "random:10"      # 'random:N' | 'replay_id:id1,id2,...'
    map_seed: int = 42
    games_per_map: int = 1
    slot_rotations: int = 1      # effective K; CLI sugar like --swap-slots is resolved before construction
    skip_dupes: bool = False
    max_turns: int = 1000
    device: str = "auto"         # 'auto' | 'cpu' | 'mps' | 'cuda'
    sample_interval: int = 25    # land/army curve sampling, in ticks (0 = off)
    concurrent_games: int = 0    # batched-pool size; 0 = auto (~32 NN rows/forward)
