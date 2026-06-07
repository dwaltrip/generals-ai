"""Equivalence guard: the live inference obs path and the training obs path must
produce identical observations for the same game.

There are two independent obs-construction code paths that re-implement the same
per-tick accumulation logic, and nothing else cross-checks them:

  * training: `init_memory` + per-tick `step_memory` + `build_obs` (dataset.py)
  * live:     `BCPerspective.reset` + per-tick `encode` (inference.py), as the
              batched runner drives it

This test runs one real `sim_core` game — terrain from a single ASCII fixture,
trajectory from the simulator — and asserts the two paths agree on the obs
tensor and the legality mask at *every* tick, for *every* perspective.

Two design choices make it robust and minimally coupled:

  * It is a *differential* test (path A vs path B), not a golden-value snapshot.
    There are no hard-coded obs numbers, so it survives obs-channel changes and
    only fails when the two paths actually diverge — which is the whole bug class.
  * The live side is driven exactly as the runner drives it: `reset()` on the
    t=0 view, then `encode()` once per tick *starting at t=0*. A sequencing bug
    in that handshake is precisely what this test exists to catch, so we must
    reproduce the handshake rather than an idealized version of it. (The board
    must also change every tick — the marching general below guarantees that, so
    any one-tick staleness shows up instead of hiding on unchanged ticks.)

No checkpoint or DB needed: `encode` is pure numpy, so both paths run on an
`ObsConfig` alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import sim_core
from bc import bfs as bc_bfs
from bc import obs as bc_obs
from bc.inference import BCPerspective
from bc.mask import build_mask
from bc.obs_config import OBS_CONFIG_DEFAULTS
from bc.visibility import compute_visibility
from game_runner.ascii_map import load_ascii_map
from game_runner.save import write_eval_game
from game_runner.sim_adapter import state_to_view

# packages/self-play/tests/ → repo root is 3 levels up.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_FIXTURE = _REPO_ROOT / "training" / "tests" / "fixtures" / "small-1.txt"

_N_TICKS = 16
_MARCH_THRESHOLD = 5  # army the general accumulates before pushing out


def _diverged_channels(a: np.ndarray, b: np.ndarray) -> list[int]:
    """Channel indices where two [C, H, W] obs tensors differ."""
    diff = a.astype(np.float32) != b.astype(np.float32)
    return np.nonzero(diff.reshape(diff.shape[0], -1).any(axis=1))[0].tolist()


def _charge_move(state, lane, head_idx, marching):
    """Slot-0 controller: pass while the general accumulates, then march one tile
    per tick down `lane`. Returns (moves, head_idx, marching)."""
    armies_now = np.asarray(state.snapshots_armies[state.timestep])
    head = lane[head_idx]
    head_army = int(armies_now[head])
    if not marching and head_army >= _MARCH_THRESHOLD:
        marching = True
    moves: list[tuple[int, int, int, int]] = []
    if marching and head_idx + 1 < len(lane) and head_army >= 2:
        moves.append((0, head, lane[head_idx + 1], 0))
        head_idx += 1
    return moves, head_idx, marching


def test_live_and_training_obs_paths_agree(tmp_path):
    static = load_ascii_map(_FIXTURE)
    H, W = static.map_height, static.map_width
    n_players = len(static.initial_generals)

    def cell(r, c):
        return r * W + c

    # Open lane out of g0 at (4,0): (4,0)→(3,0)→(3,1)→(3,2)→(3,3), all plains.
    lane = [cell(4, 0), cell(3, 0), cell(3, 1), cell(3, 2), cell(3, 3)]

    state = sim_core.new_state(static)
    obs_cfg = OBS_CONFIG_DEFAULTS
    device = torch.device("cpu")

    # --- LIVE path: drive perspectives exactly as the runner does. ---
    persps = [BCPerspective(s, device, obs_cfg) for s in range(n_players)]
    for s, p in enumerate(persps):
        p.reset(state_to_view(state, static, s))

    live_obs: list[list[np.ndarray]] = [[] for _ in range(n_players)]
    live_mask: list[list[np.ndarray]] = [[] for _ in range(n_players)]

    head_idx, marching = 0, False
    for _ in range(_N_TICKS):
        for s, p in enumerate(persps):  # capture obs BEFORE stepping — runner order
            bundle = p.encode(state_to_view(state, static, s))
            live_obs[s].append(np.asarray(bundle.obs))
            live_mask[s].append(np.asarray(bundle.policy_mask))
        moves, head_idx, marching = _charge_move(state, lane, head_idx, marching)
        state.step_tick(moves=moves, afks=[])

    # --- TRAINING path: read the realized game back the way training data is. ---
    npz_path = tmp_path / "game.npz"
    write_eval_game(state, static, npz_path)
    loaded = np.load(npz_path)
    sim = {k: loaded[k] for k in loaded.files}
    raw_generals = sim["initial_generals"]

    for s in range(n_players):
        # Match the live encoder's 8-slot general padding (perspective-specific).
        sim["initial_generals"] = bc_obs.pad_initial_generals(raw_generals, s)
        mem = bc_obs.init_memory(sim, s, H, W, obs_cfg)
        cache = bc_bfs.init_bfs_cache()
        opp = bc_obs.canonical_slot_order(s)[1:]
        for t in range(_N_TICKS):
            vis = compute_visibility(sim["ownership"][t], s, H, W)
            bc_obs.step_memory(mem, sim, t, vis, s, H, W)
            train_obs = bc_obs.build_obs(sim, t, s, opp, vis, mem, cache, H, W)
            train_mask = build_mask(sim, t, s, H, W)

            assert np.array_equal(live_obs[s][t], train_obs), (
                f"obs diverged: perspective slot {s}, tick {t}, "
                f"channels {_diverged_channels(live_obs[s][t], train_obs)}"
            )
            assert np.array_equal(live_mask[s][t], train_mask), (
                f"policy mask diverged: perspective slot {s}, tick {t}"
            )
