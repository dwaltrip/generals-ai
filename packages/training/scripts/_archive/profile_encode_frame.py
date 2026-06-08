#!/usr/bin/env -S uv run python
"""Per-region timing of `encode_frame` over a few representative games.

Mirrors `dataset._walk`'s inner loop but instruments each region so we can see
which slice of build_obs (BFS, dense-history, channel builders, ...) dominates
the per-sample budget. The 6.05-1 throughput bench surfaced obs-construction
as the data-pipeline bottleneck; this is its companion microscope.

Output is mean ms/sample per region, totals per game, a final aggregate, and
the BFS cache hit rate (cheap diagnostic for `BFSCache.maybe_invalidate`).

Run from the training package dir:

    ./scripts/profile_encode_frame.py

Games are sampled from local `replay-parser/data/intermediate/` shards. Edit
the `GAMES` list at the top of this module to point at different replays.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys
import time

import numpy as np

from bc import actions, bfs
from bc.constants import H_PADDED, W_PADDED, obs_channel_count
from bc.mask import build_mask
from bc.obs import (
    MemoryState,
    _cat_bfs,
    _cat_contact_capture,
    _cat_dense_history,
    _cat_memory,
    _cat_opp_broadcast,
    _cat_persistent_map,
    _cat_scoreboard,
    _cat_self_broadcast,
    _cat_visibility,
    _cat_visible_state,
    canonical_slot_order,
    init_memory,
    step_memory,
)
from bc.obs_config import OBS_CONFIG_DEFAULTS
from bc.visibility import compute_visibility


GAMES = [
    "replay-parser/data/intermediate/pc/Pcx0OGzqK.npz",
    "replay-parser/data/intermediate/y2/y2ruajm3x.npz",
    "replay-parser/data/intermediate/vZ/vza4p9P8y.npz",
    "replay-parser/data/intermediate/v6/V6H3rus6k.npz",
    "replay-parser/data/intermediate/zQ/ZqxBalRxm.npz",
]


def _profile_game(
    sim_path: Path,
    totals: dict[str, float],
    cache_counts: dict[str, int],
) -> tuple[int, float]:
    """Walk one game from perspective 0 and accumulate ns into `totals`.

    Also accumulates `cache_counts["lookups"]` and `cache_counts["bfs_runs"]`
    from the BFSCache so we can report cache hit rate.

    Returns (n_frames, wall_ns).
    """
    meta_path = sim_path.with_name(sim_path.stem + ".meta.npz")
    with np.load(sim_path) as sim_npz:
        sim = {key: sim_npz[key] for key in sim_npz.files}
    with np.load(meta_path) as meta_npz:
        meta = {key: meta_npz[key] for key in meta_npz.files}

    T = sim["ownership"].shape[0]
    H = int(sim["map_height"])
    W = int(sim["map_width"])

    k = 0
    perspective_slot = int(meta["perspective_player_ids"][k])
    opp_slots = canonical_slot_order(perspective_slot)[1:]

    state: MemoryState = init_memory(sim, perspective_slot, H, W, OBS_CONFIG_DEFAULTS)
    bfs_cache = bfs.init_bfs_cache()

    elim_t = int(meta["elim_timestep"][k])
    end_t = T - 1 if elim_t == -1 else min(T - 1, elim_t)

    n_channels = obs_channel_count(state.obs_cfg.dense_history_n)

    wall0 = time.perf_counter_ns()
    n_frames = 0

    for t in range(end_t):
        # ---- compute_visibility ----
        a = time.perf_counter_ns()
        vis = compute_visibility(sim["ownership"][t], perspective_slot, H, W)
        b = time.perf_counter_ns()
        totals["visibility"] += b - a

        # ---- step_memory ----
        a = time.perf_counter_ns()
        step_memory(state, sim, t, vis, perspective_slot, H, W)
        b = time.perf_counter_ns()
        totals["step_memory"] += b - a

        # ---- build_obs prelude (cross-cat helper) ----
        a = time.perf_counter_ns()
        own = sim["ownership"][t].reshape(H, W)
        armies = sim["armies"][t].reshape(H, W)
        cities_now_flat = np.zeros(H * W, dtype=bool)
        cities_now_flat[sim["cities"][sim["cities_present_at"] <= t]] = True
        cities_now_2d = cities_now_flat.reshape(H, W)
        structures_in_fog_mask = (
            (state.is_structure | cities_now_2d)
            & ~state.known_mountain
            & ~state.known_city
            & ~state.known_general
        )
        b = time.perf_counter_ns()
        totals["obs_prelude"] += b - a

        # ---- cat 1: visibility ----
        a = time.perf_counter_ns()
        c1 = _cat_visibility(vis)
        b = time.perf_counter_ns()
        totals["cat1_visibility"] += b - a

        # ---- cat 2: visible state ----
        a = time.perf_counter_ns()
        c2 = _cat_visible_state(vis, own, armies, perspective_slot, opp_slots)
        b = time.perf_counter_ns()
        totals["cat2_visible_state"] += b - a

        # ---- cat 3: persistent map ----
        a = time.perf_counter_ns()
        c3 = _cat_persistent_map(state, structures_in_fog_mask)
        b = time.perf_counter_ns()
        totals["cat3_persistent_map"] += b - a

        # ---- cat 4: memory ----
        a = time.perf_counter_ns()
        c4 = _cat_memory(state, perspective_slot, opp_slots)
        b = time.perf_counter_ns()
        totals["cat4_memory"] += b - a

        # ---- cat 5: BFS distance-from-known-generals ----
        a = time.perf_counter_ns()
        c5 = _cat_bfs(
            state, t, perspective_slot, opp_slots, vis, own, armies,
            structures_in_fog_mask, bfs_cache, H, W,
        )
        b = time.perf_counter_ns()
        totals["cat5_bfs"] += b - a

        # ---- cat 6: self broadcast ----
        a = time.perf_counter_ns()
        c6 = _cat_self_broadcast(state, t, perspective_slot, H, W)
        b = time.perf_counter_ns()
        totals["cat6_self_broadcast"] += b - a

        # ---- cat 7: opp broadcast ----
        a = time.perf_counter_ns()
        c7 = _cat_opp_broadcast(state, t, opp_slots, H, W)
        b = time.perf_counter_ns()
        totals["cat7_opp_broadcast"] += b - a

        # ---- cat 8: scoreboard ----
        a = time.perf_counter_ns()
        c8 = _cat_scoreboard(state, t, opp_slots, H, W)
        b = time.perf_counter_ns()
        totals["cat8_scoreboard"] += b - a

        # ---- cat 9: contact/capture ----
        a = time.perf_counter_ns()
        c9 = _cat_contact_capture(state, perspective_slot, opp_slots, H, W)
        b = time.perf_counter_ns()
        totals["cat9_contact_capture"] += b - a

        # ---- cat 10: dense history ----
        a = time.perf_counter_ns()
        c10 = _cat_dense_history(state, H, W)
        b = time.perf_counter_ns()
        totals["cat10_dense_history"] += b - a

        # ---- assemble into padded buffer ----
        a = time.perf_counter_ns()
        channels = [*c1, *c2, *c3, *c4, *c5, *c6, *c7, *c8, *c9, *c10]
        assert len(channels) == n_channels
        obs = np.zeros((n_channels, H_PADDED, W_PADDED), dtype=np.float32)
        for i, ch in enumerate(channels):
            obs[i, :H, :W] = ch
        b = time.perf_counter_ns()
        totals["obs_assemble"] += b - a

        # ---- build_mask ----
        a = time.perf_counter_ns()
        _mask_np = build_mask(sim, t, perspective_slot, H, W)
        b = time.perf_counter_ns()
        totals["build_mask"] += b - a

        # ---- valid_mask + actions.encode + tensor conversion ----
        a = time.perf_counter_ns()
        valid_mask_np = np.zeros((1, H_PADDED, W_PADDED), dtype=np.bool_)
        valid_mask_np[0, :H, :W] = True
        src = int(sim["actions_source"][perspective_slot, t])
        dst = int(sim["actions_dest"][perspective_slot, t])
        is50 = int(sim["actions_is50"][perspective_slot, t])
        _is_pass, _flat_idx = actions.encode(src, dst, is50, W, W_PADDED)
        b = time.perf_counter_ns()
        totals["valid_mask_actions"] += b - a

        n_frames += 1

    wall_ns = time.perf_counter_ns() - wall0
    cache_counts["lookups"] += bfs_cache.n_lookups
    cache_counts["bfs_runs"] += bfs_cache.n_bfs_runs
    return n_frames, wall_ns


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    totals: dict[str, float] = defaultdict(float)
    total_frames = 0
    total_wall_ns = 0
    total_cache: dict[str, int] = defaultdict(int)

    print("Per-game timings + BFS cache hit rate:\n")
    print(f"{'game':<40s} {'T':>4s} {'HxW':>6s} {'frames':>7s} {'wall(s)':>8s} "
          f"{'ms/frame':>10s} {'bfs hit%':>9s}")
    for rel in GAMES:
        path = repo_root / rel
        per_game: dict[str, float] = defaultdict(float)
        per_game_cache: dict[str, int] = defaultdict(int)
        n, wall = _profile_game(path, per_game, per_game_cache)
        for k, v in per_game.items():
            totals[k] += v
        for k, v in per_game_cache.items():
            total_cache[k] += v
        total_frames += n
        total_wall_ns += wall
        with np.load(path) as z:
            T = z["ownership"].shape[0]
            H = int(z["map_height"])
            W = int(z["map_width"])
        lookups = per_game_cache["lookups"]
        runs = per_game_cache["bfs_runs"]
        hit_pct = (1 - runs / lookups) * 100 if lookups else 0.0
        print(f"{path.name:<40s} {T:>4d} {H:>2d}x{W:<3d} {n:>7d} "
              f"{wall/1e9:>8.2f} {wall/1e6/n:>10.3f} {hit_pct:>8.1f}%")

    print(f"\nAggregate over {total_frames} frames "
          f"({total_wall_ns/1e9:.2f}s wall):\n")
    print(f"{'region':<28s} {'ms/frame':>10s} {'share':>8s}")
    # sort by descending
    sum_regions_ns = sum(totals.values())
    for region in sorted(totals.keys(), key=lambda k: -totals[k]):
        ns = totals[region]
        ms_per_frame = ns / 1e6 / total_frames
        share = ns / sum_regions_ns * 100
        print(f"{region:<28s} {ms_per_frame:>10.3f} {share:>7.1f}%")

    summed_ms = sum_regions_ns / 1e6 / total_frames
    wall_ms = total_wall_ns / 1e6 / total_frames
    print(f"\n  sum-of-regions    {summed_ms:>10.3f} ms/frame")
    print(f"  wall              {wall_ms:>10.3f} ms/frame "
          f"(diff is timer overhead + np.load)")

    lookups = total_cache["lookups"]
    runs = total_cache["bfs_runs"]
    if lookups:
        hit = lookups - runs
        print(f"\nBFS cache: {hit:,} hits / {lookups:,} lookups = "
              f"{hit/lookups*100:.1f}% hit rate "
              f"({runs:,} _bfs calls actually ran)")


if __name__ == "__main__":
    sys.exit(main())
