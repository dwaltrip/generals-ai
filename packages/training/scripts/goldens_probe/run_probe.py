#!/usr/bin/env -S uv run python
"""Goldens pre-build size-and-runtime probe (7.10-2 §9 item 3).

Parses a handful of corpus replays to fixture npz pairs, walks the obs/targets
pipelines over them, and measures per-fixture walk time plus the compressed
size of every candidate reference arrangement: the designed event-adjacent
frozen-frame set, a minimal anchor set, the full all-frames stack (compression
bound), stats layers at two stat sets, and per-frame / per-channel hash layers.
Ends with a git pack-growth simulation for the re-bless recurring cost.

Output dir defaults to `<project>/tmp/goldens-probe/`.
Usage (from repo root):

    uv run python packages/training/scripts/goldens_probe/run_probe.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import numpy as np

# Sibling modules resolve via sys.path[0] (the script's directory).
from fixtures import (
    Candidate,
    FixtureSkip,
    candidate_replays,
    event_summary,
    load_npz,
    produce_fixture,
)
from layers import (
    STAT_SETS,
    frame_channel_hashes,
    frame_hashes,
    npy_raw_bytes,
    npz_compressed_bytes,
    pick_frozen_frames,
    stack_stats,
)
from settings import DB_PATH, PROJECT_ROOT
from training.bc.constants import H_PADDED, W_PADDED
from training.bc.filters import is_eligible
from training.bc.obs_config import OBS_CONFIG_DEFAULTS, ObsConfig
from walk import walk_end_t, walk_fixture


# The legacy occupied point (pre-status checkpoints). Constructed explicitly
# rather than importing `checkpoint.LEGACY_OBS_CFG`, whose module drags torch
# into an otherwise numpy-only probe; the build's equality assert will tie the
# two together.
LEGACY_OBS_CFG = ObsConfig(dense_history_n=5, obs_dtype="fp32", player_status_channels=False)

CONFIGS: list[tuple[str, ObsConfig]] = [
    ("default", OBS_CONFIG_DEFAULTS),
    ("legacy", LEGACY_OBS_CFG),
]

# Planning constants from 7.10-2 §3/§5: fixture count over the project's life,
# and the legacy config's designated gate-coverage subset.
PLAN_FIXTURES = 30
PLAN_LEGACY_FIXTURES = 3

# Selection specs: (label, turns quantile among candidates). All picks are
# 8-player games: the training pipeline only consumes `filters.is_eligible`
# games (player count == 8, board <= 32, T <= 2000), so goldens fixtures live
# in that population too.
PICK_SPECS = [
    ("short", 0.2),
    ("mid-1", 0.45),
    ("mid-2", 0.6),
    ("long", 0.85),
    ("mid-3", 0.5),
    ("mid-4", 0.55),
]

NP_DTYPE = {"fp16": np.float16, "fp32": np.float32}


def fmt_bytes(n: float) -> str:
    if n >= 1 << 20:
        return f"{n / (1 << 20):.2f} MiB"
    return f"{n / 1024:.1f} KiB"


def select_fixture_ids(
    conn: sqlite3.Connection, out_dir: Path, n_fixtures: int
) -> list[tuple[str, Path, Path, float]]:
    """Pick a spread of games and produce their fixture pairs.

    Returns (replay_id, sim_path, meta_path, parse_seconds) per fixture.
    """
    candidates = [c for c in candidate_replays(conn) if c.player_count == 8]
    turns = np.array([c.turns for c in candidates])
    print(f"{len(candidates):,} 8-player candidate replays in DB.")

    produced: list[tuple[str, Path, Path, float]] = []
    taken: set[str] = set()
    for label, q in PICK_SPECS[:n_fixtures]:
        target = float(np.quantile(turns, q))
        pool = sorted(candidates, key=lambda c: abs(c.turns - target))
        for cand in pool[:25]:
            if cand.replay_id in taken:
                continue
            t0 = time.perf_counter()
            try:
                sim_path, meta_path = produce_fixture(conn, cand.replay_id, out_dir / "fixtures")
            except FixtureSkip as e:
                print(f"  [{label}] skip {cand.replay_id}: {e}")
                continue
            parse_s = time.perf_counter() - t0
            if not is_eligible(sim_path):
                print(f"  [{label}] skip {cand.replay_id}: fails training eligibility")
                continue
            taken.add(cand.replay_id)
            produced.append((cand.replay_id, sim_path, meta_path, parse_s))
            print(
                f"  [{label}] picked {cand.replay_id} "
                f"(turns={cand.turns}, {cand.player_count}p, parse {parse_s:.2f}s)"
            )
            break
        else:
            print(f"  [{label}] no viable candidate in the 25 nearest — skipped")
    return produced


def collect_targets_stacks(frames_targets: list[dict]) -> dict[str, np.ndarray]:
    """Stack per-frame target dicts into [T, ...] arrays, as-produced dtypes."""
    keys = frames_targets[0].keys()
    return {k: np.stack([np.asarray(ft[k]) for ft in frames_targets]) for k in keys}


def measure_fixture(
    replay_id: str,
    sim_path: Path,
    meta_path: Path,
    parse_s: float,
    measure_dir: Path,
) -> dict:
    sim = load_npz(sim_path)
    meta = load_npz(meta_path)
    summary = event_summary(sim)
    K = len(meta["perspective_player_ids"])
    print(
        f"\n=== {replay_id}: T={summary['T']} {summary['H']}x{summary['W']} "
        f"captures={summary['captures']} deaths={summary['deaths']} "
        f"neutralizes={summary['neutralizes']} K={K} ==="
    )
    print(
        f"  fixture npz: sim {fmt_bytes(sim_path.stat().st_size)}, "
        f"meta {fmt_bytes(meta_path.stat().st_size)} (parse {parse_s:.2f}s)"
    )

    result: dict = {"replay_id": replay_id, "summary": summary, "configs": {}}
    measure_dir.mkdir(parents=True, exist_ok=True)

    for cfg_name, obs_cfg in CONFIGS:
        C = obs_cfg.obs_channels
        dtype = NP_DTYPE[obs_cfg.obs_dtype]

        end_ts = [walk_end_t(sim, meta, k) for k in range(K)]
        walkable = [k for k in range(K) if end_ts[k] >= 2]
        if not walkable:
            print(f"  [{cfg_name}] no walkable perspective — skipped")
            continue
        # Size artifacts on the longest-lived perspective (typically the
        # winner's full-game walk); time every perspective.
        sizing_k = max(walkable, key=lambda k: end_ts[k])

        walk_times: list[float] = []
        frames_walked: list[int] = []
        kept: dict = {}
        for k in walkable:
            keep = k == sizing_k
            end_t = end_ts[k]
            obs_stack = np.empty((end_t, C, H_PADDED, W_PADDED), dtype=dtype) if keep else None
            mask_stack = (
                np.empty((end_t, H_PADDED, W_PADDED, 8), dtype=np.bool_) if keep else None
            )
            frames_targets: list[dict] = []
            t0 = time.perf_counter()
            for frame in walk_fixture(sim, meta, k, obs_cfg):
                if keep:
                    obs_stack[frame.t] = frame.obs
                    mask_stack[frame.t] = frame.mask
                    frames_targets.append(frame.targets)
            walk_times.append(time.perf_counter() - t0)
            frames_walked.append(end_t)
            if keep:
                kept = {
                    "obs_stack": obs_stack,
                    "mask_stack": mask_stack,
                    "targets": collect_targets_stacks(frames_targets),
                    "end_t": end_t,
                }

        obs_stack = kept["obs_stack"]
        end_t = kept["end_t"]
        s_per_frame = float(np.sum(walk_times) / np.sum(frames_walked))
        mean_walk = end_t * s_per_frame  # per-perspective walk at full game length
        print(
            f"  [{cfg_name}] C={C} {obs_cfg.obs_dtype}: "
            f"{np.sum(frames_walked)} frames over {len(walkable)} persps in "
            f"{np.sum(walk_times):.2f}s ({1 / s_per_frame:.0f} frames/s); "
            f"full-length persp ({end_t} frames) ~{mean_walk:.2f}s"
        )

        # --- reference arrangements, sized on the first walkable perspective ---
        pre = measure_dir / f"{replay_id}-{cfg_name}"
        picks = pick_frozen_frames(sim, end_t)
        anchors = [0, end_t - 1]
        sizes = {
            "frozen_designed": npz_compressed_bytes(
                Path(f"{pre}-frozen.npz"), obs=obs_stack[picks]
            ),
            "frozen_anchors": npz_compressed_bytes(
                Path(f"{pre}-anchors.npz"), obs=obs_stack[anchors]
            ),
            "full_stack": npz_compressed_bytes(Path(f"{pre}-full.npz"), obs=obs_stack),
        }
        raw_full = obs_stack.nbytes
        print(
            f"    frozen designed: {len(picks)} frames -> {fmt_bytes(sizes['frozen_designed'])}"
            f" | anchors(2) -> {fmt_bytes(sizes['frozen_anchors'])}"
            f" | full stack {fmt_bytes(raw_full)} raw -> {fmt_bytes(sizes['full_stack'])}"
            f" ({raw_full / sizes['full_stack']:.1f}:1)"
        )

        stats_arrays = {}
        for set_name, stat_names in STAT_SETS.items():
            t0 = time.perf_counter()
            arr = stack_stats(obs_stack, stat_names)
            build_s = time.perf_counter() - t0
            stats_arrays[set_name] = arr
            sizes[f"stats_{set_name}_raw"] = npy_raw_bytes(Path(f"{pre}-stats-{set_name}.npy"), arr)
            sizes[f"stats_{set_name}_npz"] = npz_compressed_bytes(
                Path(f"{pre}-stats-{set_name}.npz"), stats=arr
            )
            print(
                f"    stats {set_name} ({len(stat_names)} stats): "
                f"raw {fmt_bytes(sizes[f'stats_{set_name}_raw'])} / "
                f"npz {fmt_bytes(sizes[f'stats_{set_name}_npz'])} "
                f"(build {build_s:.2f}s, bless-only)"
            )

        t0 = time.perf_counter()
        fh = frame_hashes(obs_stack)
        fh_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        ch = frame_channel_hashes(obs_stack)
        ch_s = time.perf_counter() - t0
        sizes["hash_frame_raw"] = npy_raw_bytes(Path(f"{pre}-hash-frame.npy"), fh)
        sizes["hash_chan_raw"] = npy_raw_bytes(Path(f"{pre}-hash-chan.npy"), ch)
        sizes["hash_chan_npz"] = npz_compressed_bytes(Path(f"{pre}-hash-chan.npz"), h=ch)
        print(
            f"    hashes: frame {fmt_bytes(sizes['hash_frame_raw'])} ({fh_s:.2f}s, "
            f"{100 * fh_s / mean_walk:.1f}% of walk) | per-channel "
            f"{fmt_bytes(sizes['hash_chan_raw'])} raw / {fmt_bytes(sizes['hash_chan_npz'])} npz "
            f"({ch_s:.2f}s, {100 * ch_s / mean_walk:.1f}% of walk)"
        )

        result["configs"][cfg_name] = {
            "s_per_frame": s_per_frame,
            "frames": end_t,
            "n_frozen_designed": len(picks),
            "sizes": sizes,
            "stats_ext": stats_arrays["ext"],
            "chan_hashes": ch,
        }

        # Targets are obs-config independent; measure once, from the default walk.
        if cfg_name == "default":
            targets = kept["targets"]
            tgt_size = npz_compressed_bytes(Path(f"{pre}-targets.npz"), **targets)
            mask_size = npz_compressed_bytes(
                Path(f"{pre}-maskstack.npz"), mask=kept["mask_stack"]
            )
            result["targets_npz"] = tgt_size
            result["mask_npz"] = mask_size
            print(
                f"    targets (all frames): {fmt_bytes(tgt_size)} | "
                f"legality-mask stack: {fmt_bytes(mask_size)}"
            )

    result["fixture_bytes"] = sim_path.stat().st_size + meta_path.stat().st_size
    return result


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    )
    return r.stdout


def _pack_kib(repo: Path) -> int:
    _git(repo, "gc", "-q")
    for line in _git(repo, "count-objects", "-v").splitlines():
        if line.startswith("size-pack:"):
            return int(line.split()[1])
    raise RuntimeError("no size-pack in git count-objects output")


def pack_growth_sim(results: list[dict], sim_root: Path) -> None:
    """Price re-bless git-history growth for the hash+stats arrangement, per
    storage format. Two re-bless scenarios: a targeted change perturbing 5% of
    frames, and a full-churn change perturbing every frame."""
    print("\n=== Pack-growth simulation (stats-ext + per-channel hashes, default cfg) ===")
    layers = {
        r["replay_id"]: (r["configs"]["default"]["stats_ext"], r["configs"]["default"]["chan_hashes"])
        for r in results
        if "default" in r["configs"]
    }
    rng = np.random.default_rng(0)

    def perturbed(stats: np.ndarray, hashes: np.ndarray, frac: float, salt: int):
        T = stats.shape[0]
        n = max(1, int(T * frac))
        sel = rng.choice(T, size=n, replace=False)
        s2, h2 = stats.copy(), hashes.copy()
        s2[sel] += np.float32(1e-3 * (salt + 1))
        h2[sel] = rng.integers(0, 2**63, size=h2[sel].shape, dtype=np.uint64)
        return s2, h2

    for fmt in ("raw", "npz"):
        # Fresh repo per run: a reused repo already contains this sim's
        # deterministic blobs, so git dedup would report zero growth.
        n = 0
        while (repo := sim_root / f"packsim-{fmt}-{n}").exists():
            n += 1
        repo.mkdir(parents=True)
        _git(repo, "init", "-q")

        def write_all(data: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
            for rid, (stats, hashes) in data.items():
                if fmt == "raw":
                    np.save(repo / f"{rid}-stats.npy", stats)
                    np.save(repo / f"{rid}-hashes.npy", hashes)
                else:
                    np.savez_compressed(repo / f"{rid}-stats.npz", stats=stats)
                    np.savez_compressed(repo / f"{rid}-hashes.npz", h=hashes)

        write_all(layers)
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "bless")
        base = initial = _pack_kib(repo)

        deltas = {}
        for i, (scenario, frac) in enumerate([("targeted-5pct", 0.05), ("full-churn", 1.0)]):
            write_all({rid: perturbed(s, h, frac, i) for rid, (s, h) in layers.items()})
            _git(repo, "add", "-A")
            _git(repo, "commit", "-qm", f"re-bless {scenario}")
            now = _pack_kib(repo)
            deltas[scenario] = now - base
            base = now

        scale = PLAN_FIXTURES / max(len(layers), 1)
        print(
            f"  [{fmt}] initial pack {fmt_bytes(initial * 1024)} | re-bless growth: "
            + " | ".join(
                f"{sc} +{fmt_bytes(d * 1024)} (x{scale:.0f} fixtures -> +{fmt_bytes(d * 1024 * scale)})"
                for sc, d in deltas.items()
            )
        )


def report(results: list[dict]) -> None:
    print("\n=== Extrapolation to the planning basis "
          f"({PLAN_FIXTURES} fixtures, legacy on {PLAN_LEGACY_FIXTURES}) ===")

    def mean_size(cfg: str, key: str) -> float:
        vals = [r["configs"][cfg]["sizes"][key] for r in results if cfg in r["configs"]]
        return float(np.mean(vals)) if vals else 0.0

    def mean_walk(cfg: str) -> float:
        """Mean full-length-perspective walk seconds across fixtures."""
        vals = [
            r["configs"][cfg]["s_per_frame"] * r["configs"][cfg]["frames"]
            for r in results
            if cfg in r["configs"]
        ]
        return float(np.mean(vals)) if vals else 0.0

    w_def, w_leg = mean_walk("default"), mean_walk("legacy")
    print(f"  full-length walk s/perspective: default {w_def:.2f}, legacy {w_leg:.2f}")
    for p in (1, 2):
        total = PLAN_FIXTURES * p * w_def + PLAN_LEGACY_FIXTURES * p * w_leg
        print(f"  suite runtime at {p} perspective(s)/fixture: ~{total:.0f}s")

    arrangements = [
        ("frozen designed (event-adjacent)", "frozen_designed"),
        ("frozen anchors (2 frames)", "frozen_anchors"),
        ("full all-frames stack", "full_stack"),
        ("stats base raw", "stats_base_raw"),
        ("stats base npz", "stats_base_npz"),
        ("stats ext raw", "stats_ext_raw"),
        ("stats ext npz", "stats_ext_npz"),
        ("per-channel hashes raw", "hash_chan_raw"),
        ("per-channel hashes npz", "hash_chan_npz"),
        ("per-frame hashes raw", "hash_frame_raw"),
    ]
    print(f"\n  committed bytes per arrangement (default x{PLAN_FIXTURES} + legacy x{PLAN_LEGACY_FIXTURES}):")
    for label, key in arrangements:
        total = PLAN_FIXTURES * mean_size("default", key) + PLAN_LEGACY_FIXTURES * mean_size(
            "legacy", key
        )
        print(f"    {label:<38} {fmt_bytes(total)}")

    tgt = float(np.mean([r["targets_npz"] for r in results if "targets_npz" in r]))
    msk = float(np.mean([r["mask_npz"] for r in results if "mask_npz" in r]))
    fix = float(np.mean([r["fixture_bytes"] for r in results]))
    print(f"    {'targets (all frames) x30':<38} {fmt_bytes(tgt * PLAN_FIXTURES)}")
    print(f"    {'legality-mask stacks x30':<38} {fmt_bytes(msk * PLAN_FIXTURES)}")
    print(f"    {'fixture npz pairs x30':<38} {fmt_bytes(fix * PLAN_FIXTURES)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixtures", type=int, default=5)
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "tmp" / "goldens-probe")
    ap.add_argument("--skip-pack-sim", action="store_true")
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        picked = select_fixture_ids(conn, args.out, args.fixtures)
    finally:
        conn.close()
    if not picked:
        print("No fixtures produced.", file=sys.stderr)
        return 1

    results = [
        measure_fixture(rid, sp, mp, ps, args.out / "measure") for rid, sp, mp, ps in picked
    ]
    report(results)
    if not args.skip_pack_sim:
        pack_growth_sim(results, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
