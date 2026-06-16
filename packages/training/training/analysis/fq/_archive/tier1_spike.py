#!/usr/bin/env -S uv run python
"""fq spike (SCRATCH — not the real toolkit).

Validates the `frame_view` walk foundation and produces the Tier-1 army/margin
answer in one pass (docs 6.15 thread):

  GOAL 1  walk: iterate IterableDataset directly with frame_view=True and read
          raw sim + encoded obs + the join key per frame.
  GOAL 2  parity: assert army-from-obs == army-from-sim (settle sim-vs-obs).
  GOAL 3  Tier-1: per-player army distribution + bottom-two-margin among alive
          players, overall and conditioned on dt<10 (the 0.42σ question).
  GOAL 4  ergonomics signal for the design doc (noted in chat, not here).

Hardcoded columns / no Deriver/FrameTable/CLI abstractions — that's deliberate;
this is the concrete instance we abstract *from*.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from settings import TRAINING_DATA_DIR
from training.analysis.obs_utils import army_totals_ch_indices, per_player_army
# Reaching into a script's private for sim-army is exactly the duplication the
# canonical-deriver design should absorb (ergonomics signal, GOAL 4).
from training.analysis.scripts.who_dies_next_baselines import _army_totals_per_tick
from training.bc.dataset import IterableDataset
from training.bc.obs_config import OBS_CONFIG_DEFAULTS, ObsConfig
from training.bc.splits import load_manifest, samples_for_split

MANIFEST = TRAINING_DATA_DIR / "splits" / "cloud_24k_sub_11k.json"


def cap_by_games(val: list[tuple[Path, int]], max_games: int):
    """Keep whole games (all perspectives of the first `max_games` distinct sims)."""
    keep = set(list(dict.fromkeys(p for p, _ in val))[:max_games])
    return [(p, k) for p, k in val if p in keep], len(keep)


def pctiles(a: np.ndarray) -> dict[int, float]:
    return {p: round(float(np.percentile(a, p)), 1) for p in (1, 25, 50, 75, 90, 99)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-games", type=int, default=100)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    args = ap.parse_args()

    man = load_manifest(args.manifest)
    intermediate = Path(man["intermediate_root"])
    val = samples_for_split(man, "val", intermediate)
    samples, n_games = cap_by_games(val, args.max_games)
    print(f"{args.manifest.name}: {len(val)} val perspectives; "
          f"capped to {n_games} games / {len(samples)} perspectives")

    # fp32 obs so the parity check isn't muddied by fp16 storage quantization
    # (the training default is fp16; Tier-1 reads sim ground truth, so unaffected).
    cfg = ObsConfig(dense_history_n=OBS_CONFIG_DEFAULTS.dense_history_n, obs_dtype="fp32")
    army_ch = army_totals_ch_indices(cfg.dense_history_n)
    ds = IterableDataset(
        samples=samples, seed=0, obs_cfg=cfg,
        elim_head_variant="next_death",   # emits next_elim_target/dt + alive_mask
        include_frame_info=True,
        frame_view=True,
    )

    army_rows, alive_rows, dt_rows = [], [], []
    max_parity = 0.0
    sim_totals, sim_id = None, None
    for frame in ds:
        fv = frame["frame_view"]
        if id(fv.sim) != sim_id:          # per-game compute; sim is shared by ref
            sim_totals = _army_totals_per_tick(fv.sim)
            sim_id = id(fv.sim)
        raw = [fv.perspective_slot, *fv.opp_slots]
        army_sim = sim_totals[fv.t, raw].astype(np.float64)                     # [8]
        army_obs = per_player_army(frame["obs"], frame["valid_mask"], army_ch)
        army_obs = army_obs.numpy().astype(np.float64) * 1000.0                 # [8]
        alive = frame["alive_mask"].numpy().astype(bool)                        # [8]
        if alive.any():
            max_parity = max(max_parity, float(np.abs(army_obs[alive] - army_sim[alive]).max()))
        army_rows.append(army_sim)
        alive_rows.append(alive)
        dt_rows.append(int(frame["next_elim_dt"]))

    army = np.array(army_rows)     # [N, 8] real army units (sim ground truth)
    alive = np.array(alive_rows)   # [N, 8] bool
    dt = np.array(dt_rows)         # [N]
    n = army.shape[0]
    print(f"walked {n} frames over {n_games} games")

    print("\n# GOAL 2 — sim/obs army parity (alive players)")
    verdict = "MATCH" if max_parity < 2.0 else "MISMATCH"
    print(f"  max |army_obs*1000 - army_sim| = {max_parity:.4f}  ({verdict})")

    print("\n# GOAL 3 — Tier-1 distributions (real army units)")
    flat = army[alive]
    print(f"  alive army: mean {flat.mean():.0f}  std {flat.std():.0f}  "
          f"min {flat.min():.0f}  max {flat.max():.0f}")
    print(f"    pctiles {pctiles(flat)}")
    per_player_std = [army[alive[:, p], p].std() for p in range(8) if alive[:, p].sum() > 1]
    sigma = float(np.mean(per_player_std))
    print(f"  per-player army std (mean over players): {sigma:.0f}  "
          f"->  0.42σ ≈ {0.42 * sigma:.0f} army")

    margin = np.full(n, -1.0)
    for i in range(n):
        a = np.sort(army[i][alive[i]])
        if a.size >= 2:
            margin[i] = a[1] - a[0]
    valid = margin >= 0
    imm = valid & (dt >= 0) & (dt < 10)
    res = 0.42 * sigma

    def margin_report(mask: np.ndarray, label: str) -> None:
        m = margin[mask]
        if m.size == 0:
            print(f"  {label}: no frames")
            return
        print(f"  {label}: n={m.size}  median {np.median(m):.0f}  pctiles {pctiles(m)}")
        for thr, name in [(res, "0.42σ"), (50, "50"), (100, "100"), (200, "200")]:
            print(f"      frac margin < {name:>5}: {(m < thr).mean() * 100:5.1f}%")

    print("\n  bottom-two army margin among alive players:")
    margin_report(valid, "all (>=2 alive)")
    margin_report(imm, "imminent dt<10 ")


if __name__ == "__main__":
    main()
