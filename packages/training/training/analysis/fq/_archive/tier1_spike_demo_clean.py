#!/usr/bin/env -S uv run python
"""fq DESIGN DEMO (SCRATCH — not the real toolkit, not the MVP).

A cleaned-up sibling of `tier1_spike.py` that simulates the *shape* of the
proposed fq interfaces so we can feel the ergonomics before the design write-up.
It still runs end-to-end (produces the Tier-1 answer), but the point is the
shape, not the numbers.

Each PART is tagged REAL (a lean working sketch of a proposed interface) or
MOCK/SKIP (signature + intent only). The design questions it deliberately
exercises (from the question-compilation):

  - Deriver model: per-frame vs per-game derivers, and the per-game caching seam
    (the spike hand-rolled this with id(sim); here it's a reusable wrapper).   [B6/B7/B8]
  - The deriver-facing `Frame` view: encoded obs + raw sim unified in one object. [A1]
  - `FrameTable` stays a DUMB container; ops are free functions; drop to numpy.   [D16/D18]
  - A model-free rule (lowest-army) becomes a COMPUTED COLUMN, not a hardcoded
    baseline constant — matched to the eval frames by construction.              [E/L45]
  - `check_representative` as a helper (games-not-frames + per-group spread).     [F23/F24]

Explicitly NOT built here (see PART 7): join_dump, caching, the fq CLI, the
family registry, FrameNeeds routing, per-deriver declared-needs / exclude.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch import Tensor

from settings import TRAINING_DATA_DIR
from training.analysis.obs_utils import army_totals_ch_indices, per_player_army
from training.analysis.scripts.who_dies_next_baselines import _army_totals_per_tick
from training.bc.dataset import IterableDataset
from training.bc.obs_config import OBS_CONFIG_DEFAULTS, ObsConfig
from training.bc.splits import load_manifest, samples_for_split

MANIFEST = TRAINING_DATA_DIR / "splits" / "cloud_24k_sub_11k.json"
# Real design: a deriver receives the obs_cfg; here a module constant keeps the
# demo lean (army-total channels are dense_history_n-independent, so this is safe).
ARMY_CH = army_totals_ch_indices(OBS_CONFIG_DEFAULTS.dense_history_n)


# ============================================================================
# PART 1 — Frame: the deriver-facing view  [REAL]
# ----------------------------------------------------------------------------
# One object a deriver reads, unifying the dataset's encoded outputs (obs, alive)
# with the raw sim it attached via frame_view. (In the real design this is likely
# `FrameView` grown to expose obs lazily, rather than a separate class.)
# ============================================================================
@dataclass
class Frame:
    obs: Tensor
    valid_mask: Tensor
    alive: np.ndarray          # [8] bool
    sim: dict[str, np.ndarray] # raw sim, shared by reference across the game
    t: int
    raw_order: list[int]       # canonical [perspective, *opp] slots, len 8


# ============================================================================
# PART 2 — Derivers: canonical single-source columns  [REAL]
# ----------------------------------------------------------------------------
# A Deriver owns the ONE way to compute a named quantity, reused by any analysis.
# Two scopes: per-frame (row-local) and per-game (compute once per game, index
# per frame). `per_game` wraps the id(sim) caching the spike hand-rolled.
# ============================================================================
@dataclass
class Deriver:
    name: str
    per_player: bool
    fn: Callable[[Frame], np.ndarray]


def per_game(name: str, per_player: bool, prepare, index) -> Deriver:
    """A per-game deriver: `prepare(sim)` runs once per game (cached on the sim
    object, valid because the walk is single-pass and games are contiguous),
    `index(state, frame)` reads this frame's value out. Stateful → one walk only."""
    cache: dict = {"id": None, "state": None}

    def fn(f: Frame) -> np.ndarray:
        if id(f.sim) != cache["id"]:
            cache["state"] = prepare(f.sim)
            cache["id"] = id(f.sim)
        return index(cache["state"], f)

    return Deriver(name, per_player, fn)


# army from the encoded obs — what the MODEL sees (carries fp16 storage noise).
ARMY_OBS = Deriver(
    "army_obs", per_player=True,
    fn=lambda f: per_player_army(f.obs, f.valid_mask, ARMY_CH).numpy().astype(float) * 1000.0,
)
# army from raw sim — GROUND TRUTH. Per-game (totals over all ticks), then indexed.
# Reuses the canonical sim function instead of a second copy of the army logic.
ARMY_SIM = per_game(
    "army_sim", per_player=True,
    prepare=_army_totals_per_tick,                              # sim -> [T, 8]
    index=lambda totals, f: totals[f.t, f.raw_order].astype(float),
)


# ============================================================================
# PART 3 — FrameSpec: a family of columns + its dataset emit  [REAL, lean]
# ----------------------------------------------------------------------------
# A family declares what the dataset must emit and which columns to build.
# `dataset_kwargs` is inlined here; the real design routes it through
# FrameNeeds -> dataset_args_for. `emit_cols` copies dataset-emitted fields
# straight into columns; `derivers` are computed.
# ============================================================================
@dataclass
class FrameSpec:
    name: str
    dataset_kwargs: dict
    emit_cols: dict[str, str]    # yielded-frame key -> table column name
    derivers: list[Deriver]


ELIM_HEAD_DEBUG = FrameSpec(
    name="elim_head_debug",
    dataset_kwargs=dict(elim_head_variant="next_death", include_frame_info=True, frame_view=True),
    emit_cols={"alive_mask": "alive", "next_elim_target": "victim", "next_elim_dt": "dt"},
    derivers=[ARMY_OBS, ARMY_SIM],
)


# ============================================================================
# PART 4 — FrameTable: the DUMB container + free-function ops  [REAL]
# ----------------------------------------------------------------------------
# No behavior beyond holding columns. Every column's axis 0 is the frame index N
# ([N] per-frame, [N, 8] per-player). Slicing/joining/grouping are free functions
# or plain numpy — we deliberately never grow groupby/agg/pivot here.
# ============================================================================
@dataclass
class FrameTable:
    cols: dict[str, np.ndarray]
    sample_idx: np.ndarray
    frame_t: np.ndarray
    game_id: np.ndarray
    n_games: int


def select(t: FrameTable, mask: np.ndarray) -> FrameTable:
    """Boolean-mask every column at once (axis 0). The one convenience we keep —
    and resist growing. Anything fancier: drop to numpy on `.cols`."""
    return FrameTable(
        cols={k: v[mask] for k, v in t.cols.items()},
        sample_idx=t.sample_idx[mask], frame_t=t.frame_t[mask],
        game_id=t.game_id[mask], n_games=int(np.unique(t.game_id[mask]).size),
    )


# ============================================================================
# PART 5 — build_frame_table: the walk engine  [REAL]
# ----------------------------------------------------------------------------
# Walks IterableDataset directly with the spec's emit, runs each deriver per
# frame, stacks into a FrameTable. Note how small it is — the complexity lives in
# the (write-once) derivers, not here.
# ============================================================================
def cap_by_games(val: list[tuple[Path, int]], max_games: int):
    keep = set(list(dict.fromkeys(p for p, _ in val))[:max_games])
    return [(p, k) for p, k in val if p in keep]


def build_frame_table(spec: FrameSpec, val, obs_cfg, max_games: int) -> FrameTable:
    samples = cap_by_games(val, max_games)
    ds = IterableDataset(samples=samples, seed=0, obs_cfg=obs_cfg, **spec.dataset_kwargs)

    cols: dict[str, list] = defaultdict(list)
    sample_idx, frame_t, game_id = [], [], []
    prev_sim_id, gid = None, -1
    for d in ds:
        fv = d["frame_view"]
        if id(fv.sim) != prev_sim_id:        # contiguous games -> monotonic id
            prev_sim_id, gid = id(fv.sim), gid + 1
        f = Frame(
            obs=d["obs"], valid_mask=d["valid_mask"],
            alive=d["alive_mask"].numpy().astype(bool),
            sim=fv.sim, t=fv.t, raw_order=[fv.perspective_slot, *fv.opp_slots],
        )
        for key, colname in spec.emit_cols.items():
            v = d[key]
            cols[colname].append(v.numpy() if hasattr(v, "numpy") else v)
        for der in spec.derivers:
            cols[der.name].append(der.fn(f))
        sample_idx.append(int(d["sample_idx"]))
        frame_t.append(int(d["frame_t"]))
        game_id.append(gid)

    out = {k: np.array(v) for k, v in cols.items()}
    n = len(frame_t)
    for k, v in out.items():                 # axis-0 invariant (catches shape bugs)
        assert v.shape[0] == n, f"col {k} axis0 {v.shape[0]} != N {n}"
    return FrameTable(out, np.array(sample_idx), np.array(frame_t), np.array(game_id), gid + 1)


# ============================================================================
# PART 6 — Derived columns + the rule-as-column + representativeness  [REAL, lean]
# ----------------------------------------------------------------------------
# Derived columns are plain numpy over a built table (no walk, no library). The
# lowest-army RULE is one such column — so a head-vs-rule comparison is matched to
# the eval frames by construction (kills the 6.14-1 hardcoded-baseline mismatch).
# ============================================================================
def lowest_army_victim(t: FrameTable) -> np.ndarray:
    masked = np.where(t.cols["alive"], t.cols["army_sim"], np.inf)
    return masked.argmin(1)


def bottom_two_margin(t: FrameTable) -> np.ndarray:
    s = np.sort(np.where(t.cols["alive"], t.cols["army_sim"], np.inf), axis=1)
    m = s[:, 1] - s[:, 0]
    m[~np.isfinite(m)] = -1.0                # <2 alive -> sentinel
    return m


def check_representative(t: FrameTable, per_player_target: np.ndarray) -> None:
    """The 6.15-2 guard: a per-group metric needs each group to hold enough GAMES
    (not just frames) and enough target spread, or variance-normalized metrics
    invert. Here: per player slot."""
    print(f"  n_games={t.n_games}  n_frames={t.frame_t.size}")
    for p in range(8):
        sel = t.cols["alive"][:, p]
        col = per_player_target[sel, p]
        ng = int(np.unique(t.game_id[sel]).size)
        print(f"    p{p}: alive={sel.sum():6d}  games={ng:3d}  std={col.std():6.0f}")


# ============================================================================
# PART 7 — MOCK / SKIP: shape only, not built here
# ----------------------------------------------------------------------------
# join_dump(t, dump) -> FrameTable
#     Inner-join a model dump on (sample_idx, frame_t); add its columns; assert
#     shared truth columns (e.g. victim) agree. Needed for head-vs-table, not Tier-1.
#
# caching                STRETCH GOAL — tables rebuild each run; most walks are fast.
# fq.py CLI              `fq --table elim_head_debug --split val -e "<numpy on t>"`;
#                        builds the table, execs the snippet with t/np in scope.
# family registry        name -> FrameSpec lookup for the CLI (a handful of families).
# FrameNeeds routing     spec.needs -> dataset_args_for, replacing inline dataset_kwargs.
# per-deriver needs /    declare obs/sim usage so `exclude` can skip an expensive
#   exclude-skips-work   deriver's work — earns its keep at the first costly deriver.
# ============================================================================


def pctiles(a: np.ndarray) -> dict[int, float]:
    return {p: round(float(np.percentile(a, p)), 1) for p in (25, 50, 75, 90, 99)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-games", type=int, default=100)
    args = ap.parse_args()

    man = load_manifest(MANIFEST)
    val = samples_for_split(man, "val", Path(man["intermediate_root"]))
    cfg = ObsConfig(dense_history_n=OBS_CONFIG_DEFAULTS.dense_history_n, obs_dtype="fp32")

    # --- build via the spec ---
    t = build_frame_table(ELIM_HEAD_DEBUG, val, cfg, args.max_games)
    print(f"built '{ELIM_HEAD_DEBUG.name}': {t.frame_t.size} frames / {t.n_games} games"
          f"  cols={list(t.cols)}")

    # --- parity: two columns of the same quantity (settles sim-vs-obs) ---
    alive = t.cols["alive"]
    parity = float(np.abs(t.cols["army_obs"] - t.cols["army_sim"])[alive].max())
    print(f"\nparity max|army_obs-army_sim| (alive) = {parity:.4f}  "
          f"({'MATCH' if parity < 2.0 else 'MISMATCH'})")

    # --- derived columns: assign straight onto the dumb dict (inline extend) ---
    t.cols["margin"] = bottom_two_margin(t)
    t.cols["low_army_victim"] = lowest_army_victim(t)

    # --- Tier-1, expressed as table ops ---
    imm = select(t, (t.cols["dt"] >= 0) & (t.cols["dt"] < 10))
    flat = t.cols["army_sim"][alive]
    print(f"\n# Tier-1\n  alive army: mean {flat.mean():.0f}  std {flat.std():.0f}  pctiles {pctiles(flat)}")
    for tbl, lbl in [(t, "all      "), (imm, "imminent ")]:
        mar = tbl.cols["margin"][tbl.cols["margin"] >= 0]
        print(f"  margin [{lbl}]: median {np.median(mar):.0f}  <50 {(mar < 50).mean() * 100:.0f}%  "
              f"<100 {(mar < 100).mean() * 100:.0f}%")

    # --- the rule as a computed column, matched to the eval frames ---
    vmask = imm.cols["victim"] >= 0
    rule = (imm.cols["low_army_victim"][vmask] == imm.cols["victim"][vmask]).mean()
    print(f"  lowest-army RULE top1 (imminent, computed column): {rule * 100:.1f}%")

    print("\n# representativeness guard")
    check_representative(t, t.cols["army_sim"])


if __name__ == "__main__":
    main()
