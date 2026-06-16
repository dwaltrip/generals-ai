"""The dumb columnar container + the walk that builds it.

`FrameTable` holds columns and nothing else; slicing is the one convenience
(`select`), everything else is plain numpy on `.cols`. We deliberately never grow
groupby/agg/pivot/merge here — if tempted, drop to numpy (or `pd.DataFrame(t.cols)`
for all-`[N]` columns). `build_frame_table` walks `IterableDataset` once and runs
each deriver per frame; the complexity lives in the (write-once) derivers, not here.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np

from training.analysis.fq.derivers import Deriver, Frame
from training.bc.dataset import IterableDataset, SimFrame
from training.bc.obs_config import ObsConfig


@dataclass
class FrameSpec:
    """A family: what the dataset must emit and which columns to build.

    `emit_cols` copies dataset-emitted fields straight into columns (quantities
    the encode path already computes — alive/victim/dt); `derivers` are computed
    from the `Frame`; `derived_cols` are plain-numpy functions applied to the
    *built* table (the lowest-army rule lives here, matched to the eval frames by
    construction). `dataset_kwargs` is inlined for the MVP (Q9: -> FrameNeeds).
    """

    name: str
    dataset_kwargs: dict          # inline IterableDataset kwargs
    emit_cols: dict[str, str]     # dataset-emitted field -> column name
    derivers: list[Deriver]       # computed-from-Frame columns
    derived_cols: dict[str, Callable[[FrameTable], np.ndarray]] = field(default_factory=dict)


@dataclass
class FrameTable:
    """Columns + identity. Axis 0 of every column is the frame index `N`
    (`[N]` per-frame, `[N, 8]` per-player), so `select` masks uniformly and
    `t.cols["army_sim"][t.cols["alive"]]` flattens to the long 1-D of alive values.

    `sample_idx` is the dataset's emitted index — a *subset* index once capped by
    games, so it identifies rows within this table only. The full-split join key
    (`persp_val_index`) and `join_dump` are a deferred follow-up (design Appendix A).
    `game_id` is a walk-local monotonic counter (representativeness only; correct
    only at `shuffle_buffer_size=0`).
    """

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
        sample_idx=t.sample_idx[mask],
        frame_t=t.frame_t[mask],
        game_id=t.game_id[mask],
        n_games=int(np.unique(t.game_id[mask]).size),
    )


def cap_by_games(samples: list[tuple[Path, int]], max_games: int) -> list[tuple[Path, int]]:
    """Keep all perspectives of the first `max_games` games. Caps by whole games
    (never mid-game), so per-game state and representativeness stay correct."""
    keep = set(list(dict.fromkeys(p for p, _ in samples))[:max_games])
    return [(p, k) for p, k in samples if p in keep]


def build_frame_table(
    spec: FrameSpec,
    samples: list[tuple[Path, int]],
    obs_cfg: ObsConfig,
    max_games: int | None = None,
) -> FrameTable:
    """Walk the samples once with the spec's emit, run each deriver per frame, and
    stack into a `FrameTable`.

    Runs at `shuffle_buffer_size=0`: `game_id`, `n_games`, and the `per_game` cache
    are correct only on the contiguous, tick-ordered walk — a reservoir shuffle
    would interleave games and silently corrupt all three.
    """
    if max_games is not None:
        samples = cap_by_games(samples, max_games)
    assert "shuffle_buffer_size" not in spec.dataset_kwargs, (
        "build_frame_table fixes shuffle_buffer_size=0; the spec must not set it"
    )
    ds = IterableDataset(
        samples=samples, seed=0, obs_cfg=obs_cfg, shuffle_buffer_size=0, **spec.dataset_kwargs
    )

    cols: dict[str, list] = defaultdict(list)
    sample_idx: list[int] = []
    frame_t: list[int] = []
    game_id: list[int] = []
    prev_sim_id, gid = None, -1
    for d in ds:
        sf = cast(SimFrame, d["sim_frame"])  # non-Tensor analysis seam in the sample dict
        if id(sf.sim) != prev_sim_id:        # contiguous games -> monotonic gid
            prev_sim_id, gid = id(sf.sim), gid + 1
        f = Frame(
            obs=d["obs"],
            valid_mask=d["valid_mask"],
            alive=d["alive_mask"].numpy().astype(bool),
            sim=sf.sim,
            t=sf.t,
            raw_order=[sf.perspective_slot, *sf.opp_slots],
            game_id=gid,
        )
        for src, colname in spec.emit_cols.items():
            v = d[src]
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


def check_representative(t: FrameTable, per_player_target: np.ndarray) -> None:
    """The 6.15-2 guard (opt-in helper): a per-group metric needs each group to
    hold enough GAMES (not just frames) and enough target spread, or a
    variance-normalized metric inverts. Reports per player slot. Call it before
    computing per-group metrics — the dumb container can't know your grouping, so
    it won't auto-run this."""
    print(f"  n_games={t.n_games}  n_frames={t.frame_t.size}")
    for p in range(8):
        sel = t.cols["alive"][:, p]
        col = per_player_target[sel, p]
        ng = int(np.unique(t.game_id[sel]).size)
        print(f"    p{p}: alive={int(sel.sum()):6d}  games={ng:3d}  std={col.std():6.0f}")
