"""The dumb columnar container + the walk that builds it.

`FrameTable` holds columns and nothing else: `select` is the one slicing
convenience, everything else is plain numpy on `.cols`. No groupby/agg/pivot/merge
here — reach for numpy (or `pd.DataFrame(t.cols)` for all-`[N]` columns) instead.
`build_frame_table` walks `IterableDataset` once and runs each deriver per frame;
the weight lives in the derivers, not here.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import cast

import numpy as np

from training.analysis.fq.derivers import Deriver, Frame
from training.bc.dataset import IterableDataset
from training.bc.emit_spec import PartialEmitSpec
from training.bc.obs_config import OBS_CONFIG_DEFAULTS, ObsConfig
from training.bc.sim_types import SimFrame


# Obs config for ground-truth fq tables: default `n`, fp32, no player-status
# channels (these tables compare `army_obs`, which lives in the base channels —
# unaffected by the gated status group). fp32 avoids the fp16 storage
# quantization on `army_obs` (so `army_obs == army_sim` holds exactly), and the
# army channels are `dense_history_n`-independent so the default count is safe. A
# table comparing `army_obs` against a specific checkpoint should instead pin
# obs_cfg to that run.
GROUND_TRUTH_OBS_CFG = ObsConfig(
    dense_history_n=OBS_CONFIG_DEFAULTS.dense_history_n,
    obs_dtype="fp32",
    player_status_channels=False,
)


@dataclass
class FrameTableSpec:
    """A table spec: what the dataset must emit and which columns to build.

    Columns come from three sources: `emit_cols` copies fields the dataset's encode
    path already computes (alive/victim/dt); `derivers` compute from the `Frame`;
    `derived_cols` are plain numpy over the *built* table.

    `derived_cols` MUST be row-local — each output row a function of that same input
    row only — because `build_frame_table` computes them once and `select` later
    just masks the result. A cross-row reduction (a rank, a normalization over all
    rows) would be wrong here; compute it in the consumer, after slicing.
    """

    name: str
    emit: PartialEmitSpec         # obs + frame_info are bound by build_frame_table
    emit_cols: dict[str, str]     # dataset-emitted field -> column name
    derivers: list[Deriver]       # columns computed from the Frame
    derived_cols: dict[str, Callable[[FrameTable], np.ndarray]] = field(default_factory=dict)
    # shared-truth cols join_dump cross-checks against a dump: {table col -> dump col}
    truth_map: dict[str, str] = field(default_factory=dict)


@dataclass
class FrameTable:
    """Columns + identity, one row per frame. Axis 0 of every column is the frame
    index `N` (`[N]` per-frame, `[N, 8]` per-player), so `select` masks uniformly
    and `t.cols["army_sim"][t.cols["alive"]]` flattens to the 1-D of alive values.
    """

    cols: dict[str, np.ndarray]
    persp_val_index: np.ndarray   # full val-split position; join key to model dumps (join_dump)
    frame_t: np.ndarray           # tick within the game; the join key's other half
    game_id: np.ndarray           # walk-local game counter (representativeness, not a stable id)
    n_games: int


def select(t: FrameTable, mask: np.ndarray) -> FrameTable:
    """Boolean-mask every column and identity array at once (axis 0), returning a
    new `FrameTable`."""
    return FrameTable(
        cols={k: v[mask] for k, v in t.cols.items()},
        persp_val_index=t.persp_val_index[mask],
        frame_t=t.frame_t[mask],
        game_id=t.game_id[mask],
        n_games=int(np.unique(t.game_id[mask]).size),
    )


def join_dump(
    t: FrameTable,
    dump: str | Path | dict[str, np.ndarray],
    truth_map: dict[str, str],
) -> FrameTable:
    """Inner-join a model dump onto the table on `(persp_val_index, frame_t)`.

    Returns a new table of only the rows that matched a dump frame, with the dump's
    columns attached. The table should be a subset of the dump's frames (same val
    split, possibly capped by games), so a match rate below 1.0 means misaligned
    keys — warns rather than fails.

    `truth_map` (`{table_col: dump_col}`) asserts the named shared-truth columns
    agree on every matched row — the check that validates fq's `persp_val_index`
    against the dump's independently-produced one.
    """
    d = dict(np.load(dump)) if not isinstance(dump, dict) else dump
    for key in ("persp_val_index", "frame_t"):
        if key not in d:
            raise ValueError(f"dump is missing join key {key!r}")

    # Composite int64 key: persp_val_index * C + frame_t, with C above every
    # frame_t on both sides so the two fields never collide. val index (~1e4) *
    # C (~1e4) stays well within int64.
    C = int(max(int(t.frame_t.max(initial=0)), int(d["frame_t"].max(initial=0)))) + 1
    tk = t.persp_val_index.astype(np.int64) * C + t.frame_t.astype(np.int64)
    dk = d["persp_val_index"].astype(np.int64) * C + d["frame_t"].astype(np.int64)

    order = np.argsort(dk, kind="stable")
    dk_sorted = dk[order]
    pos = np.searchsorted(dk_sorted, tk)
    in_range = pos < dk_sorted.size
    matched = np.zeros(tk.size, dtype=bool)
    matched[in_range] = dk_sorted[pos[in_range]] == tk[in_range]
    dump_row = order[np.clip(pos, 0, dk_sorted.size - 1)]

    coverage = float(matched.mean()) if matched.size else 0.0
    if coverage < 1.0:
        print(
            f"join_dump: WARNING {matched.sum()}/{matched.size} table rows matched "
            f"({coverage:.1%}) — table should be a subset of the dump; check key alignment.",
            file=sys.stderr,
        )

    out = select(t, matched)
    drows = dump_row[matched]
    for col, arr in d.items():
        if col in ("persp_val_index", "frame_t"):
            continue
        if col in out.cols:
            raise ValueError(f"join_dump: dump column {col!r} collides with a table column")
        out.cols[col] = arr[drows]

    for tcol, dcol in truth_map.items():
        if not np.array_equal(out.cols[tcol], d[dcol][drows]):
            raise AssertionError(
                f"join_dump: shared-truth column {tcol!r} disagrees with dump {dcol!r} "
                "on the join overlap — persp_val_index is misaligned (wrong manifest?)."
            )
    return out


def cap_by_games(samples: list[tuple[Path, int]], max_games: int) -> list[tuple[Path, int]]:
    """Keep all perspectives of the first `max_games` games. Caps by whole games
    (never mid-game), so per-game state and representativeness stay correct."""
    keep = set(list(dict.fromkeys(p for p, _ in samples))[:max_games])
    return [(p, k) for p, k in samples if p in keep]


def build_frame_table(
    spec: FrameTableSpec,
    samples: list[tuple[Path, int]],
    obs_cfg: ObsConfig,
    max_games: int | None = None,
) -> FrameTable:
    """Walk the samples once with the spec's emit, run each deriver per frame, and
    stack into a `FrameTable`.

    `samples` MUST be the complete, canonical val (or train) split (the order
    `samples_for_split` returns): that order defines full-split position, the
    `persp_val_index` join key against model dumps. Capping by games is safe (the
    full-split indices are preserved — see below), but a pre-subset list makes
    `persp_val_index` silently wrong.

    Runs at `shuffle_buffer_size=0` (asserted): `game_id`, `n_games`, and the
    `per_game` cache are correct only on the contiguous, tick-ordered walk.
    """
    # Record full-split positions before any cap; the dataset numbers `sample_idx`
    # over whatever (possibly capped) list it walks, so `subset_to_full` maps it back.
    full_pos = {pair: i for i, pair in enumerate(samples)}
    if max_games is not None:
        samples = cap_by_games(samples, max_games)
    subset_to_full = np.array([full_pos[(p, k)] for p, k in samples], dtype=np.int64)
    # TODO(fq-emit-spec-partial-fix): honor False for these two flags. `Frame`
    # reads alive_mask and sim_frame unconditionally, and game-boundary
    # detection relies on the attached sim.
    assert spec.emit.emit_alive_mask and spec.emit.attach_sim_frame, (
        "fq requires emit_alive_mask and attach_sim_frame"
        " — see TODO(fq-emit-spec-partial-fix) above"
    )
    ds = IterableDataset(
        samples=samples,
        seed=0,
        spec=spec.emit.to_spec(obs_cfg, emit_frame_info=True),
        shuffle_buffer_size=0,
    )

    cols: dict[str, list] = defaultdict(list)
    persp_val_index: list[int] = []
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
            # The dataset walk only ever attaches parser-npz sims, so the
            # narrowing from SimFrame's shared live/replay annotation is safe.
            sim=cast(dict[str, np.ndarray], sf.sim),
            t=sf.t,
            raw_order=list(sf.slot_order.order),
            game_id=gid,
        )
        for src, colname in spec.emit_cols.items():
            v = d[src]
            cols[colname].append(v.numpy() if hasattr(v, "numpy") else v)
        for der in spec.derivers:
            cols[der.name].append(der.fn(f))
        persp_val_index.append(int(subset_to_full[int(d["sample_idx"])]))
        frame_t.append(int(d["frame_t"]))
        game_id.append(gid)

    out = {k: np.array(v) for k, v in cols.items()}
    n = len(frame_t)
    for k, v in out.items():                 # axis-0 invariant (catches shape bugs)
        assert v.shape[0] == n, f"col {k} axis0 {v.shape[0]} != N {n}"
    t = FrameTable(
        out, np.array(persp_val_index), np.array(frame_t), np.array(game_id), gid + 1
    )
    # Attach derived columns so a built table arrives complete; they must be
    # row-local (see FrameTableSpec.derived_cols).
    for name, fn in spec.derived_cols.items():
        t.cols[name] = fn(t)
    return t


def check_representative(t: FrameTable, per_player_target: np.ndarray) -> None:
    """Opt-in representativeness report, per player slot. A per-group metric needs
    each group to hold enough GAMES (not just frames) and enough target spread, or a
    variance-normalized metric inverts; this prints each slot's game count and spread.
    Call it before such metrics — it won't auto-run."""
    print(f"  n_games={t.n_games}  n_frames={t.frame_t.size}")
    for p in range(8):
        sel = t.cols["alive"][:, p]
        col = per_player_target[sel, p]
        ng = int(np.unique(t.game_id[sel]).size)
        print(f"    p{p}: alive={int(sel.sum()):6d}  games={ng:3d}  std={col.std():6.0f}")
