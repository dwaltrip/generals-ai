"""From the fq table to the tensors the toy trains on.

Operates on `FrameTable` (so it composes with `fq`'s `select`) until the final
`to_tensors`:

  - `prep_target` — add the target-dependent trio (`label`, `margin`, `argmin_set`)
    keyed off the target's membership mask (`alive` vs `army>0`). The surrender
    slice runs the grid once per target, so the trio is recomputed per target and
    the three never disagree (6.18-4 §8 #2).
  - `surrender_filter` — drop frames with any surrendered slot (`~alive & army>0`),
    restoring the clean `alive ≡ army>0` binary. The MVP/anchor population; the
    surrender slice runs unfiltered and slices in the metrics instead.
  - `all_alive` / `mixed` — the `n_alive == 8` / `< 8` populations the MVP grid splits
    on (the surrender grid trains on the full population, `population="full"`).
  - `split_by_game` — partition rows by `game_id` into train/val sub-splits, so no
    game straddles the split. A frame's target is a pure function of `(army, alive)`,
    so adjacent ticks in one game are near-duplicate `(input, label)` pairs; a
    frame-level split would leak them. Hygiene against degenerate memorization,
    not a hard generalization test (val ≈ train is expected for these tiny models).
  - `to_tensors` — the surviving columns → a torch dict the loop reads by role.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from training.analysis.fq.frame_table import FrameTable, select


def prep_target(t: FrameTable, target: str) -> FrameTable:
    """Return a copy of `t` with the target-dependent trio attached.

    `target='alive'` masks the argmin by the alive set (weakest agent still
    playing); `target='army_pos'` masks by `army>0` (weakest player with army still
    on the board). `label`, `margin`, and `argmin_set` all key off the one masked
    army (`np.inf` sentinel off-set) so the tie boundary is shared (6.18-4 §8 #2).
    Shares the input's column arrays — the trio is added to a fresh dict, so the
    input table is left untouched for the next target's prep.
    """
    army = t.cols["army_sim"].astype(np.int64)
    if target == "alive":
        member = t.cols["alive"].astype(bool)
    elif target == "army_pos":
        member = army > 0
    else:
        raise ValueError(f"unknown target {target!r}")

    masked = np.where(member, army, np.inf)               # float; dead/off-set → inf
    label = masked.argmin(1)                              # lowest-index argmin
    srt = np.sort(masked, axis=1)
    margin = srt[:, 1] - srt[:, 0]                        # gap to the second-lowest
    margin[~np.isfinite(margin)] = -1.0                  # <2 in-set members → -1
    argmin_set = masked == masked.min(1, keepdims=True)   # tied minima (off-set never ties)

    cols = dict(t.cols)
    cols["label"] = label
    cols["margin"] = margin
    cols["argmin_set"] = argmin_set
    return FrameTable(cols, t.persp_val_index, t.frame_t, t.game_id, t.n_games)


def surrender_filter(t: FrameTable) -> FrameTable:
    """Drop every frame holding a surrendered slot (`~alive & army>0`). On the
    remaining frames every dead player is genuinely at army 0."""
    surr_slots = ((~t.cols["alive"].astype(bool)) & (t.cols["army_sim"] > 0)).sum(1)
    return select(t, surr_slots == 0)


def all_alive(t: FrameTable) -> FrameTable:
    """The `n_alive == 8` population: no dead to encode, so the encodings collapse
    to one (army-only) and A1 is a clean linear oracle (`W ≈ −I`)."""
    return select(t, t.cols["n_alive"] == 8)


def mixed(t: FrameTable) -> FrameTable:
    """The `n_alive < 8` population: where the dead-player encoding bites."""
    return select(t, t.cols["n_alive"] < 8)


def split_by_game(
    t: FrameTable, val_frac: float, seed: int
) -> tuple[FrameTable, FrameTable]:
    """Partition rows into (train, val) by `game_id` — no game in both. Returns two
    `FrameTable`s; `val_frac` is the fraction of *games* held out."""
    games = np.unique(t.game_id)
    rng = random.Random(seed)
    shuffled = list(games)
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_frac))
    val_games = set(shuffled[:n_val])
    is_val = np.isin(t.game_id, list(val_games))
    return select(t, ~is_val), select(t, is_val)


# Column → torch dtype. army/land/margin are real-valued; alive/captured/argmin_set
# are boolean masks; label/n_alive are integer (label feeds cross_entropy, so int64).
_DTYPES = {
    "army_sim": torch.float32,
    "land_sim": torch.float32,
    "alive": torch.bool,
    "captured": torch.bool,
    "label": torch.int64,
    "margin": torch.float32,
    "n_alive": torch.int64,
    "surr_frame": torch.bool,
    "argmin_set": torch.bool,
}


def to_tensors(t: FrameTable) -> dict[str, torch.Tensor]:
    """The toy columns → a torch dict the loop indexes by role name. `army_sim` /
    `land_sim` are renamed to the role names (`army`, `land`) the model reads;
    everything else keeps its column name."""
    out = {k: torch.as_tensor(t.cols[k], dtype=dt) for k, dt in _DTYPES.items()}
    out["army"] = out.pop("army_sim")
    out["land"] = out.pop("land_sim")
    return out
