"""From the fq table to the tensors the toy trains on.

Four steps, all operating on `FrameTable` (so they compose with `fq`'s `select`)
until the final `to_tensors`:

  - `surrender_filter` — drop frames with any surrendered slot (`~alive & army>0`),
    restoring the clean `alive ≡ army>0` binary the encoding ladder rests on
    (6.18-1 §0, Risk #7). The MVP population.
  - `all_alive` / `mixed` — the two grid populations (`n_alive == 8` vs `< 8`).
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


# Column → torch dtype. army/land/margin are real-valued; alive/argmin_set are
# boolean masks; label/n_alive are integer (label feeds cross_entropy, so int64).
_DTYPES = {
    "army_sim": torch.float32,
    "land_sim": torch.float32,
    "alive": torch.bool,
    "label": torch.int64,
    "margin": torch.float32,
    "n_alive": torch.int64,
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
