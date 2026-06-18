"""The input encoder — the 6.18-1 §3 sweep surface, and the experiment's pivot.

`encode(army, land, alive, cfg)` maps one frame's raw per-player quantities to the
`[B, 8, C]` tensor a scorer reads (batch × player × per-player feature). It owns the
single `/1000` normalization, so the table stays in raw units (and `margin`'s ε is
raw-army). Encoder correctness *is* the experiment: a silent bug yields a clean-
looking wrong headline, since the result is a comparison *across* these encodings
(6.18-1 Risk #2) — hence the load-bearing test in `tests/test_encode.py`.

The three encodings, on the filtered binary population (every dead player at army 0):

  - `explicit_mask` → `[army_norm, alive]`: dead can be pushed *high* by a learned
    `+BIG·alive` term, so the argmin is **linear in the augmented input**.
  - `zero_overload` → `army_norm` as-is: dead sit at exactly 0, the value plain
    `−army` most prefers but must never pick → a **non-monotonic notch** a linear
    model can't draw. `gap_scale` rescales the army channel (the rescaled-gap
    control; default 1.0 is a no-op).
  - `neg1_sentinel` → dead army overwritten to −1: still a notch, but dead is now
    linearly separable from all live (≥0).

`channels` appends a `land_norm` distractor (the `+land` rider). `layout=vector` is
the MVP; `broadcast` (a synthesized spatial dim) is a deferred branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


NORM = 1000.0  # the single normalization constant: raw army/land → normalized units

Encoding = Literal["explicit_mask", "zero_overload", "neg1_sentinel"]
Channels = Literal["army_only", "army_land"]
Layout = Literal["vector", "broadcast"]


@dataclass(frozen=True)
class EncoderCfg:
    encoding: Encoding = "zero_overload"
    channels: Channels = "army_only"
    layout: Layout = "vector"
    gap_scale: float = 1.0  # zero_overload only; rescales the army channel

    @property
    def n_channels(self) -> int:
        """Per-player feature width `C` the scorer must size to."""
        c = 2 if self.channels == "army_land" else 1
        if self.encoding == "explicit_mask":
            c += 1  # the alive channel
        return c


def encode(
    army: torch.Tensor, land: torch.Tensor, alive: torch.Tensor, cfg: EncoderCfg
) -> torch.Tensor:
    """`[B, 8]` raw army/land + `[B, 8]` alive → `[B, 8, C]` per-player features.

    `alive` is the binary mask (bool or 0/1). Army/land arrive in raw units; this
    is the one place `/NORM` happens.
    """
    alive_f = alive.to(army.dtype)
    army_norm = army / NORM

    if cfg.encoding == "zero_overload":
        army_ch = army_norm * cfg.gap_scale  # dead already at 0 (post-filter)
    elif cfg.encoding == "neg1_sentinel":
        army_ch = torch.where(alive.bool(), army_norm, army_norm.new_full((), -1.0))
    elif cfg.encoding == "explicit_mask":
        army_ch = army_norm  # dead at 0; exclusion comes from the alive channel
    else:  # pragma: no cover — exhaustive
        raise ValueError(f"unknown encoding {cfg.encoding!r}")

    chans = [army_ch]
    if cfg.channels == "army_land":
        chans.append(land / NORM)
    if cfg.encoding == "explicit_mask":
        chans.append(alive_f)

    x = torch.stack(chans, dim=-1)  # [B, 8, C]

    if cfg.layout == "vector":
        return x
    if cfg.layout == "broadcast":  # pragma: no cover — deferred rider
        raise NotImplementedError("broadcast layout is a deferred rider (6.18-1 §3)")
    raise ValueError(f"unknown layout {cfg.layout!r}")
