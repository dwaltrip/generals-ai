"""The input encoder — the status-sufficiency ladder, and the experiment's pivot.

`encode(army, land, alive, captured, cfg)` maps one frame's raw per-player
quantities to the `[B, 8, C]` tensor a scorer reads (batch × player × per-player
feature). It owns the single `/1000` normalization, so the table stays in raw units
(and `margin`'s ε is raw-army). Encoder correctness *is* the experiment: a silent
bug yields a clean-looking wrong headline, since the result is a comparison *across*
these encodings (6.18-1 Risk #2) — hence the load-bearing test in `tests/test_encode.py`.

The encodings, on the full (unfiltered) population — three player statuses now
exist: alive, surrendered-present (`~alive & army>0`), eliminated (`army==0`). The
axis is **status sufficiency**: does the encoding carry what's needed to exclude the
right players (6.18-4 §4)?

  - `none` → `army_norm` as-is: no status channel. Eliminated players sit at exactly
    0 — the value plain `−army` most prefers but must never pick → a **notch** a
    linear model can't draw. Surrendered players (`army>0`) are indistinguishable
    from live, so `none` ceilings on the surrender subpopulation under the `alive`
    target (6.18-4 §5).
  - `capture_status` → `[army_norm, captured]`: prod's capture-based channel (§6),
    collapsed to a binary captured flag. Captured players are already at `army==0`,
    so the flag adds nothing over the army-zero notch; neutralized players (also 0)
    aren't flagged → still a notch; surrendered players aren't flagged → same ceiling
    as `none`. The provable `none ≡ capture_status` identity (6.18-4 §4).
  - `boolean` → `[army_norm, alive]`: the true alive mask as a channel. Dead can be
    pushed *high* by a learned `+BIG·alive` term, so the argmin is **linear in the
    augmented input** — no notch, and surrendered players excluded.
  - `categorical` → `[army_norm, alive, surrendered, eliminated]`: 3-way one-hot
    status (derivable from `(alive, army>0)`). Linearly solvable like `boolean`; the
    extra surrendered/eliminated split is a confirmatory negative for the `alive`
    target (alive-vs-rest already suffices).
  - `neg1_sentinel` → dead army overwritten to −1 (the MVP control, kept re-runnable
    via the `mvp` experiment): dead is linearly separable from live (≥0), but the
    argmin still needs a notch at −1.

`channels` appends a `land_norm` distractor (the `+land` rider). `layout=vector` is
the MVP; `broadcast` (a synthesized spatial dim) is a deferred branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


NORM = 1000.0  # the single normalization constant: raw army/land → normalized units

Encoding = Literal["none", "capture_status", "boolean", "categorical", "neg1_sentinel"]
Channels = Literal["army_only", "army_land"]
Layout = Literal["vector", "broadcast"]

# Per-encoding count of appended status channels (beyond the army channel).
_STATUS_CHANNELS = {
    "none": 0,
    "neg1_sentinel": 0,
    "capture_status": 1,  # the captured flag
    "boolean": 1,         # the alive mask
    "categorical": 3,     # alive / surrendered / eliminated one-hot
}


@dataclass(frozen=True)
class EncoderCfg:
    encoding: Encoding = "none"
    channels: Channels = "army_only"
    layout: Layout = "vector"

    @property
    def n_channels(self) -> int:
        """Per-player feature width `C` the scorer must size to."""
        c = 2 if self.channels == "army_land" else 1
        return c + _STATUS_CHANNELS[self.encoding]


def encode(
    army: torch.Tensor,
    land: torch.Tensor,
    alive: torch.Tensor,
    captured: torch.Tensor,
    cfg: EncoderCfg,
) -> torch.Tensor:
    """`[B, 8]` raw army/land + `[B, 8]` alive/captured masks → `[B, 8, C]` features.

    `alive`/`captured` are binary masks (bool or 0/1). Army/land arrive in raw units;
    this is the one place `/NORM` happens.
    """
    alive_b = alive.bool()
    army_norm = army / NORM

    if cfg.encoding == "neg1_sentinel":
        army_ch = torch.where(alive_b, army_norm, army_norm.new_full((), -1.0))
    else:  # none / capture_status / boolean / categorical all carry raw army
        army_ch = army_norm

    chans = [army_ch]
    if cfg.channels == "army_land":
        chans.append(land / NORM)

    if cfg.encoding == "boolean":
        chans.append(alive_b.to(army.dtype))
    elif cfg.encoding == "capture_status":
        chans.append(captured.to(army.dtype))
    elif cfg.encoding == "categorical":
        army_pos = army > 0
        chans.append(alive_b.to(army.dtype))                       # alive
        chans.append(((~alive_b) & army_pos).to(army.dtype))       # surrendered-present
        chans.append(((~alive_b) & ~army_pos).to(army.dtype))      # eliminated

    x = torch.stack(chans, dim=-1)  # [B, 8, C]

    if cfg.layout == "vector":
        return x
    if cfg.layout == "broadcast":  # pragma: no cover — deferred rider
        raise NotImplementedError("broadcast layout is a deferred rider (6.18-1 §3)")
    raise ValueError(f"unknown layout {cfg.layout!r}")
