"""Self-describing architecture spec for the BC model.

`ModelConfig` is the single object that flows config-file → model → checkpoint
→ inference, so any checkpoint carries everything needed to reconstruct the
model that wrote it. Two different-width models can therefore coexist in one
process (the head-to-head eval the width sweep needs).

Deliberately **torch-free**: it imports only `bc.constants` (pure ints), so
`train_config`, `checkpoint`, and `inference` can compose `ModelConfig` without
dragging the torch-heavy `model.py` into their import graph at module load.
`VALUE_HEAD_VARIANTS` lives here for the same reason — `model.py` and
`train_config.py` both read it without a deferred-import dance.
"""

from __future__ import annotations

from dataclasses import dataclass

from bc.constants import H_PADDED, OBS_CHANNELS, W_PADDED


VALUE_HEAD_VARIANTS = ("direct", "pyramid")


@dataclass(frozen=True)
class ModelConfig:
    """The BC model's architecture — trunk widths/depths + value-head variant.

    Field defaults are the current half-width `128/128/160` trunk (the "0.5×"
    of DeepNash's `256/256/320`, picked for fast M1 iteration). `BCModel()` with
    no args builds exactly this, so the no-op fingerprints stay green.
    """

    # --- swept: the trunk's design ---
    outer_width: int = 128
    middle_width: int = 128
    inner_width: int = 160
    n_outer: int = 2
    m_middle: int = 2
    m_inner: int = 2
    value_head_variant: str = "direct"
    # --- pinned: recorded for self-description, not swept ---
    # in_ch is OBS_CHANNELS (= len(CHANNEL_ORDER)); it's downstream of the obs
    # encoder, not a free knob. Recorded so a future channel-count change fails
    # load with a clear arch-mismatch instead of a cryptic state_dict error.
    in_ch: int = OBS_CHANNELS
    H: int = H_PADDED
    W: int = W_PADDED

    def __post_init__(self) -> None:
        # Widths required-even: the bottleneck ResBlock's `C/2` intermediate
        # floors on odd channel counts, a silent asymmetry. (GroupNorm
        # divisibility is handled separately by `_gn`'s fallback ladder.)
        for name, w in (("outer", self.outer_width), ("middle", self.middle_width),
                        ("inner", self.inner_width)):
            if w < 1 or w % 2 != 0:
                raise ValueError(f"{name}_width must be positive and even; got {w}")
        for name, m in (("n_outer", self.n_outer), ("m_middle", self.m_middle),
                        ("m_inner", self.m_inner)):
            if m < 0:
                raise ValueError(f"{name} must be >= 0; got {m}")
        if self.value_head_variant not in VALUE_HEAD_VARIANTS:
            raise ValueError(
                f"value_head_variant must be one of {VALUE_HEAD_VARIANTS}; "
                f"got {self.value_head_variant!r}"
            )
        if self.in_ch < 1 or self.H < 1 or self.W < 1:
            raise ValueError("in_ch/H/W must be positive")
