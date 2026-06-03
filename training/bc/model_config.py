"""Self-describing spec for the BC model — input encoding + trunk + heads.

`ModelConfig` is the single object that flows config-file → model → checkpoint
→ inference, so any checkpoint carries everything needed to reconstruct both the
model AND the obs encoding that produced its inputs. Two differently-configured
models can therefore coexist in one process (the head-to-head eval sweeps need).

The obs-encoder config nests here as `obs: ObsConfig`; `in_ch` is *derived* from
it (the obs channel count), not a free knob. Nesting keeps the input contract
and the trunk in one self-describing unit, traveling under the single `arch`
checkpoint key — no separate serialization path.

Deliberately **torch-free**: imports only `bc.constants` + `bc.obs_config`
(pure ints / dataclass), so `train_config`, `checkpoint`, and `inference` can
compose `ModelConfig` without dragging the torch-heavy `model.py` into their
import graph at module load. `VALUE_HEAD_VARIANTS` lives here for the same
reason — `model.py` and `train_config.py` both read it without a deferred-import
dance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from bc.constants import H_PADDED, W_PADDED
from bc.obs_config import OBS_CONFIG_DEFAULTS, ObsConfig


VALUE_HEAD_VARIANTS = ("direct", "pyramid")


@dataclass(frozen=True)
class ModelConfig:
    """The BC model's full spec — obs encoding + trunk widths/depths + value head.

    Trunk defaults are the current half-width `128/128/160` trunk: the "0.5×" of
    DeepNash's `256/256/320` (1/4 the params -> theoretically faster compute for
    the proof-of-concept phase while still possibly viable).
    """

    # --- swept: the trunk's design ---
    outer_width: int = 128
    middle_width: int = 128
    inner_width: int = 160
    n_outer: int = 2
    m_middle: int = 2
    m_inner: int = 2
    value_head_variant: str = "direct"
    # --- obs encoding: determines in_ch ---
    obs: ObsConfig = OBS_CONFIG_DEFAULTS
    # --- derived: recorded for self-description, not a free knob ---
    # in_ch is the obs channel count (= obs.obs_channels). -1 is the "derive from
    # obs" sentinel; an explicit value (e.g. from a checkpoint's arch dict) is
    # validated against the derived count, so a channel-count change fails load
    # with a clear arch-mismatch instead of a cryptic state_dict error. Resolved
    # to the real positive count in __post_init__ (never stays -1), so readers
    # can rely on `int`.
    in_ch: int = -1
    H: int = H_PADDED
    W: int = W_PADDED

    def __post_init__(self) -> None:
        # Coerce a dict-valued `obs` (asdict round-trip / config JSON), filling
        # missing keys from the live defaults so partial obs blocks are legal.
        if isinstance(self.obs, dict):
            merged = {**asdict(OBS_CONFIG_DEFAULTS), **self.obs}
            object.__setattr__(self, "obs", ObsConfig(**merged))
        # Derive in_ch from the obs config (or validate an explicit value).
        derived_in_ch = self.obs.obs_channels
        if self.in_ch == -1:
            object.__setattr__(self, "in_ch", derived_in_ch)
        elif self.in_ch != derived_in_ch:
            raise ValueError(
                f"in_ch={self.in_ch} contradicts dense_history_n="
                f"{self.obs.dense_history_n} (→ {derived_in_ch} channels)"
            )
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
        if self.H < 1 or self.W < 1:
            raise ValueError("H/W must be positive")
