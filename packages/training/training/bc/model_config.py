"""Self-describing spec for the BC model — input encoding + trunk + heads.

`ModelConfig` is the single object that flows config-file → model → checkpoint
→ inference, so any checkpoint carries everything needed to reconstruct both the
model AND the obs encoding that produced its inputs. Two differently-configured
models can therefore coexist in one process (the head-to-head eval sweeps need).

The obs-encoder config nests here as `obs: ObsConfig`; `in_ch` is a derived
property (the obs channel count), not a stored field or a free knob. Nesting
keeps the input contract and the trunk in one self-describing unit, traveling
under the single `arch` checkpoint key — no separate serialization path.

Default policy lives in `MODEL_CONFIG_DEFAULTS` (a named instance), not as inline
field defaults — symmetric with `obs_config.OBS_CONFIG_DEFAULTS`. Build a
customized config via `build_model_cfg(**overrides)`, which fills the policy
defaults; the bare class carries only the structural `H`/`W` defaults.

Deliberately **torch-free**: imports only `bc.constants` + `bc.obs_config`
(pure ints / dataclass), so `train_config`, `checkpoint`, and `inference` can
compose `ModelConfig` without dragging the torch-heavy `model.py` into their
import graph at module load. `VALUE_HEAD_VARIANTS` lives here for the same
reason — `model.py` and `train_config.py` both read it without a deferred-import
dance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any

from training.bc.constants import H_PADDED, W_PADDED
from training.bc.obs_config import OBS_CONFIG_DEFAULTS, ObsConfig


VALUE_HEAD_VARIANTS = ("direct", "pyramid")

# Where the value head's channel dropout sits relative to its `pre` module
# (the pyramid variant's embedded U-Net). "post_pre" noises the pre-module's
# *output*; "pre_pre" noises the trunk features *entering* it, so the
# pre-module itself can't co-adapt to exact channel combinations. For the
# `direct` variant (`pre` = Identity) the two are the same computation.
VALUE_HEAD_DROPOUT2D_SITES = ("post_pre", "pre_pre")


@dataclass(frozen=True)
class ModelConfig:
    """The BC model's full spec — obs encoding + trunk widths/depths + value head.

    Policy fields carry no inline defaults — the live defaults live in
    `MODEL_CONFIG_DEFAULTS`; construct via `build_model_cfg` (or explicit fields).
    The default `128/128/160` trunk is the "0.5×" of DeepNash's `256/256/320`
    (1/4 the params -> theoretically faster compute for the proof-of-concept
    phase while still possibly viable).
    """

    # --- swept: the trunk's design ---
    outer_width: int
    middle_width: int
    inner_width: int
    n_outer: int
    m_middle: int
    m_inner: int
    value_head_variant: str
    # --- swept: value-head regularization (train-time only; eval is a no-op,
    # and nn.Dropout has no params, so state_dicts are identical across
    # settings). Insertion points in `ValueHead.forward` / its pre-module:
    #   dropout2d — channel dropout (whole feature maps) on [B, C, H, W]
    #               features; elementwise dropout is weak on conv maps
    #               because spatially-correlated neighbors fill holes in.
    #               `dropout2d_site` picks where it sits relative to `pre`
    #               (see VALUE_HEAD_DROPOUT2D_SITES).
    #   skip_dropout2d — channel dropout on the pyramid pre-module's skip
    #               connections (pyramid variant only). The skips carry the
    #               full-resolution detail route around the 8×8 bottleneck;
    #               taxing them pushes the head onto compressed global
    #               context, the route placement signal should live on.
    #   dropout   — elementwise on the flattened [B, H·W] vector, directly in
    #               front of the Linear most able to do per-game lookups.
    value_head_dropout2d: float
    value_head_dropout: float
    value_head_dropout2d_site: str
    value_head_skip_dropout2d: float
    # --- obs encoding (determines in_ch) ---
    obs: ObsConfig
    # --- structural constants ---
    H: int = H_PADDED
    W: int = W_PADDED

    @property
    def in_ch(self) -> int:
        """The trunk's input channel count = the obs channel count.

        Derived from `obs`, not a dataclass field: it enters the network only at
        the trunk's first conv, so `obs` fully determines it. Checkpoints still
        record it as a checksum (`TrainingState.save`) and validate it on load
        (`checkpoint._arch_for_load`), turning an obs-channel-formula change into
        a clear error rather than a cryptic state_dict shape mismatch.
        """
        return self.obs.obs_channels

    @classmethod
    def validate_partial(cls, d: dict) -> list[str]:
        valid = {f.name for f in fields(cls)}
        # in_ch appears in checkpoint arch dicts; build_model_cfg pops it.
        valid.add("in_ch")
        errors = []
        for key in d:
            if key == "obs":
                if isinstance(d[key], dict):
                    errors.extend(ObsConfig.validate_partial(d[key]))
            elif key not in valid:
                errors.append(f"unknown ModelConfig field: {key!r}")
        return errors

    def __post_init__(self) -> None:
        # Coerce a dict-valued `obs` (asdict round-trip / config JSON), filling
        # missing keys from the live defaults so partial obs blocks are legal.
        if isinstance(self.obs, dict):
            merged = {**asdict(OBS_CONFIG_DEFAULTS), **self.obs}
            object.__setattr__(self, "obs", ObsConfig(**merged))
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
        for name, p in (("value_head_dropout2d", self.value_head_dropout2d),
                        ("value_head_dropout", self.value_head_dropout),
                        ("value_head_skip_dropout2d", self.value_head_skip_dropout2d)):
            if not 0.0 <= p < 1.0:
                raise ValueError(f"{name} must be in [0, 1); got {p}")
        if self.value_head_dropout2d_site not in VALUE_HEAD_DROPOUT2D_SITES:
            raise ValueError(
                f"value_head_dropout2d_site must be one of "
                f"{VALUE_HEAD_DROPOUT2D_SITES}; got {self.value_head_dropout2d_site!r}"
            )
        # A set-but-inert knob would let a sweep config lie about what ran.
        if self.value_head_variant == "direct" and self.value_head_skip_dropout2d > 0:
            raise ValueError(
                "value_head_skip_dropout2d requires the pyramid variant — "
                "the direct head has no pre-module skips to drop"
            )
        if self.H < 1 or self.W < 1:
            raise ValueError("H/W must be positive")


# Live default policy — the single home for the default trunk/obs/value-head
# spec. Referenced by `build_model_cfg` and as `BCModel`'s default arg.
MODEL_CONFIG_DEFAULTS = ModelConfig(
    outer_width=128,
    middle_width=128,
    inner_width=160,
    n_outer=2,
    m_middle=2,
    m_inner=2,
    value_head_variant="direct",
    value_head_dropout2d=0.0,
    value_head_dropout=0.0,
    value_head_dropout2d_site="post_pre",
    value_head_skip_dropout2d=0.0,
    obs=OBS_CONFIG_DEFAULTS,
)


def build_model_cfg(**overrides: Any) -> ModelConfig:
    """Build a `ModelConfig`, filling unset fields from `MODEL_CONFIG_DEFAULTS`.

    The single construction path for partial configs — app code
    (`build_model_cfg(value_head_variant=...)`) and dict-bearing loaders (config
    files, checkpoint arch dicts, resume overlays) alike.
    """
    # in_ch is a derived property, not a field — but checkpoint arch dicts carry
    # it as a recorded checksum, so drop it rather than letting it hit replace().
    # The checksum is validated in checkpoint._arch_for_load, not here.
    overrides.pop("in_ch", None)
    return replace(MODEL_CONFIG_DEFAULTS, **overrides)
