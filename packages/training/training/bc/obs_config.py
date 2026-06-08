"""Obs-encoder configuration — the static hyperparameters that shape the obs
tensor a model is trained on.

`ObsConfig` is nested inside `ModelConfig` (so it rides into every checkpoint's
`arch` key) and the encoder reads it off `MemoryState`. It carries the knobs
that determine the *input contract*: the channel count (via `dense_history_n`
→ `obs_channel_count`) and the tensor's element dtype (`obs_dtype`).

Defaults are policy, not structure, so they live here as a named instance
(`OBS_CONFIG_DEFAULTS`) rather than as inline field defaults — symmetric with
`checkpoint.LEGACY_OBS_CFG` (the frozen historical value). The two are equal
today and allowed to diverge the day we re-default `n`: conflating them would
silently re-describe old checkpoints. The class itself carries no default
policy, so construct via `OBS_CONFIG_DEFAULTS` (or explicit fields), not
`ObsConfig()`.

Deliberately torch-free — imports only `bc.constants` (pure ints + the
channel-count formula).
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from bc.constants import obs_channel_count


@dataclass(frozen=True)
class ObsConfig:
    """Static obs-encoder hyperparameters. No inline field defaults — the live
    defaults live in `OBS_CONFIG_DEFAULTS`."""

    # Depth of the dense recent-spatial-history window: `ownership_transition`
    # and `army_delta` are each emitted for the last `n` ticks, adding `2 * n`
    # channels. `0` ablates history entirely.
    dense_history_n: int

    # Element dtype of the assembled obs tensor: "fp32" or "fp16". fp16 halves
    # every host-side byte on the worker→GPU handoff path (the measured n=20
    # throughput ceiling; see docs/2026-06/6.06-7). Under CUDA autocast the conv
    # consumes fp16 directly; the fp32/no-autocast path upcasts at the model
    # boundary (`shared.device.obs_for_model`). String, not a torch.dtype, to
    # keep this module torch-free.
    obs_dtype: str

    @property
    def obs_channels(self) -> int:
        """Total obs-tensor channel count implied by this config."""
        return obs_channel_count(self.dense_history_n)

    @classmethod
    def validate_partial(cls, d: dict) -> list[str]:
        valid = {f.name for f in fields(cls)}
        return [f"unknown ObsConfig field: {k!r}" for k in d if k not in valid]

    def __post_init__(self) -> None:
        if self.dense_history_n < 0:
            raise ValueError(
                f"dense_history_n must be >= 0; got {self.dense_history_n}"
            )
        if self.obs_dtype not in ("fp32", "fp16"):
            raise ValueError(
                f"obs_dtype must be 'fp32' or 'fp16'; got {self.obs_dtype!r}"
            )


# Live default policy — the single home for the current default `n`. Referenced
# by `ModelConfig.obs`'s default and by every "I want the defaults" call site.
OBS_CONFIG_DEFAULTS = ObsConfig(
    dense_history_n=5,
    obs_dtype="fp16",
)

# Default obs channel count (n=5 → 96). Convenience alias for code that builds a
# default-config model/tensor; config-aware sites should read `model.cfg.in_ch`
# (or `obs_cfg.obs_channels`) instead.
OBS_CHANNELS = OBS_CONFIG_DEFAULTS.obs_channels
