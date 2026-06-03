"""Obs-encoder configuration — the static hyperparameters that shape the obs
tensor a model is trained on.

`ObsConfig` is nested inside `ModelConfig` (so it rides into every checkpoint's
`arch` key) and the encoder reads it off `MemoryState`. It carries the knobs
that determine the *input contract* — currently just `dense_history_n`, which
sets the depth of the recent-spatial-history window and therefore the total
channel count (`obs_channel_count`).

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

from dataclasses import dataclass

from bc.constants import obs_channel_count


@dataclass(frozen=True)
class ObsConfig:
    """Static obs-encoder hyperparameters. No inline field defaults — the live
    defaults live in `OBS_CONFIG_DEFAULTS`."""

    # Depth of the dense recent-spatial-history window: `ownership_transition`
    # and `army_delta` are each emitted for the last `n` ticks, adding `2 * n`
    # channels. `0` ablates history entirely.
    dense_history_n: int

    @property
    def obs_channels(self) -> int:
        """Total obs-tensor channel count implied by this config."""
        return obs_channel_count(self.dense_history_n)

    def __post_init__(self) -> None:
        if self.dense_history_n < 0:
            raise ValueError(
                f"dense_history_n must be >= 0; got {self.dense_history_n}"
            )


# Live default policy — the single home for the current default `n`. Referenced
# by `ModelConfig.obs`'s default and by every "I want the defaults" call site.
OBS_CONFIG_DEFAULTS = ObsConfig(dense_history_n=5)

# Default obs channel count (n=5 → 96). Convenience alias for code that builds a
# default-config model/tensor; config-aware sites should read `model.cfg.in_ch`
# (or `obs_cfg.obs_channels`) instead.
OBS_CHANNELS = OBS_CONFIG_DEFAULTS.obs_channels
