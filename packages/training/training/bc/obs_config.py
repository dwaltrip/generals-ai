"""Obs-encoder configuration — the static hyperparameters that shape the obs
tensor a model is trained on.

`ObsConfig` is nested inside `ModelConfig` (so it rides into every checkpoint's
`arch` key) and the encoder reads it off `MemoryState`. It carries the knobs
that determine the *input contract*: the channel count (`dense_history_n` +
the enabled gated groups → `obs_channels`) and the tensor's element dtype
(`obs_dtype`).

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

from collections.abc import Callable
from dataclasses import dataclass, fields

from training.bc.constants import ungated_obs_channel_names


# Per-opponent player-status channels (a gated group). 1-indexed `opp_N` per the
# obs per-opp convention, block-major (all `is_present`, then all `is_alive`).
_PLAYER_STATUS_CHANNEL_NAMES = [
    f"opp_{i}_{field}"
    for field in ("is_present", "is_alive")
    for i in range(1, 8)
]


@dataclass(frozen=True)
class ChannelGroup:
    """A gated obs channel group, appended after the frozen base + dense-history.

    The single source of truth for an optional group: `enabled` reads the config,
    `names` yields its channel names (width = `len(names(cfg))`). The matching
    tensor builder lives in `bc.obs.channels._GATED_GROUP_BUILDERS`, keyed by the
    same `key`; a contract test asserts the two stay aligned. Adding a future
    group is one entry here + one builder + one config field — count/names/build
    all fold over `GATED_CHANNEL_GROUPS`, so they can't drift.
    """

    key: str
    enabled: Callable[[ObsConfig], bool]
    names: Callable[[ObsConfig], list[str]]


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

    # Whether to append the per-opponent player-status group (`is_present` /
    # `is_alive`, 14 channels) — surrender disambiguation. A gated group, so it
    # rides the `GATED_CHANNEL_GROUPS` spec; off for pre-status checkpoints (the
    # `LEGACY_OBS_CFG` pin) so they reconstruct the original obs unchanged.
    player_status_channels: bool

    @property
    def channel_names(self) -> list[str]:
        """The obs tensor's channel names for this config: the frozen base +
        dense-history tail + each enabled gated group's names, appended in spec
        order. The single source of truth for layout and for `obs_channels`."""
        names = ungated_obs_channel_names(self.dense_history_n)
        for group in GATED_CHANNEL_GROUPS:
            if group.enabled(self):
                names = names + group.names(self)
        return names

    @property
    def obs_channels(self) -> int:
        """Total obs-tensor channel count implied by this config — derived from
        `channel_names` (one source, so count and names can't drift)."""
        return len(self.channel_names)

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


# The gated channel groups, in append order (after the frozen base + dense
# history). One entry per optional group; count/names/build all fold over this
# list. Builders are registered by `key` in `bc.obs.channels`.
GATED_CHANNEL_GROUPS: list[ChannelGroup] = [
    ChannelGroup(
        key="player_status",
        enabled=lambda cfg: cfg.player_status_channels,
        names=lambda cfg: _PLAYER_STATUS_CHANNEL_NAMES,
    ),
]


# Live default policy — the single home for the current default `n`. Referenced
# by `ModelConfig.obs`'s default and by every "I want the defaults" call site.
OBS_CONFIG_DEFAULTS = ObsConfig(
    dense_history_n=5,
    obs_dtype="fp16",
    player_status_channels=True,
)

# Obs channel count for the default config (110). Convenience alias for code
# that builds a default-config model/tensor. Config-aware sites should read
# `model.cfg.in_ch` (or `obs_cfg.obs_channels`) instead.
OBS_CHANNELS = OBS_CONFIG_DEFAULTS.obs_channels
