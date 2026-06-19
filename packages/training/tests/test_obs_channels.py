"""Contract tests for the gated obs channel-group machinery.

The spec (`bc.obs_config.GATED_CHANNEL_GROUPS`) and the builder registry
(`bc.obs.channels._GATED_GROUP_BUILDERS`) are kept in two modules, joined by
`key`; these tests pin that they stay aligned, that the obs channel count
derives from the names (so they can't drift), and that the player-status
builder's width / shape match its declared names.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import cast

import numpy as np

from training.bc.obs.channels import (
    _GATED_GROUP_BUILDERS,
    GroupBuildCtx,
    _cat_player_status,
)
from training.bc.obs.memory import MemoryState
from training.bc.obs_config import (
    _PLAYER_STATUS_CHANNEL_NAMES,
    GATED_CHANNEL_GROUPS,
    OBS_CONFIG_DEFAULTS,
)
from training.bc.player_status import PlayerStatusCtx


def test_spec_and_builder_registry_aligned() -> None:
    """Every gated group has a builder and vice versa — no orphan either way."""
    spec_keys = {g.key for g in GATED_CHANNEL_GROUPS}
    assert spec_keys == set(_GATED_GROUP_BUILDERS)


def test_obs_channels_count_derives_from_names() -> None:
    on = OBS_CONFIG_DEFAULTS
    off = dataclasses.replace(on, player_status_channels=False)
    assert on.obs_channels == len(on.channel_names)
    assert off.obs_channels == len(off.channel_names)
    # The player-status group is exactly the 14-channel difference.
    assert on.obs_channels - off.obs_channels == len(_PLAYER_STATUS_CHANNEL_NAMES) == 14


def test_player_status_channel_names_format() -> None:
    # 1-indexed opp_N, block-major: all is_present, then all is_alive.
    assert _PLAYER_STATUS_CHANNEL_NAMES == (
        [f"opp_{i}_is_present" for i in range(1, 8)]
        + [f"opp_{i}_is_alive" for i in range(1, 8)]
    )


def test_player_status_builder_width_shape_and_values() -> None:
    # 4 real players (0=perspective, 1-3 opp), 4-7 phantom. opp 1 alive, opp 2
    # surrendered (present, not alive), opp 3 eliminated (neither).
    sc = PlayerStatusCtx(
        is_real=np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool),
        death_by_slot=np.array([99, 99, 5, 8, -1, -1, -1, -1], dtype=np.int64),
        removal_by_slot=np.array([99, 99, 15, 8, -1, -1, -1, -1], dtype=np.int64),
        sentinel=99,
    )
    opp_slots = [1, 2, 3, 4, 5, 6, 7]  # canonical opponents for perspective 0
    ctx = GroupBuildCtx(
        state=cast(MemoryState, SimpleNamespace(player_status=sc)),
        t=10, perspective_slot=0, opp_slots=opp_slots, H=3, W=3,
    )
    planes = _cat_player_status(ctx)

    assert len(planes) == len(_PLAYER_STATUS_CHANNEL_NAMES) == 14
    assert all(p.shape == (3, 3) and p.dtype == np.float32 for p in planes)

    # planes[0:7] = is_present(opp 1..7); planes[7:14] = is_alive(opp 1..7).
    present = [float(p.flat[0]) for p in planes[:7]]
    alive = [float(p.flat[0]) for p in planes[7:]]
    # opp 1 (death 99 >= 10, removal 99 >= 10): alive & present.
    assert present[0] == 1.0 and alive[0] == 1.0
    # opp 2 (death 5 < 10 → not alive; removal 15 >= 10 → present): surrendered.
    assert present[1] == 1.0 and alive[1] == 0.0
    # opp 3 (death 8 < 10, removal 8 < 10): eliminated — neither.
    assert present[2] == 0.0 and alive[2] == 0.0
    # phantom opp 4-7 (is_real False): masked to 0 on both.
    assert present[3:] == [0.0] * 4 and alive[3:] == [0.0] * 4
