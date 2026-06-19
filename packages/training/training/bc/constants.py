"""
Cross-cutting tensor-shape + pipeline constants for the training pipeline.

Shared across the dataloader, model, and augmentation code. Action-encoding
constants (direction enum, sub-channel layout, flat-index layout) live in
`actions.py` — they're owned by the encoder that defines them.

Padding convention: top-left. The unpadded board occupies rows 0..H-1 and
cols 0..W-1 of the padded grid; the right/bottom margins are filler. The
padded re-index in `actions.py` and the channel-assembly code both assume
this convention.

This module is purely structural: channel *names* (`_BASE_OBS_CHANNEL_NAMES`)
and the size *formula* (`obs_channel_count`). It holds no default policy — the
live obs-encoder defaults live in `bc.obs_config` (`OBS_CONFIG_DEFAULTS`), which
imports from here. `_BASE_OBS_CHANNEL_NAMES` documents the fixed channels for
human reference; it does NOT drive assembly — the real source of truth for
layout is the stacking order in `bc.obs.build_obs`, guarded by the channel-count
assert there.

The dense-history tail (`ownership_transition[t-k]` + `army_delta[t-k]`,
`2 * dense_history_n` channels) is appended after the base channels and sized by
`ObsConfig.dense_history_n`, so the total count is a function of `n`, not a fixed
constant.
"""

# TODO(config): `CITY_TRAVERSABILITY_FACTOR` is the remaining obs-encoder
# hyperparameter still pinned here; it changes obs *values* (BFS passability),
# not the tensor *shape*, so it's a natural next addition to `ObsConfig`
# (bc/obs_config.py) — its own small pass. Genuine invariants that stay fixed
# (sim/data structure, not knobs): H_PADDED/W_PADDED, ELIGIBLE_PLAYER_COUNT,
# MAX_BOARD_SIDE.

H_PADDED = 32
W_PADDED = 32

# Drop-filter membership.
ELIGIBLE_PLAYER_COUNT = 8
MAX_BOARD_SIDE = 32  # inclusive; drop games where max(w, h) > MAX_BOARD_SIDE

# BFS city-passability knob (obs cat 5). A non-own city is treated as passable
# iff `perspective_total_army > city_army * CITY_TRAVERSABILITY_FACTOR`. Crude
# v1 model of "do I have enough army that capturing this city is feasible" —
# scales naturally with both my strength and the defender's. Replace with
# weighted-edge BFS (Dijkstra) post-spike if the per-cell cost actually matters
# to model quality.
CITY_TRAVERSABILITY_FACTOR = 4

# Per-opp channel ordering contract:
#   For channel index `i ∈ 1..7` in any per-opp group, the opponent referenced is
#   the raw slot at `opp_slots[i-1]`, where `opp_slots = canonical_slot_order(perspective)[1:]`.
#   Channel 0 is always the perspective player ("self").
# This contract is enforced uniformly across `opp_N_owned`, `opp_N_army_count`,
# `last_seen_owner_opp_N`, BFS-to-opp-N, `opp_N_contacted`, `opp_N_captured_by`, etc.

# Fixed (non-history) obs channels — everything before the dense-history tail.
# Grouped by category; comments mark the boundaries. The order mirrors the
# stack order in `bc.obs.build_obs` for readability, but this list is reference
# documentation only — `build_obs` is what actually defines the layout.
_BASE_OBS_CHANNEL_NAMES = [
    # Visibility (1)
    "fog_cells",
    # Visible state (9): self + 7 canonical opponents + log-armies
    "self_owned",
    "opp_1_owned", "opp_2_owned", "opp_3_owned", "opp_4_owned",
    "opp_5_owned", "opp_6_owned", "opp_7_owned",
    "army_magnitude",
    # Persistent map knowledge (4)
    "mountains", "cities", "generals", "structures_in_fog",
    # Memory: last_seen_owner one-hot (9)
    "last_seen_owner_self",
    "last_seen_owner_opp_1", "last_seen_owner_opp_2", "last_seen_owner_opp_3",
    "last_seen_owner_opp_4", "last_seen_owner_opp_5", "last_seen_owner_opp_6",
    "last_seen_owner_opp_7",
    "last_seen_owner_neutral",
    # Memory: scalars (3)
    "last_seen_armies", "turns_since_seen", "historically_seen",
    # Memory: per-opp has-seen (7) — Moore-nbhd of any cell agent observed
    # to be owned by opp N. Derived per 5.05-1 §3.4.2 / §I.
    "opp_1_has_seen", "opp_2_has_seen", "opp_3_has_seen", "opp_4_has_seen",
    "opp_5_has_seen", "opp_6_has_seen", "opp_7_has_seen",
    # BFS distance-from-known-generals (8)
    "bfs_self_general",
    "bfs_opp_1_general", "bfs_opp_2_general", "bfs_opp_3_general",
    "bfs_opp_4_general", "bfs_opp_5_general", "bfs_opp_6_general",
    "bfs_opp_7_general",
    # Self broadcast scalars (3)
    "self_army_count", "self_land_count", "timestep",
    # Per-opp broadcast scalars (14)
    "opp_1_army_count", "opp_2_army_count", "opp_3_army_count", "opp_4_army_count",
    "opp_5_army_count", "opp_6_army_count", "opp_7_army_count",
    "opp_1_land_count", "opp_2_land_count", "opp_3_land_count", "opp_4_land_count",
    "opp_5_land_count", "opp_6_land_count", "opp_7_land_count",
    # Scoreboard-derived broadcasts (14)
    "opp_1_city_inference", "opp_2_city_inference", "opp_3_city_inference",
    "opp_4_city_inference", "opp_5_city_inference", "opp_6_city_inference",
    "opp_7_city_inference",
    "opp_1_land_delta", "opp_2_land_delta", "opp_3_land_delta", "opp_4_land_delta",
    "opp_5_land_delta", "opp_6_land_delta", "opp_7_land_delta",
    # Contact & capture (14)
    "opp_1_contacted", "opp_2_contacted", "opp_3_contacted", "opp_4_contacted",
    "opp_5_contacted", "opp_6_contacted", "opp_7_contacted",
    "opp_1_captured_by", "opp_2_captured_by", "opp_3_captured_by", "opp_4_captured_by",
    "opp_5_captured_by", "opp_6_captured_by", "opp_7_captured_by",
]

_BASE_OBS_CHANNELS = len(_BASE_OBS_CHANNEL_NAMES)  # 86


def dense_history_channel_names(n: int) -> list[str]:
    """The `2 * n` dense-history channel names for a window depth of `n`.

    `ownership_transition[t-k]` then `army_delta[t-k]` for k = 1..n — matching
    the stack order `bc.obs.channels._cat_dense_history` produces.
    """
    return (
        [f"ownership_transition_t-{k}" for k in range(1, n + 1)]
        + [f"army_delta_t-{k}" for k in range(1, n + 1)]
    )


def obs_channel_names(n: int) -> list[str]:
    """Full ordered channel-name list for dense-history depth `n` (base + tail).

    Reference/debugging only — not consumed by the encoder. `build_obs`'s
    assembly order is authoritative; this just names the positions.
    """
    return _BASE_OBS_CHANNEL_NAMES + dense_history_channel_names(n)


def get_obs_channel_indices(dense_history_n: int, names: list[str]) -> list[int]:
    """Indices of the named channels in the obs tensor (dense-history depth
    `dense_history_n`). Raises ValueError naming any channel that doesn't exist."""
    idx = { name: i for i, name in enumerate(obs_channel_names(dense_history_n)) }
    missing = [name for name in names if name not in idx]
    if missing:
        raise ValueError(f"Invalid obs channel(s): {missing}")
    return [idx[n] for n in names]


def obs_channel_count(n: int) -> int:
    """Total obs-tensor channel count for dense-history depth `n`."""
    return _BASE_OBS_CHANNELS + 2 * n
