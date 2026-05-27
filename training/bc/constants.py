"""
Cross-cutting tensor-shape + pipeline constants for the training pipeline.

Shared across the dataloader, model, and augmentation code. Action-encoding
constants (direction enum, sub-channel layout, flat-index layout) live in
`actions.py` — they're owned by the encoder that defines them.

Padding convention: top-left. The unpadded board occupies rows 0..H-1 and
cols 0..W-1 of the padded grid; the right/bottom margins are filler. The
padded re-index in `actions.py` and the channel-assembly code both assume
this convention.

`CHANNEL_ORDER` is the single source of truth for the obs tensor layout.
`obs.py` assembles by stacking in this order; tests + model side read
`OBS_CHANNELS = len(CHANNEL_ORDER)`. When you add or rearrange channels,
this list is the one place to edit — and the assembly code's named-var
stack should be reordered to match.
"""

H_PADDED = 32
W_PADDED = 32

# Drop-filter membership.
ELIGIBLE_PLAYER_COUNT = 8
MAX_BOARD_SIDE = 32  # inclusive; drop games where max(w, h) > MAX_BOARD_SIDE

# Dense-history window size for `ownership_transition[t-k]` + `army_delta[t-k]`.
# v1 spike uses N=5 (vs. design's 10–12) to keep dataloader cost lower. Tune up
# post-spike if the model is starved for tactical-history signal.
DENSE_HISTORY_N = 5

# Sentinel value for cells outside the perspective's vision in a
# `PerspectiveView` (see `bc/obs.py`). Distinct from existing ownership values
# (-2 = mountain, -1 = neutral, 0..7 = player slot). The corresponding armies
# value at a fogged cell is 0 — there's no separate armies sentinel.
OWN_FOG = -3

# BFS city-passability knob (obs.py cat 5). A non-own city is treated as passable
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

# Channel layout for the v1 spike obs tensor (96 channels).
# Grouped by category; comments mark the boundaries. Order here = stack order in
# `bc.obs.build_obs`.
CHANNEL_ORDER = [
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
    # Dense recent spatial history (2 * DENSE_HISTORY_N)
    "ownership_transition_t-1", "ownership_transition_t-2", "ownership_transition_t-3",
    "ownership_transition_t-4", "ownership_transition_t-5",
    "army_delta_t-1", "army_delta_t-2", "army_delta_t-3", "army_delta_t-4", "army_delta_t-5",
]

OBS_CHANNELS = len(CHANNEL_ORDER)  # 96
assert OBS_CHANNELS == 96, f"channel count drifted; CHANNEL_ORDER has {OBS_CHANNELS} entries"

CHANNEL_INDEX = {name: i for i, name in enumerate(CHANNEL_ORDER)}
