# TODO: I used game-types as the home for this file because it already existed
# and was convenient. It's a dependency leaf-node which is exactly what we want
# for these values.
# If not exactly right, it's probably close enough.

# Sentinel value for cells outside the perspective's vision in a
# `PerspectiveView` (see `bc/obs.py`). Distinct from existing ownership values
# (-2 = mountain, -1 = neutral, 0..7 = player slot). The corresponding armies
# value at a fogged cell is 0 — there's no separate armies sentinel.
# TODO: I think this is basically just "unknown" structures-in-fog.
#   The "distinct from..." comment above makes me think this.
#   For reference, fog can have:
#     1. "passable tiles": no structure, unknown ownership, blank initially
#     2. known structures: mountain or city (player-owned or neutral)
#     3. unknown structure: "structures in fog". either city or mountain
#     4. general: looks equivalent to a regular passable tile prior when fogged.
#   If true, we we should rename it.
OWN_FOG = -3
# sentinel for structures-in-fog proven to be a mountain (by exploration)
OWN_MOUNTAIN = -2
