"""Visual constants for the timestep viewer. Palette matches the official
generals.io replay viewer (v31.4.1) so side-by-side comparison is direct.

Sourced from colors.txt (user-provided) and the leaderboard / map screenshots
in this dir. Colors are kept as rgb() strings to drop straight into CSS.
"""

# Slot colors used by the official replay viewer. The first 8 are the FFA
# defaults — at 4-8 players, slot i takes colors[i]. The remaining 8 are
# alternates the viewer cycles through for team modes / non-FFA lobbies;
# the parser only handles vanilla FFA, but we keep the full list for
# completeness so the viewer never silently falls back to a non-canonical
# color if a corpus replay turns out to have >8 players.
SLOT_COLORS: list[str] = [
    "rgb(255, 0, 0)",      # 0 — red
    "rgb(39, 146, 255)",   # 1 — blue
    "rgb(0, 128, 0)",      # 2 — green
    "rgb(0, 128, 128)",    # 3 — teal
    "rgb(250, 140, 1)",    # 4 — orange
    "rgb(240, 50, 230)",   # 5 — magenta
    "rgb(128, 0, 128)",    # 6 — purple
    "rgb(155, 1, 1)",      # 7 — dark red
    "rgb(179, 172, 50)",
    "rgb(154, 94, 36)",
    "rgb(16, 49, 255)",
    "rgb(89, 76, 165)",
    "rgb(133, 169, 28)",
    "rgb(255, 102, 104)",
    "rgb(180, 127, 202)",
    "rgb(180, 153, 113)",
]

# Tile background / styling. Approximated from replay-viewer-map-screenshot.png;
# tune against the live viewer during Step 4 iteration.
NEUTRAL_TILE_BG = "rgb(255, 255, 255)"      # empty owned-by-nobody land
MOUNTAIN_TILE_BG = "rgb(200, 200, 200)"     # tile background under the mountain SVG
NEUTRAL_CITY_BG = "rgb(130, 130, 130)"      # tile background for an uncaptured neutral city
TILE_BORDER = "rgb(0, 0, 0)"                # 1px gridline between tiles (rendered via .board gap)

# Leaderboard / event-log treatment, approximated from the screenshots.
DEAD_PLAYER_BG = "rgb(80, 80, 80)"       # heavily-desaturated row for AFK / no-tiles
EVENT_LOG_BG = "rgb(20, 20, 20)"
