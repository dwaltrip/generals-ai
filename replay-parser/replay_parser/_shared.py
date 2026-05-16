"""Shared utilities used by parser scripts.

TODO: There is conceptual overlap between `is_vanilla_ffa` here (a wire-content
filter applied after decoding) and `replay_collector.sql_helpers.ffa_match_filter`
(a SQL pre-filter on listing metadata: ladder_id='ffa' AND player_count 4..8).
They operate at different layers and aren't strict duplicates, but they express
related intent. Worth unifying or at least cross-referencing once the shape of
the parser's downstream consumers settles.
"""


def is_vanilla_ffa(wire: list) -> bool:
    """Return True if the decoded wire array represents a vanilla FFA game —
    no modifiers, custom maps, or game-mode tile types.

    Implements the §4 hard-filter list from docs/replay-parser-design.md.
    Each check rejects a non-vanilla mode/feature; surviving games are
    standard 4–8 player FFA on a procgen map with default tile types.

    Wire indices reflect the v15+ replay format (docs/replay-format.md).
    """
    # wire[12] = teams. Non-null = team mode (2v2, bigteam, etc).
    if wire[12] is not None:
        return False
    # wire[13] = customMap. Non-null = lobby played on a custom (uploaded) map.
    if wire[13] is not None:
        return False
    # wire[16] = swamps. Non-empty = swamp tiles on the map.
    if wire[16]:
        return False
    # wire[21] = modifiers. Non-empty = any modifier enabled (defection, silent
    # war, leapfrog, etc).
    if wire[21]:
        return False
    # wire[22..] are version-gated tile-type fields. Each is a list of tile
    # indices for that special tile type; non-empty = present on the map.
    if len(wire) > 22 and wire[22]: # observatories
        return False
    if len(wire) > 23 and wire[23]: # lookouts
        return False
    if len(wire) > 24 and wire[24]: # deserts
        return False
    if len(wire) > 27 and wire[27]: # generalTrades
        return False
    if len(wire) > 28 and wire[28]: # tunnels
        return False
    # wire[30] = chessClock. Non-null = chess-clock timing rules.
    if len(wire) > 30 and wire[30] is not None:
        return False
    # wire[34] = strongholds. Non-empty = stronghold tiles on the map.
    if len(wire) > 34 and wire[34]:
        return False
    return True
