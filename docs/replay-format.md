# generals.io Replay Format (`.gior`)

Reference for fetching and parsing replays from generals.io. Reverse-engineered from the live web client.

> **Currency:** Findings below were verified on **2026-04-30** against the main client bundle `generals-main-prod-v31.4.1-d51b92c0.js`. Format version observed in live replays at that time: **v18**. If the format changes, the deserializer in the current bundle is the source of truth.

## How replays are stored and served

- **Binary file (`.gior`):** LZ-compressed array of game data. This is what the server stores and what the client downloads.
- **JSON file (`.gioreplay`):** plain-text equivalent, produced by decoding a `.gior`. Not served by the server — only used locally by parsers.

### Fetch URLs

| What | URL |
|---|---|
| Single replay binary | `https://generalsio-replays-na.s3.amazonaws.com/<replay_id>.gior` |
| Recent replays metadata | `https://generals.io/api/replays?count=N&offset=M` (both params required, `count` ≤ 200) |

The S3 bucket is public and unauthenticated. There is a separate bucket `generalsio-replays-bot` used for bot-vs-bot games on the dev/bot subdomain — ignore it for human-game collection.

### Recent-replays response shape

```json
[
  {
    "type": "1v1",
    "ladder_id": "duel",
    "id": "F-eulqc6q",
    "started": 1777582825424,
    "turns": 4,
    "ranking": [
      {"name": "...", "stars": 52, "kills": 1, "currentName": "..."},
      ...
    ]
  },
  ...
]
```

The `id` is what feeds into the S3 URL. There is no documented filter-by-username param on this endpoint (`u=` is ignored).

### S3 serving behavior

Replay objects on `generalsio-replays-na` are served by Amazon S3 with:

- `Content-Type: application/octet-stream`
- `Last-Modified` header set per object — usable for incremental sync, cache validation, or "did this game just finish?"
- AES256 server-side encryption at rest
- No CORS errors (the browser fetches the bucket directly from the web client)

### Storage characteristics

> **Caveat:** Numbers below are from a single arbitrarily-selected sample game (`vsVin3bP7`, an 8-player FFA, 658 turns, 2325 moves). Update this section once we have more samples and can describe the realistic range across game modes (1v1 duel, 2v2, FFA, big team).

| Metric | Value (sample) |
|---|---|
| Compressed `.gior` | 14.5 KB |
| Decompressed JSON | ~125 KB |
| Compression ratio | ~9× |

## Decoding pipeline

1. Download the `.gior` bytes.
2. Decompress with `lz-string`'s `decompressFromUint8Array` variant. `lz-string` has several incompatible compress/decompress variants — this is one specific one.
3. JSON-parse the result. You get a flat positional array.
4. Map array slots to named fields per the schema below.

### Python: `lzstring` package gotcha

The `lzstring` PyPI package (latest is `1.0.4` as of 2026-04) does **not** ship a `decompressFromUint8Array` method, even though the JavaScript library does. Bridge: pair input bytes big-endian into 16-bit codepoints, build a string, hand to plain `decompress()`:

```python
import json
import lzstring

_lzs = lzstring.LZString()

def decompress_gior(raw: bytes) -> list:
    s = "".join(chr((raw[i*2] << 8) | raw[i*2+1]) for i in range(len(raw) // 2))
    return json.loads(_lzs.decompress(s))
```

This mirrors what the JS variant does internally: each 16-bit char split into two big-endian bytes on compress, paired back up on decompress.

This is what `replay_collector/fetcher.py` does.

## Schema (v18)

The decoded value is a JSON array. Each slot is a fixed field; new versions append slots rather than restructuring existing ones, which is why a 2017 parser still extracts the first 14 fields correctly from a v18 replay.

> **Wire shape vs. post-mapping shape.** The "Type" column below describes the **wire shape** — what comes out of `decompress_gior()` (and what `replay_collector/fetcher.py` returns). Several record-typed slots (`moves`, `afks`, `chat`, `pings`, `generalTrades`) are stored as positional arrays on the wire; the reference JS bundle wraps each record into a named-field object via `.map(...)`. If your code does the same wrapping, those slots will be lists of dicts; if it doesn't, they're lists of arrays.

| Slot | Field | Type | Notes |
|---|---|---|---|
| 0 | `version` | int | `18` at time of writing |
| 1 | `id` | string | matches the URL replay ID |
| 2 | `mapWidth` | int | |
| 3 | `mapHeight` | int | |
| 4 | `usernames` | string[] | one entry per player |
| 5 | `stars` | number[] | player rating at game start |
| 6 | `cities` | int[] | flat tile indices of city tiles |
| 7 | `cityArmies` | int[] | parallel to `cities` — starting armies |
| 8 | `generals` | int[] | parallel to `usernames` — each player's general tile |
| 9 | `mountains` | int[] | flat tile indices |
| 10 | `moves` | array[] | wire: `[index, start, end, is50, turn]` per move. See [Move record](#move-record) for field meanings. |
| 11 | `afks` | array[] | wire: `[index, turn]` per AFK event — `index` is player, `turn` is when they went AFK |
| 12 | `teams` | array\|null | `null` for FFA; populated for team modes |
| 13 | `map` | string\|null | custom-map reference; `null` for random maps (was `map_title` in v7) |
| 14 | `neutrals` | int[] | non-city neutral tiles |
| 15 | `neutralArmies` | int[] | parallel to `neutrals` |
| 16 | `swamps` | int[] | swamp tiles (modifier) |
| 17 | `chat` | array[] | wire: `[text, prefix, playerIndex, turn]` per message. `prefix` is `""` for normal chat; non-empty for team/whisper/system chat. |
| 18 | (playerColors source) | int[]\|null | `playerColors` is derived; `null` → fall back to `[0..N-1]` |
| 19 | `lights` | int[] | |
| 20 | (settings array) | float[] | `[speed, city_density, mountain_density, swamp_density, city_fairness, spawn_fairness, desert_density, lookout_density, observatory_density]` |
| 21 | `modifiers` | int[] | active modifier IDs |
| 22 | `observatories` | int[] | tile indices (modifier) |
| 23 | `lookouts` | int[] | tile indices (modifier) |
| 24 | `deserts` | int[] | tile indices (modifier) |
| 25 | `player_transforms` | object | wire: `{"<playerIdx>": int}` — `JSON.stringify` on a `Uint8Array` produces an object keyed by stringified indices, not an array. Iterate `.values()` rather than indexing. Each value is a 3-bit map-orientation flag (bit 0 = flip-x, bit 1 = flip-y, bit 2 = transpose) — fairness mechanism so each player's general appears in a consistent orientation. |
| 26 | `pings` | array[] | wire: `[player, turn, tileIndex]` per ping event |
| 27 | `generalTrades` | array[] | wire: `[playerIndexA, playerIndexB, turn]` per trade event (v ≥ 16) |
| 28 | `tunnels` | int[] | tile indices (modifier) |
| 29 | `tunnelLimits` | int[] | parallel to `tunnels` |
| 30 | `chessClockTimingsByMove` | array\|null | timing data when chess clock is enabled |
| 31 | `stronghold_density` | float | v ≥ 18 |
| 32 | `stronghold_strength_min` | int | v ≥ 18 |
| 33 | `stronghold_strength_max` | int | v ≥ 18 |
| 34 | `strongholds` | int[] | tile indices (v ≥ 18) |
| 35 | `strongholdStrengths` | int[] | parallel to `strongholds` (v ≥ 18) |
| 36 | `chessClockMoveExecutionTimestamps` | array\|null | optional — may be absent from the array entirely, or present and `null` |

### Tile coordinates

All tile-index fields use **flat row-major** indexing: `idx = row * mapWidth + col`. Valid range is `0` to `mapWidth * mapHeight - 1`.

### Move record

Each move on the wire is a 5-element array `[index, start, end, is50, turn]`:

| Position | Field | Meaning |
|---|---|---|
| 0 | `index` | player index into `usernames` / `generals` / `stars` |
| 1 | `start` | source tile (flat) |
| 2 | `end` | dest tile (flat) — always orthogonally adjacent in valid play |
| 3 | `is50` | `0` or `1` — half-army move? |
| 4 | `turn` | game turn the move was issued on |

The reference JS parser wraps these into `{index, start, end, is50, turn}` objects via `.map(deserializeMove)`. `replay_collector/fetcher.py` returns the wire array form unchanged.

## Version gates

Behaviors that change with `version`:

| Version | Behavior |
|---|---|
| v < 5 | Old turn-priority rules |
| v < 6 | Cities regenerate (changed in v6) |
| v ≥ 5 and < 15 | `old_priority_v2` rules apply |
| v ≥ 9 | Chat field populated |
| v ≥ 13 | Settings array slots 4–8 populated (fairness + modifier densities) — bundle checks `version > 12` |
| v ≥ 16 | General-trade events present |
| v ≥ 18 | Strongholds present |

## What is *not* stored in the replay

- **Game outcome (winner / final placements).** Derived by replaying moves through the game simulator.
- **Visibility / fog of war.** Reconstructed by the client from game state.
- **Per-turn army counts and tile ownership.** Computed by the simulator from the move log + initial state.

To get any of the above, you need a working simulator. The canonical reference implementation is JavaScript (`vzhou842/generals.io-Replay-Utils`); it's stale (2017–2019) but still useful as a starting point.

## Sources and saved artifacts

- Live client: [generals.io](https://generals.io). The `deserialize` function in `generals-main-prod-v31.4.1-d51b92c0.js` is the authoritative current spec.
- Saved locally:
  - `research/gior-format/generals-main-prod-v31.4.1-d51b92c0.js` — main client bundle
  - `research/gior-format/samples/vsVin3bP7.gior` — example binary
  - `research/gior-format/samples/vsVin3bP7.gioreplay` — example decoded JSON
- Legacy reference parser (JavaScript): [vzhou842/generals.io-Replay-Utils](https://github.com/vzhou842/generals.io-Replay-Utils). Authored by Victor Zhou, the original creator of generals.io. Linked from `dev.generals.io` as the canonical replay-processing resource. Victor sold the site in **2018**, and the repo went effectively dormant after that (last meaningful commit 2017-04-05; last commit 2019-04-24). The dev docs still point to it, but no one is keeping it current with the live format. Useful as a starting point for understanding the format and as a sanity-check decoder for the first 14 slots, but not authoritative for current (v18) replays — the live client bundle's `deserialize` function is.
