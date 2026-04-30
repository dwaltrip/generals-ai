# generals.io HTTP & WebSocket API

Reverse-engineered notes on the generals.io web client's API surface (everything not specific to the replay format itself — for that, see [`replay-format.md`](./replay-format.md)).

> **Currency:** Verified on **2026-04-30** against the main client bundle `generals-main-prod-v31.4.1-d51b92c0.js`. Endpoint surface is observation-based; semantics for endpoints we haven't called are inferred from name and context.

## HTTP API endpoints

Base: `https://generals.io`. All paths below are relative to that.

### Confirmed working

| Endpoint | Notes |
|---|---|
| `GET /api/serverSettings` | Server config snapshot — sample response below |
| `GET /api/replays?count=N&offset=M` | Recent replays metadata. Both params required, `count` ≤ 200. Documented in [`replay-format.md`](./replay-format.md). |

### Known but not tested

Names pulled from the JS bundle. Semantics are inferred from the name; we haven't actually called these.

| Endpoint | Likely purpose |
|---|---|
| `/api/games/public` | Currently in-progress public games |
| `/api/starsAndRanks?u=<name>` | Player rating / profile data |
| `/api/validateUsername?u=<name>` | Username availability |
| `/api/isSupporter?u=<name>` | Supporter (paid) status |
| `/api/profileModerationData?u=<name>` | Per-user moderation history |
| `/api/event/getEncryptedUsername?u=<name>` | Event-related |
| `/api/maps/lists/{best,hot,new,top}` | Curated map lists |
| `/api/maps/random` | Random map |
| `/api/maps/search?q=<query>` | Map search |
| `/api/map?name=<name>` | Fetch a specific map |
| `/api/createCustomMap`, `/api/maps/upvote`, `/api/mapToFile`, `/api/mapFromFile`, `/api/publishedMapToFile` | Map editor / upload-related |

### Internal / admin

`/api/adminSettings`, `/api/adminSetting`, `/api/moderate`, `/api/moderate/regenerate_user_id`, `/api/ackWarning`. Listed for completeness; not for external use.

## `/api/serverSettings` response

Snapshot from 2026-04-30:

```json
{
  "is_ladder_chat_restricted": false,
  "enabled_ladders": ["2v2", "duel", "ffa"],
  "event_name": null,
  "event_banner_text": null,
  "event_notification_popup_text": null,
  "ffa_player_count": 8,
  "bigteam_player_count": 16,
  "bigteam_team_size": 4,
  "desert_enabled": true,
  "lookout_enabled": true,
  "observatory_enabled": true,
  "tunnel_enabled": true,
  "stronghold_enabled": false,
  "unhidden_modifiers": [0, 1, 2, 3, 4, 5, 7, 6, 8, 10, 9, 11],
  "non_supporter_modifiers": [0, 2, 3, 5, 6, 8, 10, 9, 1, 7, 11]
}
```

Notable:
- The stronghold modifier is **disabled server-side**, even though v18 replays support it in the format.
- Modifier IDs 0–11 exist; we don't yet know what each one means semantically.
- `enabled_ladders` is the source of truth for which ranked queues are live.

## WebSocket endpoints

generals.io uses Socket.IO for live game communication. Two separate clusters:

| URL | Purpose |
|---|---|
| `wss://ws.generals.io` | Human games |
| `wss://botws.generals.io` | Bot games (separate from the human ladder) |

Companion HTTP hosts: `generals.io` and `bot.generals.io` respectively.

Not relevant for replay collection. Listed in case we eventually build a live agent.

## Replay S3 bucket name routing

The web client picks a different replay bucket depending on the hostname it's served from:

| Hostname (web client served from…) | Replay bucket |
|---|---|
| `generals.io`, `ws.generals.io` | `generalsio-replays-na` |
| `bot.generals.io`, `botws.generals.io` | `generalsio-replays-bot` |
| Raw IP address (test deploys) | `generalsio-replays-na` (with API on `:8080`) |
| `localhost` | `generalsio-replays-dev` (served from a local MinIO at `127.0.0.1:9000`) |

The `-na` suffix is suggestive of regional sharding, but we've found no evidence of other regional buckets — NA appears to be the only public human-replay bucket today.

For replay-collector purposes, only `generalsio-replays-na` matters. The `-bot` bucket is for bot-vs-bot games played via the bot API (see WebSocket section).
