# replay-collector

Pulls top players' recent FFA replays from generals.io and stores them in a local SQLite DB. Used to build a training/analysis corpus for the AI agent.

**Corpus snapshot (2026-05-01):** 11.8k replay listings, 6.7k of which have full `.gior` data (FFA only), pulled across the top ~30 players from the `ffa` and `ffawin` ladders (41 players combined after dedup).

## Workflow

A run has three steps. The first two are usually one-shot; the third is what you re-run.

### 1. Get a leaderboard dump

generals.io serves the leaderboard over a websocket — there is no HTTP endpoint for it. Open the in-browser client, watch the websocket frames, and save the JSON payload to `data/<date>-<ladder>-leaderboard.json`. Two examples are checked in.

Expected shape: a one-element list whose entry has `users`, `stars`, and `supporters` arrays (200 entries each, best-first). See `data/2026-04-30-ffa-leaderboard.json`.

### 2. Build a player list

```sh
uv run python scripts/build_player_list.py \
    data/2026-04-30-ffa-leaderboard.json \
    data/2026-04-30-ffa-win-leaderboard.json \
    --top 30 --output data/players.txt
```

Takes the top N from each input file, dedupes (preserving first appearance across files), writes one username per line.

### 3. Run the collector

The CLI is **dry-run by default** — it prints estimates without making any HTTP calls:

```sh
uv run python -m replay_collector data/players.txt --n-ffa 200
```

You get a worst/best-case table (FFA share 25–100%) for listings walked, API calls, S3 fetches, and est. wall time. Pass `--no-dry-run` to actually execute:

```sh
uv run python -m replay_collector data/players.txt --n-ffa 200 --no-dry-run
```

## CLI reference

| Arg | Default | Notes |
|---|---|---|
| `players_file` | *required* | text file, one username per line |
| `--n-ffa N` | *required* | per-player FFA replay target |
| `--max-listings M` | 1000 | safety cap on listings walked per player |
| `--max-failures K` | 10 | abort the run after K HTTP failures (run-wide budget) |
| `--dry-run` | default | print estimates without making API calls |
| `--test-logger` | off | walk one bucket per player, skip `.gior` fetches; for testing log output |
| `--no-dry-run` | off | execute the run for real |

`--dry-run`, `--test-logger`, and `--no-dry-run` are mutually exclusive.

Exit code is `1` if the failure budget tripped, else `0`.

## Re-run behavior

The collector is idempotent. Every run starts from the user's most-recent game (offset 0), but:

- Listing rows go through `INSERT OR IGNORE` — already-seen replays are no-ops.
- `.gior` bytes are only fetched when `replays.raw IS NULL`.
- The `--n-ffa` target counts already-cached FFAs walked past, so a re-run keeps a sliding window of "the N most recent FFAs cached."

So re-runs are cheap — usually just listing pages plus any new games played since the last run. **To backfill older games for a player, bump `--n-ffa` higher** (re-running with the same `--n-ffa` will not reach further back).

## Logs

Each run writes two files under `tmp/`:

- `<timestamp>-replay_collector.log` — condensed: per-player intro/summary lines and per-bucket progress with streamed dots. `tail -f` shows mid-bucket progress live.
- `<timestamp>-replay_collector-verbose.log` — full audit trail: same content plus httpx requests and per-replay save records.

Use `--test-logger` to exercise the format without hitting S3 or writing replay rows.

## Module map

| File | Purpose |
|---|---|
| `replay_collector/client.py` | `httpx.Client` wrapper: `TrackedClient` bundles per-host rate limiting + run-wide failure budget; `RateLimiter` |
| `replay_collector/generals_api.py` | Thin endpoint layer: `user_exists`, `iter_user_replays`, `fetch_replay`, `decompress_gior` |
| `replay_collector/runner.py` | Orchestration: `collect_one`, `collect_many`, `UserStats`, `RunStats` |
| `replay_collector/db.py` | SQLite schema + persistence helpers |
| `replay_collector/__main__.py` | CLI (dry-run estimator + entry into `collect_many`) |
| `replay_collector/logging_setup.py` | Two-file logging + streaming bucket-progress writer |
| `scripts/build_player_list.py` | Merge leaderboard JSON dumps into a deduped player list |

## Data

SQLite DB lives at `data/generals.sqlite`. Schema is defined inline in `replay_collector/db.py`:

- `replays` — listing metadata (always populated) plus `.gior` bytes + decoded fields (populated only for ladder=`ffa`)
- `players` — username dictionary
- `replay_players` — junction table with per-position ranking (stars, kills, currentName)

`.gior` format reference: [`../docs/replay-format.md`](../docs/replay-format.md).

## Design notes

- One `httpx.Client` + one `RateLimiter` are shared across an entire multi-user run — the 1 req/s budget is per-host, not per-user.
- `generals.io` and the S3 replay bucket have independent rate-limit slots, so listing pages and `.gior` fetches interleave during waits.
- The ladder filter (`FULL_DATA_LADDER_ID_FILTER = {"ffa"}` in `runner.py`) decides which games trigger a `.gior` fetch. Listings for non-FFA games walked past are still upserted as metadata.
- The failure budget is shared across all users — a 4xx/5xx burst aborts the whole job, not just the current user.
