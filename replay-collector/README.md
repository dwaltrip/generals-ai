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

### 3. Collect replays

The CLI exposes three subcommands. The two-pass workflow (`sweep-metadata` → `fetch-gior`) is the primary path; `collect-recent` is the original one-shot mode. All commands are **dry-run by default** where applicable.

#### Pass 1: `sweep-metadata` — discover the per-player history depth

Walks every replay listing for each named player and upserts the metadata. No `.gior` fetches; populates `replays` rows with `raw IS NULL` for FFA games.

```sh
uv run python -m replay_collector sweep-metadata data/players.txt --no-dry-run
```

The end-of-run summary prints a per-player Pass 2 budget table (cumulative across all prior runs).

#### Pass 2: `fetch-gior` — download the `.gior` backlog

Selects FFA listings with `raw IS NULL` from the DB and downloads each from S3, round-robin newest-first per player. Capped at `--limit` per run (default 1,000).

```sh
uv run python -m replay_collector fetch-gior --limit 1000        # dry-run
uv run python -m replay_collector fetch-gior --limit 1000 --no-dry-run
```

`--players file.txt` (optional) restricts the work set to replays where at least one listed player is in the ranking, and balances round-robin among those players.

#### `collect-recent` — fetch the most recent N FFA replays per player

Walks each player's listings until `--n-ffa` FFA games have been seen, fetching each one's `.gior` inline. Useful when you want a sliding window of newest-N rather than a full backfill.

```sh
uv run python -m replay_collector collect-recent data/players.txt --n-ffa 200 --no-dry-run
```

## CLI reference

Common to all subcommands:
- `--max-failures K` (default 10) — abort the run after K HTTP failures (run-wide budget).
- Exit code is `1` if the failure budget tripped, else `0`.

`collect-recent`:

| Arg | Default | Notes |
|---|---|---|
| `players_file` | *required* | text file, one username per line |
| `--n-ffa N` | *required* | per-player FFA replay target |
| `--max-listings M` | 1000 | safety cap on listings walked per player |
| `--dry-run` / `--test-logger` / `--no-dry-run` | dry-run | mutually exclusive; `--test-logger` walks one bucket per player and skips `.gior` fetches |

`sweep-metadata`:

| Arg | Default | Notes |
|---|---|---|
| `players_file` | *required* | text file, one username per line |
| `--max-listings-per-player M` | 100,000 | safety rail; sweep stops at this count per player |

`fetch-gior`:

| Arg | Default | Notes |
|---|---|---|
| `--players` | *(none)* | optional player filter (see Pass 2 above) |
| `--limit N` | 1000 | max replays to fetch this run |
| `--dry-run` / `--no-dry-run` | dry-run | mutually exclusive |

## Re-run behavior

The collector is idempotent. Every run starts from the user's most-recent game (offset 0), but:

- Listing rows go through `INSERT OR IGNORE` — already-seen replays are no-ops.
- `.gior` bytes are only fetched when `replays.raw IS NULL`.
- The `--n-ffa` target counts already-cached FFAs walked past, so a re-run keeps a sliding window of "the N most recent FFAs cached."

So re-runs are cheap — usually just listing pages plus any new games played since the last run. **To backfill older games for a player, bump `--n-ffa` higher** (re-running with the same `--n-ffa` will not reach further back).

## Logs

All runs write to `tmp/` and prepend a `# command:` / `# started:` header so you can tell what invocation produced the file.

- `collect-recent` writes two files: `<timestamp>-replay_collector.log` (condensed: per-player intros + streamed bucket-progress dots) and `<timestamp>-replay_collector-verbose.log` (httpx + per-save audit trail). `tail -f` the condensed file for live mid-bucket progress. Use `--test-logger` to exercise the format without hitting S3 or writing replay rows.
- `sweep-metadata` and `fetch-gior` each write a single `<timestamp>-<command>.log` with INFO+ records mirrored to stdout. Per-page (every 10th page) and per-fetch (every 100th fetch) progress lines.

## Module map

| File | Purpose |
|---|---|
| `replay_collector/client.py` | `httpx.Client` wrapper: `TrackedClient` bundles per-host rate limiting + run-wide failure budget; `RateLimiter`, `DEFAULT_RATES` |
| `replay_collector/config.py` | `API_BASE`, `S3_BASE`, and `.env`-backed runtime config |
| `replay_collector/generals_api.py` | Thin endpoint layer: `user_exists`, `iter_user_replays`, `fetch_replay`, `decompress_gior`, `ReplayDecodeError` |
| `replay_collector/runner.py` | `collect-recent` orchestration: `collect_one`, `collect_many`, `UserStats`, `RunStats` |
| `replay_collector/sweep.py` | `sweep-metadata` orchestration: `sweep_one`, `sweep_many`, `SweepStats` |
| `replay_collector/fill.py` | `fetch-gior` orchestration: `fill`, `FillStats` |
| `replay_collector/db.py` | SQLite schema + persistence helpers (WAL mode) |
| `replay_collector/__main__.py` | CLI entry — wires subparsers from `cli/` |
| `replay_collector/cli/` | Per-subcommand argparse + `run` wrappers (`collect_recent.py`, `sweep_metadata.py`, `fetch_gior.py`); `_shared.py` for cross-subcommand helpers |
| `replay_collector/logging_setup.py` | `setup_logging` (collect-recent's two-file split) and `setup_simple_logging` (single-file for sweep/fetch-gior) |
| `scripts/build_player_list.py` | Merge leaderboard JSON dumps into a deduped player list |

## Data

SQLite DB lives at `data/generals.sqlite`. Schema is defined inline in `replay_collector/db.py`:

- `replays` — listing metadata (always populated) plus `.gior` bytes + decoded fields (populated only for ladder=`ffa`)
- `players` — username dictionary
- `replay_players` — junction table with per-position ranking (stars, kills, currentName)

`.gior` format reference: [`../docs/replay-format.md`](../docs/replay-format.md).

## Design notes

- One `httpx.Client` + one `RateLimiter` are shared across an entire multi-user run — the rate budget is per-host, not per-user.
- Per-host rates: 1 req/s on `generals.io` (community-friendly), 2 req/s on the S3 replay bucket (Amazon-hosted archival, no community concern). Set in `client.DEFAULT_RATES`.
- `generals.io` and the S3 replay bucket have independent rate-limit slots, so listing pages and `.gior` fetches interleave during waits.
- The ladder filter (`FULL_DATA_LADDER_ID_FILTER = {"ffa"}` in `runner.py`, hardcoded `'ffa'` in `sweep.py` / `db.py`) decides which games trigger a `.gior` fetch. Listings for non-FFA games walked past are still upserted as metadata.
- The failure budget is shared across all users — a 4xx/5xx burst aborts the whole job, not just the current user.
- `fetch-gior` uses round-robin newest-first per-player ordering: each replay is "owned" by `MIN(p.name)` of its ranking, and the work set is sorted by `(rank-within-owner, owner_name)` so a partial run gives proportional coverage rather than draining one player.
