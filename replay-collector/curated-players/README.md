# Curated player lists

Top-FFA-player username lists used by the parser corpus driver to
filter replays — only games where at least one curated player
participated enter the per-game intermediate.

## Files in use

Listed in `curated-player-lists.txt` at the repo root, in load order:

1. `2026-04-30-leaderboard-ffa-top-100-combined.txt`
2. `leadeboard-s42-ffawin-elite-gsheets.txt`
3. `leadeboard-s42-ffa-elite-gsheets.txt`
4. `2026-05-10-new-top-players-from-wr-start-bucket-analysis.txt`

Files under `_wip/` are out-of-scope for the driver — staging area for
new candidate names not yet promoted.

## Loading

Each file is read via `utils.player_name_lists.load_players`: one
username per line, edge-whitespace stripped, blank lines skipped, then
filtered through `utils.usernames.filter_valid` (drops
invalid names with a warning).

## Deduping

Files are unioned in load order with insertion-order dedupe: a name is
kept the first time it appears, later occurrences are ignored. The
ordering in `curated-player-lists.txt` therefore controls precedence
for any name that appears in multiple lists.

Current union size: **205 names** (per-file: 132 / 50 / 45 / 107).

The dedupe lives in `utils.player_name_lists.load_union`.
