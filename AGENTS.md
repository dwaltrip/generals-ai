# generals-ai

A project for working on an AI program to play the [generals.io](https://generals.io) strategy game.

## Sub-projects

### replay-collector

Accumulates replays from top generals.io players, used as training/analysis data for the AI.

Operates at a safe and reasonable rate to avoid placing strain on the generals.io server. The generals community is friendly to external/hobby projects that do this kind of thing, as long as you are respectful.

## Documentation

- [`replay-collector/README.md`](./replay-collector/README.md) — operator guide: workflow, CLI, re-run behavior, module map.
- [`docs/replay-format.md`](./docs/replay-format.md) — `.gior` file format reference (current at v18).
- [`docs/generals-io-api.md`](./docs/generals-io-api.md) — generals.io HTTP + WebSocket API surface.

The collector entry point is `replay_collector.runner.collect_many` (see `replay-collector/replay_collector/runner.py`).

## Tools

- [`tools/docs_info.py`](./tools/docs_info.py) — list project docs with mtime + recent git history (commit hash, date, diff size, subject). Useful when judging doc freshness or finding what changed recently. In-doc `Date:` headers are origination dates and don't always reflect last edits; this tool is the authoritative signal. Run `uv run tools/docs_info.py -h` for options.
