# generals-ai

Building an AI bot to play the [generals.io](https://generals.io) strategy game in FFA mode.

Always check [`README.md`](./README.md) — it carries current material this file doesn't duplicate (e.g. the frozen-representation probe tooling). It's known to be partly out of date and is being brought back into sync over time, so prefer the docs it links and this file where they conflict, but read it for anything not covered here.

## Sub-projects

### replay-collector

Accumulates replays from top generals.io players, used as training/analysis data for the AI.

Operates at a safe and reasonable rate to avoid placing strain on the generals.io server. The generals community is friendly to external/hobby projects that do this kind of thing, as long as you are respectful.

## Documentation

- [`replay-collector/README.md`](./replay-collector/README.md) — operator guide: workflow, CLI, re-run behavior, module map.
- [`docs/replay-format.md`](./docs/replay-format.md) — `.gior` file format reference (current at v18).
- [`docs/generals-io-api.md`](./docs/generals-io-api.md) — generals.io HTTP + WebSocket API surface.

The collector entry point is `replay_collector.runner.collect_many` (see `replay-collector/replay_collector/runner.py`).

## Setup (one-time, after clone)

```sh
./tools/setup-git-hooks.sh
```

Points git at the repo's tracked hooks under `.githooks/`, which includes a pre-commit hook that regenerates `modal_requirements.txt` files whenever `uv.lock` changes.

## Tools

- [`tools/docs_info.py`](./tools/docs_info.py) — list project docs with mtime + recent git history (commit hash, date, diff size, subject). Useful when judging doc freshness or finding what changed recently. In-doc `Date:` headers are origination dates and don't always reflect last edits; this tool is the authoritative signal. Run `./tools/docs_info.py -h` for options.
- [`tools/regen_modal_reqs.sh`](./tools/regen_modal_reqs.sh) — regenerate per-package `modal_requirements.txt` files from `uv.lock`. Run automatically by the pre-commit hook when `uv.lock` is in the commit; can also be invoked manually.
- [`tools/setup-git-hooks.sh`](./tools/setup-git-hooks.sh) — wire up `core.hooksPath` so the repo's `.githooks/` are active. Idempotent; safe to re-run.
