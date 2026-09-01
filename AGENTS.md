# generals-ai

> **Editing this file:** `AGENTS.md` is the actual file and `CLAUDE.md` is a symlink to it. Edit `AGENTS.md` directly (writing through the symlink doesn't work).

Building an AI bot to play the [generals.io](https://generals.io) strategy game in FFA mode.

Always check [`README.md`](./README.md). It's known to be partly out of date and is being brought back into sync over time, so prefer the docs it links and this file where they conflict, but read it for anything not covered here.

## Code comments and docstrings cleanup

This is the project's addition to the global "Finishing pass", alongside read-clean / plainly-written / well-calibrated. It is an ongoing effort to clean up existing code comments and docstrings, as well as ensure that new ones don't inadvertently inherit the problematic styles (see below).

Existing comments and docstrings are uneven. Much of it was agent-authored without sufficient oversight. Recent comments are generally better. We're in an open-ended "fix as we go" phase: apply the "rewrite-trim-drop" test below to the comments and docstrings on code that you touch.

The rewrite-trim-drop test for questionable prose: (1) rewrite it plainly, (2) trim it to the sentences that earn their place, or (3) drop it entirely (or move it to where it belongs).

A piece of prose is worth keeping only if:

1. It is significant and non-obvious.
2. It is isn't repeating something already sufficiently covered elsewhere.
3. It makes sense in this specific location, moreso than the alternatives.

Working guidelines:

- **The new baseline is very minimal comments or docstrings.** This is an intentional deviation from the much of the current codebase.
- **One authoritative home per fact** (check 3 in practice). For every sentence past the one-line "what", ask: does this fact have (or deserve) an authoritative home elsewhere? If yes, delete the duplicate content.


Note (2026-06-23, extended 2026-07-03): The guidelines above are new. We will likely need to calibrate them over time. One route for that: `scripts/one_offs/dump_prose_semi.py` is a prototype that flags semicolon-chain tells in comments and docstrings (could help with a cleanup sweep).

## Interfaces are provisional

Much of this codebase was agent-authored with inconsistent design oversight, so an established interface often reflects "an LLM picked it and nothing challenged it yet", not a deliberate decision. Two working rules follow:

- **Question the interface before designing around it.** When a design discussion hits an apparent intrinsic constraint or awkward trade-off, check whether it's really an artifact of some existing interface's current shape, and whether an interface change or cleanup would dissolve the problem. Surface that option instead of presenting the constraint as fixed.
- **The verbose-docstring tell.** A verbose docstring that is densely cluttered often marks an interface that had less design oversight than usual. Treat its shape as especially open to change, not load-bearing.

## Sub-projects

*NOTE: This section is quite stale (it was written very early in the project). A brief, modern (2026/8/15) inventory: Most code is carefully factored into packages in `packages/`. Training (`packages/training`) is the most substantial package, with `training.bc` as the headline sub-package. There are 3 older "sub-projects" or packages that are still in the project root: replay-collector, replay-parser, and sim-core.*

### replay-collector

Accumulates replays from top generals.io players, used as training/analysis data for the AI.

Operates at a safe and reasonable rate to avoid placing strain on the generals.io server. The generals community is friendly to external/hobby projects that do this kind of thing, as long as you are respectful.

## Documentation

- [`replay-collector/README.md`](./replay-collector/README.md) — operator guide: workflow, CLI, re-run behavior, module map.
- [`docs/replay-format.md`](./docs/replay-format.md) — `.gior` file format reference (current at v18).
- [`docs/generals-io-api.md`](./docs/generals-io-api.md) — generals.io HTTP + WebSocket API surface.
- [`docs/agent-tooling/content-review/notes.md`](./docs/agent-tooling/content-review/notes.md) — maintainer notes for the content-review agent + skill (blind prose reviewer): design decisions, trial results, open questions. Read before touching that tooling. `docs/agent-tooling/` is the home for notes on in-repo agent tooling generally.

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
