# generals-ai

> **Editing this file:** `AGENTS.md` is the real file; `CLAUDE.md` is a symlink to it. Edit `AGENTS.md` directly — writing through the `CLAUDE.md` symlink is refused.

Building an AI bot to play the [generals.io](https://generals.io) strategy game in FFA mode.

Always check [`README.md`](./README.md) — it carries current material this file doesn't duplicate (e.g. the frozen-representation probe tooling). It's known to be partly out of date and is being brought back into sync over time, so prefer the docs it links and this file where they conflict, but read it for anything not covered here.

## Code comments and docstrings cleanup

This is the project's addition to the global "Finishing pass", alongside read-clean / plainly-written / well-calibrated. It is an ongoing effort to clean up existing code comments and docstrings, as well as ensure that new ones don't inadvertently inherit the problematic styles (see below).

Existing comments and docstrings are uneven — written fast and agent-authored without much oversight. Two past workflow issues pushed the prose in opposite directions: a semicolon style-contagion made it overly terse and cramped, and a since-corrected memory rule that demanded long docstrings across the training code made it overly verbose. Beyond those, plenty is simply low-value or misplaced. Recent comments are generally better. We're in an open-ended fix-as-we-go phase: apply the keep-or-drop test below to the comments and docstrings on code you touch, rather than trusting them on sight.

The keep-or-drop test: a comment or docstring should exist only if it —

1. makes sense in this specific location,
2. carries its own weight, and
3. isn't repeating something already expressed well enough by the code or nearby docs.

When it passes, write it as clear, legible prose at a length matched to its importance. Otherwise drop it, or move it to where it belongs.

The test is easy to state and hard to apply, because prose gets written — and judged — with the full design context loaded. At write time that context leaks in: each function's docstring picks up a half-relevant slice of it — a consumer mention, a contract paraphrase, a rationale clause. The same facts repeat partially across a subsystem, no copy is authoritative, and every copy drifts. At review time the same context works against you: every sentence reads as relevant, so "is this too verbose?" can't be answered by feel. Three working rules:

- **Keep it concise by default. A sentence must earn its keep.**

  Check 2 in practice: invert the burden of proof. Include a line of prose only if you can clearly articulate why *this* fact deserves mention here, above all the related facts currently in context. The test: what would a reader get wrong, or do differently, without it? That purpose must be specific to the code in question, and it must hold for a fresh reader — a purpose that only makes sense mid-iteration isn't one.

  A missing sentence is recoverable — the reader asks the agent, or reads the code. Vague, verbose prose is not: it distracts and confuses, and it buries the details that actually matter.

- **One authoritative home per fact** (check 3 in practice). For every sentence past the one-line "what", ask: does this fact have (or deserve) an authoritative home elsewhere — the contract at its definition, the consumer relationship at the consumer, the rationale in a doc? If yes, point to it or cut the sentence. A test docstring never paraphrases the format or contract it exercises. Detail that does survive goes next to the exact line it explains, not aggregated up into the docstring.

- **Structure before prose.** If a docstring resists a plain one-liner, suspect the code before writing anything — a hard-to-describe function is often a mis-shaped one (e.g. a test case posing as a shared helper). Flag the structural problem instead of writing prose that accommodates it.

Two common tells:

- **Singling out one of several uniformly-handled things** (check 2): e.g. a docstring noting the function "reads MFU from `summary['mfu']`" when it reads ten `summary` keys the same way. The test: would the same statement be equally true of the siblings it doesn't name? If so it isn't carrying weight — cut it, or, if there's a real pattern, describe the pattern instead of the instance. It slips in because the named item was the salient one when the line was written, and later edits extend the inherited prose instead of re-reading it.
- **A crammed trailing comment** (`# X; default Y`): prose squeezed into too small a slot. Usually it wants to become a full leading comment, or to be dropped.

Soft, cautious carve-out for ML, training, and RL code:

- A brief note on a genuine ML concept (a standard trick, a literature pattern, the reason behind a design choice) can earn its place even when the code itself is clear. This project doubles as an ML-skills-building exercise for its author, so a concept a learner might not know is worth a sentence or two. Lean toward adding one when a topic has been discussed in depth and the author has been asking learning-type questions — those mark the spots where a concise concept note pays off.
- Keep it measured and calibrated to the complexity of the topic and length of the discussion triggering the comment. Importantly, this carve-out is NOT an excuse to let verbose, vague comments stand.


Note (2026-06-23, extended 2026-07-03): The guidelines above are new. We will likely need to calibrate them over time. One route for that: `scripts/one_offs/dump_prose_semi.py` is a prototype that flags semicolon-chain tells in comments and docstrings — a starting point if we later want a detector hook or a cleanup sweep.

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
