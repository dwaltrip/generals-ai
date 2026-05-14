# Replay Parser — Status & Plan

**Status:** Simulator built and ranking-match-validated on a sampled sweep. Active focus: broader timestep-level validation before per-game intermediate output.
**Last updated:** 2026-05-13
**Purpose:** Lightweight project-management doc for the replay-parser sub-project. Owns the high-level work plan and current state. Decision rationale lives in `replay-parser-design.md`; this doc points at it rather than duplicating it.

## Companion docs

- [`replay-parser-design.md`](./replay-parser-design.md) — locked design decisions + rationale.
- [`replay-format.md`](./replay-format.md) — `.gior` wire format reference (v18).
- [`generals-io-game-mechanics.md`](./generals-io-game-mechanics.md) — rules reference (and its appendix [`game-mechanics-appendix-resolved-ambiguities.md`](./game-mechanics-appendix-resolved-ambiguities.md) for implementation-reference notes on previously-ambiguous mechanics).
- `../research/gior-format/generals-main-prod-v31.4.1-d51b92c0.js` — JS bundle; the implementation reference being ported.

## Where we are

Simulator core is built end-to-end (decode → state → combat → moves → step → `parse_replay`). The ranking-match validator hit 100% on a weekly-bucketed sample (58 buckets, 2900 replays total), clearing the design doc's ranking-match target on the sample. The validator implements two non-obvious rules: version-aware `lbSort` for the v30.9.2 server-side change, and a surrender-bonus `has_kill` rule that isn't in the JS bundle (reverse-engineered from listings + profile kill-counts; write-up in [`2026-05/5.13-4-sim-ranking-mismatch-solved.md`](./2026-05/5.13-4-sim-ranking-mismatch-solved.md)).

Ranking-match is the coarse end-of-game signal — it accepts intermediate per-tile errors that wash out by game end. Those errors matter for training data quality, so broader timestep-level validation comes next, before output-format work.

## Shipped

- Bundle reading + canonical mechanics doc updates.
- `replay-parser/` scaffold: uv workspace, ruff-enforced `_collector/` bridge.
- Wire decode layer: `decode.py`, `types.py`.
- Simulator core: `state.py`, `combat.py`, `moves.py`, `step.py`, `parser.py`.
- Ranking-match validator (`validator.py`): version-aware `lbSort` for the v30.9.2 rule change + surrender-bonus `has_kill` rule. Ranking-match target hit on weekly-bucketed sample. Surrender-bonus rule write-up: [`2026-05/5.13-4-sim-ranking-mismatch-solved.md`](./2026-05/5.13-4-sim-ranking-mismatch-solved.md).
- Sweep + drill-down tooling: `sweep_match_rates.py`, `debug_rankings.py`.

## Up next

- **Timestep-level simulator validation** (active focus). Per-tile / per-timestep checks against an independent reference, beyond the end-of-game ranking signal. Methods TBD — `replay-parser-design.md` §10 enumerates the candidates. Internal-invariant checks are likely the cheapest starting point. Hand-checking a handful of replays against the official replay viewer on generals.io is another low-cost qualitative signal. Node-side bundle diff is the heavier per-tile option, weighed next session.
- **Full-corpus ranking-match sweep** on the ~170k filtered games. ~100 min at current parser speed; closing argument on the sampled 100% result.

## Remaining for v1

- **Per-game intermediate output.** Action streams, per-perspective records, metadata enrichment (rolling rates, placement, elim_timestep), `.npz` writer.
- **Parser output written for the main corpus.**

## v1 done =

- Ranking-match target hit on full corpus.
- Timestep-level validation has reached a level sufficient to trust output for training.
- Per-game intermediate output landed and written for the main corpus.

## Deferred from v1

- Live-game observation capture + WebSocket harness (design doc §10).
- Train/val split policy beyond the tentative spec in `replay-parser-design.md` §9.
- Exact on-disk format for the per-game intermediate (gzipped `.npz` per game is the working default).
- Phase 2 self-play simulator (separate fork-from-strakam effort if it happens).

## Doc debt (low priority, revisit later)

- **Appendix file framing tension.** `game-mechanics-appendix-resolved-ambiguities.md` is currently framed as "Historical reference" but in practice serves as the live implementation-reference for parser developers (bundle line refs, etc.). A future consolidation / restructure pass could rename or reframe. OK as-is for now.

## Historical: previous Phase 1–6 framing

Recent session notes (5.11-3, 5.12-1, 5.13-1, 5.13-2, 5.13-3) reference a numbered Phase 1–6 work plan that this doc previously laid out:

1. JS bundle reading + canonical doc updates.
2. Stand up `replay-parser/` subproject.
3. Wire → typed records.
4. Simulator (NumPy).
5. Ranking-match validator.
6. Action extraction + per-perspective state + intermediate-format writer.

Phases 1–5 are folded into "Shipped" above; Phase 6 work is what "Remaining for v1" covers. The numbering isn't load-bearing — preserved here only for cross-reference with those session notes.
