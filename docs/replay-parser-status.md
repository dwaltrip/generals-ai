# Replay Parser — Status & Plan

**Status:** Pre-implementation. Phase 1 (JS bundle reading + canonical doc updates) done; Phase 2 (stand up `replay-parser/` subproject) next.
**Last updated:** 2026-05-11
**Purpose:** Lightweight project-management doc for the replay-parser sub-project. Owns the high-level phase sequencing and current state. Decision rationale lives in `replay-parser-design.md`; this doc points at it rather than duplicating it.

## Companion docs

- [`replay-parser-design.md`](./replay-parser-design.md) — locked design decisions + rationale.
- [`replay-format.md`](./replay-format.md) — `.gior` wire format reference (v18).
- [`generals-io-game-mechanics.md`](./generals-io-game-mechanics.md) — rules reference (and its appendix [`game-mechanics-appendix-resolved-ambiguities.md`](./game-mechanics-appendix-resolved-ambiguities.md) for implementation-reference notes on previously-ambiguous mechanics).
- `../research/gior-format/generals-main-prod-v31.4.1-d51b92c0.js` — JS bundle; the implementation reference being ported.

## Where we are

Phase 1 done as of 2026-05-11. JS bundle traced end-to-end on the replay-mode game class; mechanics-doc ambiguities resolved and folded into the canonical rules (with a side appendix preserving the trace + bundle line refs); design-doc §11 TBDs closed; empirical sanity checks against ~5000 sampled replays confirmed the bundle's claims (modulo one reframe: 1-event AFK records are common, not edge-case). Detailed findings + line refs preserved in `2026-05/5.11-2-bundle-reading.md`.

Next concrete step: Phase 2 — stand up `replay-parser/` as a sibling of `replay-collector/`.

## Phases

1. **JS bundle reading session.** Resolve `game-mechanics.md` §11 and `replay-parser-design.md` §11. Output: spec updates to those docs sharp enough to port from. *Done (2026-05-11).*
2. **Stand up `replay-parser/` subproject** as a sibling to `replay-collector/`. *Not started.*
3. **Wire → typed records.** Pure decode layer; no simulation logic. *Not started.*
4. **Simulator (NumPy).** Direct port of the JS bundle's replay-mode class. *Not started.*
5. **Ranking-match validator.** Built alongside (4); the v1 quality gate. *Not started.*
6. **Action extraction + per-perspective state + C-format writer.** Begin once (5) clears the target. *Not started.*

**v1 done** = phases 1–6 complete, ranking-match target hit, C-format intermediate written for the filtered corpus.

## Quality gate

Ranking-match accuracy **≥ 99.9%** (≤ ~170 mismatches across ~170k filtered games); stretch ≥ 99.99%. Compares the simulator's deduced final ranking against the listings-API ranking stored in `replay_players`.

## Validation strategy

- **Primary signal:** placement-outcome ranking-match per replay. Server-truth, bundle-independent.
- **Implementation reference, not ground truth:** the JS bundle's replay-mode class. The bundle is the replay-viewer's reconstruction, not the server's authoritative logic, and has had known bugs historically.
- **Deferred unless ranking-match plateaus below target:** live-game observation capture (per-timestep server-truth via WebSocket; gold-standard but a real chunk of work to build).
- **Subtle per-tile errors:** acceptable for BC v1. Live ladder eval is the real generalization test for the trained bot.

## Subproject layout

`replay-parser/` as a sibling of `replay-collector/`. NumPy-heavy; imports the wire decoder from `replay_collector`. The filter-counts report stays in `replay-collector/scripts/` (read-only DB analysis fits there naturally).

## Deferred from v1

- Live-game observation capture + WebSocket harness.
- Node-side bundle-diff validation harness (`replay-parser-design.md` §10).
- Train/val split policy beyond the tentative spec in `replay-parser-design.md` §9.
- Exact on-disk format for C-output (gzipped `.npz` per game is the working default).
- Phase 2 self-play simulator (separate fork-from-strakam effort if it happens).
