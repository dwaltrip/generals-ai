# Observation Tensor Design — v1 Working Spec

**Date:** 2026.05.06
**Status:** **Stub.** Captures deltas from session 5.06-1 onwards. The bulk of the v1 obs tensor design lives in `2026-05/5.05-1-observation-tensor-design-part2.md` (Part 2) and `2026-05/5.03-3-observation-tensor-design-part1.md` (Part 1) — those are historical-but-load-bearing pending consolidation here.

**Companion docs:**
- `2026-05/5.05-1-observation-tensor-design-part2.md` — current load-bearing v1 design (long-term memory, complete tensor)
- `2026-05/5.03-3-observation-tensor-design-part1.md` — settled the bulk of channel categories
- `distance-features-design-space.md` — design rationale for the §2.1 channels added this session
- `network-architecture-design.md` — companion architecture doc; references this doc for input channel layout

---

## 1. Scope and how to read this

This doc will eventually be the canonical v1 obs tensor spec. **For now it is a stub** — a delta layer over `5.05-1`, capturing only what's been added or changed since.

When making decisions about the v1 obs tensor:
- This doc takes precedence for any channel definition or design decision it covers.
- For everything else (the bulk of the v1 design — map structure, ownership, army magnitude, broadcast scalars, scoreboard-derived features, dense recent history, contact/capture, fogged cell state) consult `5.05-1` Part 2.

A planned consolidation pass will migrate the channel definitions from `5.05-1` here once the part-1/part-2 history has been reviewed and any changes from that review are folded in.

### 1.1 Status legend (consistent with `network-architecture-design.md`)

- **Locked** — Decision made, unlikely to change without specific reason.
- **Tentative** — Decision made, but might be revisited based on training signal or further analysis.
- **Default (inherited)** — Punted; using the inherited choice without project-specific scrutiny yet.
- **Open / TBD** — Explicitly unresolved.

---

## 2. Deltas from 5.05-1 (this session)

### 2.1 Distance-from-known-generals channels — **Locked for v1**

**Channels added:** 8 (self general + 7 opponent generals)

**Per-channel encoding:** for each general G, the channel value at cell `(r, c)` is the BFS shortest-path distance from G to `(r, c)`, log-scaled via `log(1 + d)`.

**BFS scope:**
- Edges only through cells known to be passable (mountains as deleted nodes)
- Fogged cells treated as opaque (not traversable for the purpose of this BFS — the agent should plan based on what it knows, not optimistic assumptions)
- Cities are passable (cost approximation deferred — uniform cost for v1)

**Sentinel handling:**
- `-1` for unreachable cells (fully separated from G by mountains/fog)
- `-1` for unrevealed opponent generals (whole channel is sentinel until that opponent's general is observed)
- Sentinel applied **after** the log transform — do not feed `-1` through `log(1 + x)`

**Update cadence:** Recomputed every frame in the parser. Cheap (~microseconds per BFS at 25×25). Could be optimized later by gating updates to "when terrain or general-knowledge changes" but not load-bearing.

**Slot ordering:** consistent with the existing per-opponent broadcast/spatial channels — canonical-by-ID at runtime, slot-permutation augmentation at training time (per `5.05-3-session-notes.md` §3.3).

**Why these channels:** the design rationale, including the design space we explored before settling on this minimal v1 set, lives in `distance-features-design-space.md`. The short version: CNNs are bad at computing topological (graph) distances on a grid with obstacles, and this is the diagnosed cause of Strakam's "mountain dead-ends" failure mode (`5.02-5` §7.5). Pre-computing BFS distance and feeding it as input is essentially free at our compute scale and addresses the limitation directly.

**Interaction with channel budget:** these 8 channels are an addition to the 5.05-1 baseline (~78 + 2N channels at v1 spec). At the dense-history default of N=10, the new total is approximately **86 + 2N = 106 channels.** The compute reframe (`compute-considerations.md`) confirms this is well within budget — channel count is not the binding compute lever.

---

## 3. Channels not changed by this session

For all channel categories not listed in §2, consult `5.05-1` Part 2 §4 (the complete observation tensor table). Specifically:

- §A — Map Structure (6 spatial channels)
- §B — Ownership (8 spatial channels)
- §C — Army Magnitude (1 spatial channel)
- §D — Self Broadcast Scalars (3 broadcast channels)
- §E — Per-Opponent Broadcast Scalars (14 broadcast channels)
- §F — Scoreboard-Derived Features (14 broadcast channels)
- §G — Dense Recent Spatial History (2N spatial channels)
- §H — Contact & Capture (14 broadcast channels)
- §I — Fogged Cell State / Long-Term Memory (19 spatial channels)

Note: `5.05-3-session-notes.md` §2 records that the `passable` channel was dropped from §A (derivable from `mountains`). When the consolidation pass happens, that drop should be reflected here.

---

## 4. Open / TBD

Carried forward from `5.05-1` Part 2 §7 unless otherwise noted:

- Dense history window size N (recommendation: N=10–12 to start; tune empirically)
- Newly-visible-tile handling for army deltas (v1: zero-out; documented simplification with clear upgrade path)
- Trimming candidates if budget gets tight (`opp_N_has_seen`, `historically_seen`, reducing N)
- Normalization scheme for broadcast scalars
- Smoothing window k for scoreboard-derived features
- `last_seen_owner` 9-channel one-hot empirical check (deferred per `5.05-3-session-notes.md` §2)
- Whether to add more distance-feature schemes from `distance-features-design-space.md` §7 (block-rep grid, block-region encoding, encoding variants) — held out of v1 in favor of parsimony

---

## 5. Future consolidation

When this doc graduates from stub to canonical v1 spec, it will:

1. Absorb the channel layout table from `5.05-1` Part 2 §4 (with the §2 deltas folded in).
2. Drop the `passable` channel per `5.05-3-session-notes.md` §2.
3. Add the §2.1 distance-from-known-generals channels as a new category.
4. Update channel budget summary and reference points accordingly.
5. Mark `5.05-1` (and `5.03-3`) as "historical — consolidated into observation-tensor-design.md."

Until then, both docs need to be consulted together.
