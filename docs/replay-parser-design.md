# Replay Parser Design — v1 Working Spec

**Date:** 2026.05.07

**Status:** **Working draft.** High-level design and key decisions captured; nitty-gritty implementation details deferred. Several early-implementation TODOs flagged in §11 to resolve before/during simulator code.

**Companion docs:**
- `replay-format.md` — `.gior` wire format reference (v18)
- `generals-io-game-mechanics.md` — canonical mechanics reference (drives simulator implementation)
- `observation-tensor-design.md` — obs tensor v1 working spec (channel layout, computed in dataloader, not parser)
- `network-architecture-design.md` — companion architecture doc; action space spec is a parser dependency
- `compute-considerations.md` — sample-volume math + BC compute calibration
- `2026-05/5.02-4-initial-mvp-sketch.md` §2 — parser identified as Phase 1's highest-risk dependency
- `2026-05/5.02-5-strakam-paper-summary.md` / `5.02-6-strakam-paper-takeaways.md` — BC scale calibration anchor
- `replay-collector/README.md` — upstream of this parser; produces the cached `.gior` corpus

---

## 1. Status legend

- **Locked** — decision made, unlikely to change without specific reason.
- **Tentative** — decision made, may be revisited based on data or implementation friction.
- **Open / TBD** — explicitly unresolved.

---

## 2. Goal and scope

- Produces `(observation_tensor, action, metadata)` training data for the BC agent.
- Identified as Phase 1's highest-risk dependency in `5.02-4-initial-mvp-sketch.md` §2 — correctness is cheap to verify, expensive to debug downstream.
- Source data: ~135k cached `.gior` replays in `replay-collector/data/generals.sqlite`. Wire-shape decoding is already done in `replay_collector/generals_api.py:decompress_gior`; this project starts from the wire-shape array.

---

## 3. Output design — C (lean intermediate) — **Locked**

- Parser writes a **compact per-game intermediate**, not materialized observation tensors.
- Channel assembly happens in the **dataloader**, not the parser.
- This decouples the parser from obs-tensor schema → revisions to `observation-tensor-design.md` cause dataloader changes only, not corpus re-parses.
- Per-game contents:
  - **Game-static**: id, version, map dims, mountains, cities + initial cityArmies, neutrals + initial neutralArmies, generals, ranking, modifiers list, `player_transforms`.
  - **Per timestep, raw state**: ownership (H×W int8) + armies (H×W int16), ≈ 3 KB/timestep (at 30×30).
  - **Per perspective**: action stream (supervised targets), per-game metadata (player_id, stars-at-start, placement, elim turn, perspective index).
- Sizing: ~2 MB/game raw, ~160 GB raw / **~50–65 GB gzipped** at 81k filtered replays.
- Storage rationale (three options considered):
  - **Option A** — materialize the full ~106-channel observation tensor per perspective, per timestep, at parse time. The dataloader just reads tensors and feeds them straight to the GPU. ≈ 5 TB at this corpus scale; any obs-tensor revision forces a full re-parse of the corpus.
  - **Option B** — store per-timestep raw game state *plus* per-perspective precomputed fog state (last-seen owner / armies / turns-since-seen, has-seen masks, etc.); the dataloader assembles obs channels on the fly. ≈ 480 GB; supports true random-frame access (any `(game, perspective, timestep)` is one disk read).
  - **Option C (chosen)** — store *only* per-timestep raw game state; the dataloader walks each trajectory forward online, accumulating fog state as it goes. ≈ 50–65 GB gzipped. Random-frame access is approximated via a shuffle buffer + chunk prefetch in the dataloader (§9), preserving i.i.d. minibatch composition. Decouples parser from obs-tensor schema completely — any channel change is a dataloader change only.

---

## 4. Corpus filtering — vanilla FFA only — **Locked (filter rules); Tentative (frequency expectations)**

- Hard filters:
  - `player_count ∈ [4, 8]`
  - `modifiers` slot empty
  - All modifier tile arrays empty (swamps, deserts, tunnels, lookouts, observatories, strongholds) — defensive cross-check against the `modifiers` slot
  - Custom map slot (`map`) null
  - Chess-clock slot (`chessClockTimingsByMove`) null
- A **filter-counts report runs first** against the existing corpus to surface the actual distribution before commitments harden. Want to see the real numbers before locking the filter rules in code.
- Per `compute-considerations.md`: expected ~60% filter survival → ~81k filtered replays.
- See §11 for the open question on `generalTrades` (FFA position-swap mechanic): may add as an additional filter for v1.

---

## 5. Perspective selection — **Tentative**

- **Stars-at-game-start threshold** (slot 5 of the wire format) is the primary filter — game-accurate (rating drifts over time, leaderboard membership is a snapshot), and includes strong opponents in tracked games at no extra collection cost.
- Optional players.txt narrowing as a corpus-level layer (already used by `replay-collector/`).
- Per-game perspective yield: ~1.4 average (~60% of games yield 1 strong-player trajectory, ~40% yield 2; per `compute-considerations.md`).
- Placement is captured as **metadata, not a parse-time filter** (per MVP §2 — placement-based filters should be config-toggleable at training time, not baked into the corpus).
- Stars threshold value: **Open / TBD** — tune once filter-counts report is in.

---

## 6. Frameworks — **Locked**

- **Parser**: NumPy + plain Python. CPU-bound, sequential per game, parallelism via Python multiprocessing. No autograd, no GPU.
- **Trainer**: PyTorch. MPS for local dev iteration, Modal for cloud GPU runs.
- Handoff: `torch.from_numpy(...)` is zero-copy; storage format is NumPy-native.
- JAX deferred — would only matter for a Phase 2 self-play simulator, which would be a separate fork-from-strakam effort, not a port of this parser.

---

## 7. Implementation surface

What the parser must do (v1 scope):

- **Wire-shape decode** — done in `replay_collector`.
- **Wire → typed records**, with version gating (chat ≥ v9, `generalTrades` ≥ v16, strongholds ≥ v18). Wraps positional arrays into named-field records.
- **Map-orientation handling** (`player_transforms` slot 25): per-player flip-x / flip-y / transpose flag the live client uses for fairness. Must be applied per-perspective so each player's general appears in a consistent orientation. High blast radius if missed — easy to forget, produces silently inconsistent training data.
- **Game simulator** (the hard part): per-turn production (general + each owned city), per-round land tick (every 25 turns, +1 on every owned tile), move resolution with the inward-first chain rule, capture mechanics (army halving, territory transfer, captured general → city), surrender countdown for AFK'd players, simultaneous-mutual-capture position swap (if implemented — see §11).
- **Per-perspective state tracking**: raw simulator state is per-game; action targets are per-perspective.
- **Action extraction**: each `[index, start, end, is50, turn]` is the supervised target for the moving player at that timestep; non-moving frames target "pass" (assumes pass is in the action space — see §12).
- **Metadata enrichment** per MVP §2: player_id, rating-at-game-start, placement, elim turn, perspective index, etc.
- **Filter-counts report** (one-shot script, runs before any corpus parse).
- **Validation harness** (see §9).

---

## 8. Sample-volume and training calibration — **inherited from `compute-considerations.md`**

- Per filtered replay: ~530 (state, action) samples (60% × 380 + 40% × 760).
- Total raw at 81k filtered: ~36–50M (state, action) pairs.
- With augmentation (board symmetry ~2–4×, slot permutation ~2–3×) over ~5 epochs: ~1–2B effective sample-views.
- BC compute anchor: Strakam's 3h H100 BC × ~10× sample volume × ~2× per-sample → ~30–60 H100-hours per BC run.

---

## 9. Dataloader pattern (informs the C decision)

- `IterableDataset` + worker-per-process trajectory walking.
- Each worker picks a game file at random, walks t=0..T accumulating fog state online, emits frames.
- Frames flow into a shuffle buffer (~8k–32k) → mini-batches sample from the buffer → effective i.i.d. sampling at the minibatch level.
- Augmentation (symmetry + slot permutation) applied on-the-fly per frame after channel assembly.
- BC default ordering = i.i.d. random shuffling. Strakam BC is the closest scale calibration anchor; their paper is silent on batch composition specifically, so AlphaStar's BC pretraining is the methodological reference for *how* to shuffle.
- Train/val split at **game-id level** (no leakage from same game across splits, per MVP §2 sanity tests).
- Buffer size and chunk-prefetch depth: **Open / TBD** — dataloader-level tuning, not parser-level.

---

## 10. Validation strategy

- **Final-state ownership check**: simulator's end-of-game state matches the replay's `ranking`.
- **Spot checks**: dump rendered timesteps from a few games, compare against the in-browser replay player on generals.io.
- **Wire-slot cross-check**: first 14 slots vs. legacy JS parser (`vzhou842/generals.io-Replay-Utils`) — useful for slot decoding only, not v18 mechanics.
- **Round-trip parser ↔ live decoder**: deferred to bucket 3 (live deployment work). Per MVP §4: dry-run on a recorded game, compare live-decoded observations to parser-decoded observations of the same game. Catches format mismatches before they show up as bad live play.

---

## 11. Early-implementation TODOs (resolve before / during simulator code)

These are concrete unknowns whose resolution is required for simulator correctness. They're empirical questions, not design questions — flagged here so they aren't quietly skipped.

### 11.1 Surrender / AFK countdown duration

- **The unknown:** when a player goes AFK or surrenders, the `afks` array records the moment. But the duration of the countdown before their territory transitions to neutral (and their general/cities become neutral cities) is not in the replay format. `game-mechanics.md` §11 estimates ~10–15 turns; not confirmed.
- **Verification approach** (multiple cross-checks, all empirical):
  1. Pick representative games with at least one AFK / surrender event. Manually inspect when the AFK'd player issued their final move in the replay data.
  2. Cross-reference against the official web-based replay browser on generals.io to see when their territory visually transitions to neutral.
  3. Cross-check via capture events: a surrendered player can only be **captured** before they convert to neutral tiles. Capture events on AFK'd players bound the countdown from below.
- **Per-version verification:** check several games for each `version` value present in the corpus to detect breaking changes across versions.

### 11.2 `moves[].turn` timestep-within-turn disambiguation

- **The unknown:** the wire format stores only `turn` per move, not which of the 2 timesteps within that turn the move belongs to. Each player can move twice per turn. Combat resolution depends on co-timestep ordering (the inward-first chain rule operates within a single timestep), so getting this wrong silently desyncs simulator output from server truth → corrupted BC training data.
- **Working assumption:** array order within `moves` is authoritative — same player's moves with the same `turn` are listed in timestep order.
- **Verification:** read the live JS bundle's `deserialize` / move-replay code in `research/gior-format/generals-main-prod-v31.4.1-d51b92c0.js` to confirm or correct. The cost of verification is low; the cost of getting it wrong is silent corpus corruption.

### 11.3 `generalTrades` (slot 27) — FFA position-swap mechanic

- **What it is:** when two players capture each other's generals on the **same timestep**, instead of resolving via tie-breaker, both players survive and **swap positions** — tiles, generals, cities, armies, all of it. Recent FFA addition (v ≥ 16 per the wire format). Surprising but works well as a game mechanic.
- **v1 decision: Open / TBD.** Two paths:
  - **Implement** the swap mechanic in the simulator.
  - **Filter** games with non-empty `generalTrades` out of the corpus.
- **Decision criteria:** how often does this actually happen in our 81k filtered FFA games (a count is cheap), and how complex is it to implement correctly relative to the rest of the simulator? Likely deferred to the filter approach for v1 unless the implementation is trivial.

---

## 12. Mechanics hot zones (high blast radius if wrong)

- **Move-priority chain rule** (inward-first) — `game-mechanics.md` §6. Must match server resolution exactly.
- **`player_transforms` orientation** — easy to forget; produces silently inconsistent map orientation across training data if missed.
- **Half-turn vs. turn semantics** — DB and wire `turn`/`turns` fields are half-turns (= timesteps = one move per player). No ×2 conversion. Notational landmine.
- See §11 for the unresolved-but-tractable empirical questions.

---

## 13. Hard prerequisites still in flight

- **Obs-tensor consolidation** (`5.06-1` §"Obs-tensor consolidation pass"). C insulates the parser from channel layout, but the dataloader's channel-assembly code is downstream of this consolidation.
- **Action space / output schema** — referenced in MVP §2 as a parser dependency. Specifically affects §7's action-extraction step (whether non-moving frames target "pass" vs. some other no-op encoding). Need to confirm `network-architecture-design.md` settles this before parser code touches action representation.

---

## 14. Deferred / lower-level

- Exact on-disk format for C output (gzipped `.npz` per game suffices for v1; HDF5 / Zarr / webdataset are graduation paths if file-count or per-file ops become a bottleneck).
- Stars-threshold value (tune empirically once filter-counts report lands).
- Shuffle-buffer size and chunk-prefetch depth (dataloader-level tuning).
- Phase 2 self-play simulator (separate effort if Phase 2 happens; likely fork strakam's JAX `game.py`).
