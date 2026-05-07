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
- Source data: ~140–145k cached `.gior` replays (~135k 8-player + ~5–10k 4–7 player) in `replay-collector/data/generals.sqlite`. Wire-shape decoding is already done in `replay_collector/generals_api.py:decompress_gior`; this project starts from the wire-shape array.

---

## 3. Output design — C (lean intermediate) — **Locked**

- Parser writes a **compact per-game intermediate**, not materialized observation tensors.
- Channel assembly happens in the **dataloader**, not the parser.
- This decouples the parser from obs-tensor schema → revisions to `observation-tensor-design.md` cause dataloader changes only, not corpus re-parses.
- Per-game contents:
  - **Game-static**: id, version, map dims, mountains, cities + initial cityArmies, neutrals + initial neutralArmies, generals, ranking, modifiers list, `player_transforms`.
  - **Per timestep, raw state**: ownership (H×W int8) + armies (H×W int16 — sufficient; single-tile stacks rarely exceed 32k in serious play, and games where they do are non-competitive and filtered out by the strong-player skill cut) + cities mask (H×W, bitpacked), ≈ 3 KB/timestep (at 30×30). Cities mask is dynamic because eliminated generals become cities; storing per-timestep is simpler than reconstructing from capture events in the dataloader, with negligible storage cost (cities are well under mountain density of ~20%).
  - **Per perspective**: action stream (supervised targets), per-game metadata (player_id, stars-at-start, placement, elim turn, perspective index).
- Sizing: ~2 MB/game raw, ~160 GB raw / **~50–65 GB gzipped** at 81k filtered replays.
- Storage rationale (three options considered):
  - **Option A** — materialize the full ~106-channel observation tensor per perspective, per timestep, at parse time. The dataloader just reads tensors and feeds them straight to the GPU. ≈ 5 TB at this corpus scale; any obs-tensor revision forces a full re-parse of the corpus.
  - **Option B** — store per-timestep raw game state *plus* per-perspective precomputed fog state (last-seen owner / armies / turns-since-seen, has-seen masks, etc.); the dataloader assembles obs channels on the fly. ≈ 480 GB; supports true random-frame access (any `(game, perspective, timestep)` is one disk read).
  - **Option C (chosen)** — store *only* per-timestep raw game state; the dataloader walks each trajectory forward online, accumulating fog state as it goes. ≈ 50–65 GB gzipped. Random-frame access is approximated via a shuffle buffer + chunk prefetch in the dataloader (§9), preserving approximate i.i.d. minibatch composition *given a sufficiently large shuffle buffer* (§9 specifies the floor — load-bearing for the methodology argument). Decouples parser from obs-tensor schema completely — any channel change is a dataloader change only.

---

## 4. Corpus filtering — vanilla FFA only — **Locked (filter rules); Tentative (frequency expectations)**

- Hard filters:
  - `player_count ∈ [4, 8]`
  - `teams` slot null (FFA only — team modes overlap on player_count and would otherwise sneak in)
  - `version ≥ 15` (avoids pre-v15 mechanics changes — `old_priority_v2`, city-regen rules. v15 is the largest single version in the current corpus and ~80–85% of replays are v15+. Confirms via filter-counts report.)
  - `modifiers` slot empty
  - All modifier tile arrays empty (swamps, deserts, tunnels, lookouts, observatories, strongholds) — defensive cross-check against the `modifiers` slot
  - Custom map slot (`map`) null
  - Chess-clock slot (`chessClockTimingsByMove`) null
- A **filter-counts report runs first** against the existing corpus to surface the actual distribution before commitments harden. Want to see the real numbers before locking the filter rules in code.
- Per `compute-considerations.md`: expected ~60% filter survival → ~81k filtered replays.
- See §11 for the open question on `generalTrades` (FFA position-swap mechanic): may add as an additional filter for v1.

---

## 5. Perspective selection — **Tentative** (most likely section to evolve before first BC training run)

- **Primary filter — curated players.txt** (the leaderboard-tracked tier-1 list, already built and used to drive `replay-collector/`). Trusted-source signal; sidesteps the seasonal-volatility issues of stars-based filtering.
- **Secondary filter — rolling-window win-rate at game-time:** for each emitted perspective, compute the player's 1st-place rate over their most recent ~100 games as of that game's date; exclude perspectives where the rate dipped below threshold (initial baseline ~25% for 8-player FFA — random is 12.5% — but tunable via filter-counts report). Skip perspectives where the player has fewer than ~50 prior games at game-time (unstable signal). Source data: placement is already in `replay_players` for every cached game.
  - Top-3 rate may be added as a complementary metric if 1st-rate alone is too noisy at low game counts.
- **Stars-at-game-start** (slot 5): captured as **metadata only**, *not* used as a parse-time filter. Stars are season-volatile (10-week seasons reset the distribution; early-season everyone is low, late-season stratification is sharp), so a fixed threshold is unreliable. Seasonal-relative implementation is deferred — the curated list + win-rate filter cover the main quality concerns.
- **Placement at end of game:** captured as **metadata only**, not a parse-time filter (per MVP §2 — placement-based filters should be config-toggleable at training time, not baked into the corpus).
- **Per-game perspective yield:** the ~1.4 average from `compute-considerations.md` was computed under the prior stars-threshold plan. Under curated-list-only, yield depends on how often two curated players collide in the same game — likely similar but worth re-measuring via the filter-counts report. Sample-volume in §8 may need light revision.
- **Forward pointer — corpus enhancement work:** the curated list is expected to grow via win-rate-based discovery from the cached corpus — find candidate strong players in existing games (lax stars pre-filter is fine since the real test is win-rate), fetch their listing histories (metadata-only via the existing collector, cheap), promote those whose recent win-rate clears threshold. Bonus: discovery surfaces new strong players we can extend `.gior` collection to. Iterative — converges in 2–3 rounds. Full spec deferred to a separate corpus-quality / filtering doc when that pipeline gets built.

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
- **Action extraction**: each `[index, start, end, is50, turn]` is the supervised target for the moving player at that timestep; non-moving frames target "pass". Action space is settled in `network-architecture-design.md` §4: per-cell `[pass + 4 directions × 2 splits]`.
  - **Pass-frame rate** (napkin-math from user experience; verify empirically once parser produces samples): ~30–45% corpus-wide. Decomposes into:
    - ~20% **open-turtle** games — elite-player-specific behavior: deliberate early-game concession when ganged up on by 2–3 attackers, often gathering small armies onto the general while otherwise idling. Distinct from the generic turtling weaker players do. Pass-rate within these games: ~50–80%.
    - ~80% **normal** games — pass-rate < 30%, often much lower.
  - 30–45% sits comfortably inside vanilla-cross-entropy comfort range; no class-weighting expected at v1. Revisit only if empirical rate exceeds ~80%.
- **Metadata enrichment** per MVP §2: player_id, rating-at-game-start, placement, elim turn, perspective index, etc.
- **Filter-counts report** (one-shot script, runs before any corpus parse). Reports: per-filter survival counts (how many drop to each rule in §4), version-distribution histogram (confirms ≥ v15 cutoff), `generalTrades` frequency in surviving games (informs §11.3), per-curated-player game counts + rolling win-rate distributions (informs §5 threshold tuning), and per-game perspective yield under the curated-list filter (confirms or revises §8 sample-volume estimates).
- **Robustness**: skip-and-log on decode failures (`ReplayDecodeError` from `replay_collector/generals_api.py:decompress_gior`); a malformed `.gior` should never abort a corpus parse.
- **Validation harness** (see §10).

---

## 8. Sample-volume and training calibration — **inherited from `compute-considerations.md`**

- Per filtered replay: ~530 (state, action) samples (60% × 380 + 40% × 760).
- Total raw at 81k filtered: ~36–50M (state, action) pairs.
- With augmentation (board symmetry ~2–4×, slot permutation ~2–3×) over ~5 epochs: ~1–2B effective sample-views.
- BC compute anchor: Strakam's 3h H100 BC × ~10× sample volume × ~2× per-sample → ~30–60 H100-hours per BC run.
- **Note:** these numbers were computed under the prior stars-threshold perspective filter. Under the v1 plan in §5 (curated-list primary), per-game yield may revise — verify via filter-counts report and update as needed.

---

## 9. Dataloader pattern (informs the C decision)

- `IterableDataset` + worker-per-process trajectory walking.
- Each worker picks a game file at random, walks t=0..T accumulating fog state online, emits frames.
- Frames flow into a shuffle buffer (~8k–32k) → mini-batches sample from the buffer → effective i.i.d. sampling at the minibatch level.
- Augmentation (symmetry + slot permutation) applied on-the-fly per frame after channel assembly.
- BC default ordering = i.i.d. random shuffling. Strakam BC is the closest scale calibration anchor; their paper is silent on batch composition specifically, so AlphaStar's BC pretraining is the methodological reference for *how* to shuffle.
- **Train/val split — Tentative v1 policy:**
  - If ≥ 150 strong players survive corpus filtering: hold out ~10–15% of players entirely (~15–25 players, skill-stratified across star-percentile buckets) for the primary val set. Tests generalization to unseen play patterns — closer to the live-ladder deployment scenario.
  - If < 150 players: fall back to game-id-level split (prevents within-game leakage per MVP §2, but player-level leakage remains — same player appears in both splits, model can memorize player-specific tendencies). Document as a known v1 limitation; live-ladder eval is the real generalization test anyway.
  - Either way, also keep a small in-distribution game-level val set (~5% of training-player games) as a separate, fast-converging debugging signal.
  - Sanity-check post-split distributions (placement rate, game length, stars-at-start, version) and restratify if meaningfully skewed.
  - **Adjacent option not adopted for v1:** temporal split (hold out the most recent N weeks across all players) tests generalization to *future* play patterns; could complement or replace player-level later.
  - Revisit during corpus-filter / dataset-quality work.
- **Shuffle buffer sizing — Tentative:** floor of ≥ 30× mean trajectory length (~16k frames at our distribution; ~30k to comfortably accommodate p90-length trajectories). Concrete defaults: 20–30k frames per worker for local Mac dev (RAM-constrained), 50–100k per worker for cloud runs (pick generously when RAM allows). RAM cost scales as `num_workers × buffer_size × ~76 KB/frame` — e.g., 8 workers × 50k × 76 KB ≈ 30 GB.
  - **Caveat:** verify available RAM on whatever GPU instance we use before locking buffer size. Cloud providers (Modal, etc.) typically offer multiple instance tiers per GPU type with different CPU/RAM budgets and per-hour costs — pick a tier with comfortable RAM headroom for the chosen buffer. Can evaluate the potential training benefit of larger buffers (cleaner i.i.d. → fewer epochs to convergence, less noisy val signal) against the per-hour cost difference between tiers; sometimes a more-RAM tier is the cheaper total run.
- Chunk-prefetch depth: **Open / TBD** — dataloader-level tuning.

---

## 10. Validation strategy

- **Final-state ownership check**: simulator's end-of-game state matches the replay's ranking. Source of ranking: the listings API, stored in `replay_players` in the collector DB. The wire format itself does not carry it.
- **Spot checks**: dump rendered timesteps from a few games, compare against the in-browser replay player on generals.io.
- **Automated ground-truth diff** (candidate, worth doing before any major training run): small Node-side harness using the saved live JS bundle's `deserialize` (`research/gior-format/generals-main-prod-v31.4.1-d51b92c0.js`, authoritative for current mechanics) to dump per-tile per-timestep state for a handful of games; diff against parser output. Could also seed with hand-collected ground-truth captures from the official replay viewer. Strongest available signal for catching per-tile simulator drift — exactly the failure mode the other paths miss. Specifics deferred.
- **Wire-slot cross-check**: first 14 slots vs. legacy JS parser (`vzhou842/generals.io-Replay-Utils`) — useful for slot decoding only, not v18 mechanics.
- **Round-trip parser ↔ live decoder**: deferred to bucket 3 (live deployment work). Per MVP §4: dry-run on a recorded game, compare live-decoded observations to parser-decoded observations of the same game. Catches format mismatches before they show up as bad live play.

---

## 11. Early-implementation TODOs (resolve before / during simulator code)

These are concrete unknowns whose resolution is required for simulator correctness. They're empirical questions, not design questions — flagged here so they aren't quietly skipped.

### 11.1 Surrender / AFK countdown duration

- **The unknown:** when a player goes AFK or surrenders, the `afks` array records the moment. But the duration of the countdown before their territory transitions to neutral (and their general/cities become neutral cities) is not in the replay format. `game-mechanics.md` §11 estimates ~10–15 turns; not confirmed.
- **Verification approaches** (multiple paths, fastest first):
  1. **Read the live JS bundle's countdown logic directly** (`research/gior-format/generals-main-prod-v31.4.1-d51b92c0.js` — already saved, authoritative for current mechanics). Gives the exact constant in one shot.
  2. Pick representative games with at least one AFK / surrender event. Manually inspect when the AFK'd player issued their final move in the replay data.
  3. Cross-reference against the official web-based replay browser on generals.io to see when their territory visually transitions to neutral.
  4. Cross-check via capture events: a surrendered player can only be **captured** before they convert to neutral tiles. Capture events on AFK'd players bound the countdown from below.
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

### 11.4 Tie-resolution edge cases (`game-mechanics.md` §11.1–11.3)

`game-mechanics.md` §11 documents three mechanics that the simulator must resolve but whose exact rules are unconfirmed:

- **Convergent moves with exact-tie armies** (mechanics §11.1): two players moving onto the same neutral tile simultaneously with armies differing by 1 — does the larger capture with 1 remaining, or is it treated as a tie?
- **Two simultaneous attackers on one defender** (mechanics §11.2): exact resolution order unknown (player index? larger army first? something else?).
- **Production timing on capture** (mechanics §11.3): if a city is captured on a timestep coincident with a turn boundary or the land tick, does it produce for the new owner that turn? Believed "next tick" but unconfirmed.

**Verification:** same paths as §11.1 — read the live JS bundle's resolution code directly, then cross-check against representative replays where these edge cases occur. All three have the same blast-radius property as the items in §12 — silent simulator drift if mis-implemented.

---

## 12. Mechanics hot zones (high blast radius if wrong)

- **Move-priority chain rule** (inward-first) — `game-mechanics.md` §6. Must match server resolution exactly.
- **`player_transforms` orientation** — easy to forget; produces silently inconsistent map orientation across training data if missed.
- **Half-turn vs. turn semantics** — DB and wire `turn`/`turns` fields are half-turns (= timesteps = one move per player). No ×2 conversion. Notational landmine.
- See §11 for the unresolved-but-tractable empirical questions.

---

## 13. Hard prerequisites still in flight

- **Obs-tensor consolidation** (`5.06-1` §"Obs-tensor consolidation pass"). C insulates the parser from channel layout, but the dataloader's channel-assembly code is downstream of this consolidation.

(Action space was previously in flight; now resolved per `network-architecture-design.md` §4 — referenced from §7.)

---

## 14. Deferred / lower-level

- Exact on-disk format for C output (gzipped `.npz` per game suffices for v1; HDF5 / Zarr / webdataset are graduation paths if file-count or per-file ops become a bottleneck).
- Win-rate threshold value + top-3 vs. 1st-place metric mix — tune empirically via filter-counts report; see §5.
- Chunk-prefetch depth — dataloader-level tuning. Buffer-size floor itself is spec'd in §9.
- Phase 2 self-play simulator (separate effort if Phase 2 happens; likely fork strakam's JAX `game.py`).

---

## Appendix A: Decision provenance

Brief notes on calls where the doc records the conclusion but the reasoning is light. For future readers re-visiting the design.

### A.1 Option C over A and B (§3)

- Initial back-of-envelope used the stale corpus size from `replay-collector/README.md` (~6.7k full-data replays). Corrected mid-discussion to ~140–145k. Recomputed sizes: A ≈ 30 TB (infeasible), B ≈ 3 TB (painful at hobby scale), C ≈ 130 GB raw / 50–65 GB gzipped (decisively manageable).
- Beyond storage: C decouples parser from obs-tensor schema, so any channel-layout revision is a dataloader change only — no corpus re-parse. Compounds across the planned ~5 epochs of multi-experiment iteration.

### A.2 Curated player list as primary filter (§5)

- Original v1 plan was stars-at-game-start threshold as primary filter. Rejected after recognizing stars are season-cycle volatile (10-week resets; early-season suppressed across all players, including elite). A fixed threshold either excludes legitimate early-season elite play or admits late-season tier-2 play.
- Curated list is robust because tier-1 *player identity* is reasonably stable across seasons even when their absolute stars are not.
- Win-rate secondary filter is seasonally-immune by construction (rolling per-player metric, decoupled from absolute stars).

### A.3 NumPy parser, PyTorch trainer (§6)

- JAX considered for both pieces. Parser is sequential per-game CPU work — JAX's vmap/scan/JIT pay zero benefit on small-grid sequential ops with control-flow-heavy logic.
- PyTorch trainer driven by user's local-dev preference for Apple Silicon (MPS). PyTorch MPS support is mature; JAX's `jax-metal` is experimental.
- Modal cloud GPU complements rather than replaces MPS — MPS for fast inner-loop dev (no network round-trip), Modal H100 for real training runs (~5–10× faster than M-series on CNN workloads).

### A.4 Threshold and parameter values (§5, §9)

- **Shuffle buffer ≥ 30× mean trajectory length:** ensures ~30+ concurrent trajectories in buffer, comfortably above the ~15 below-which mini-batches show visible correlation. Below 15 concurrent, BC trains but val signal is noisier.
- **100-game rolling win-rate window:** large enough for statistical stability, small enough to track recent skill drift.
- **25% 1st-place rate baseline (8-player FFA):** reference points are random = 12.5%, "consistently above average" ≥ 50%, tier-1 typically ≥ 25%. Tunable via filter-counts report.
- **Top-3 rate (random = 37.5% in 8-player):** complementary metric if 1st-rate alone is too noisy at low game counts.
- **~33–45% pass-frame rate (§7):** napkin estimate from user experience; ~20% of games are open-turtle (~50–80% pass) + ~80% normal (~25% pass). Verify empirically once parser produces samples.

### A.5 Strakam as BC reference (§8, §9)

- Two-stage recipe: BC (3h H100, 16k filtered games) → PPO self-play (36h). Initially framed (incorrectly) as RL-only — corrected to BC + PPO. Their BC stage is our closest scale calibration anchor.
- Paper is silent on batch composition / sample ordering — AlphaStar BC pretraining is the methodological reference for *how* to shuffle.

### A.6 `generalTrades` mechanic (§11.3)

- Initially mistaken for a 2v2 team-mode mechanic. Corrected: it's a recent FFA addition handling simultaneous mutual-general-captures via position-swap. Both players survive, swap all tiles / cities / armies / generals. Replaces older tie-breaker rules.

### A.7 Methodological note

- This doc went through an Opus-agent fresh-perspective review after the initial draft (broad-range design review, returned ~10 substantive flags). Most findings were folded in directly; the rest surface as documented Open / TBD items. Pattern (write design → fresh-perspective review → fold corrections) worth repeating on future load-bearing design docs.
