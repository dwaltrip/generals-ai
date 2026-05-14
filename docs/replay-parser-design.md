# Replay Parser Design — v1 Working Spec

**Date:** 2026.05.07 (last revised 2026.05.12)

**Status:** **Working draft.** High-level design and key decisions captured; nitty-gritty implementation details deferred.

**Post-design implementation update (2026-05-14):** The initial implementation was Python + NumPy as described below. After it was working and ranking-match-validated, the timestep loop (decode → state → combat/moves → step) was ported to Rust as the `sim_core` PyO3 extension (`sim-core/`); `parse_replay` is now a thin orchestrator that decodes (still Python) and calls `sim_core.simulate(replay)` once per game. The high-level structural decisions in this doc are unaffected. Implementation details below that describe the Python sim (state representation, step body, parity oracle, etc.) are historical — current state lives in `sim-core/src/state.rs` and the `sim_core` `.pyi` stub. We may eventually archive or convert this doc to a session note; for now treat it as design-rationale-with-stale-implementation-details.

**Stale numbers:** Inline corpus-size, filter-survival, and sample-volume figures (§2, §3, §4 lines 63–64, §8, §10 denominator) lag the current corpus. As of 2026-05-12: ~178k v15+ replays / ~170k post-§4-filter (up from the doc's ~140–145k baseline). The expansion doesn't break any structural decisions — more data gives the §5 training-time quality filter more headroom, and the per-game intermediate scales linearly with games (well within hobby budgets at any plausible corpus size). Refresh stats via `replay-collector/scripts/basic-replay-stats.sh` and `replay-collector/scripts/filter_counts_report.py`.

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

## 3. Output design — compact per-game intermediate — **Locked**

- Parser writes a **compact per-game intermediate**, not materialized observation tensors.
- Channel assembly happens in the **dataloader**, not the parser.
- This decouples the parser from obs-tensor schema → revisions to `observation-tensor-design.md` cause dataloader changes only, not corpus re-parses.
- Per-game contents:
  - **Game-static**: id, version, map dims, mountains, cities + initial cityArmies, neutrals + initial neutralArmies, generals, ranking, modifiers list
  - **Per timestep, raw state**: ownership (H×W int8) + armies (H×W int16 — single-tile stacks above 32k aren't expected in competitive FFA play; the parser asserts this during simulation and logs+skips any violating game) + cities mask (H×W, bitpacked), ≈ 3 KB/timestep (at 30×30). Cities mask is dynamic because eliminated generals become cities; storing per-timestep is simpler than reconstructing from capture events in the dataloader, with negligible storage cost (cities are well under mountain density of ~20%).
  - **Per perspective**: action stream (supervised targets), per-game metadata (player_id, stars-at-start, placement, elim turn, perspective index).
- Sizing: ~2 MB/game raw, ~160 GB raw / **~50–65 GB gzipped** at 81k filtered replays.
- **Storage approach:** store *only* per-timestep raw game state; the dataloader walks each trajectory forward online, accumulating fog state as it goes. ≈ 50–65 GB gzipped. Random-frame access is approximated via a shuffle buffer + chunk prefetch in the dataloader (§9), preserving approximate i.i.d. minibatch composition *given a sufficiently large shuffle buffer* (§9 specifies the floor — load-bearing for the methodology argument). Decouples parser from obs-tensor schema completely — any channel change is a dataloader change only.
- **Alternatives considered and rejected on size:** materializing full obs tensors at parse time, and storing per-perspective precomputed fog state alongside raw game state. Both were prohibitively large for hobby-tier storage at this corpus scale (appendix A.1 has size comparisons).

---

## 4. Corpus filtering — vanilla FFA only — **Locked (filter rules); Tentative (frequency expectations)**

- Hard filters:
  - `player_count ∈ [4, 8]`
  - `teams` slot null (FFA only — team modes overlap on player_count and would otherwise sneak in)
  - `version ≥ 15` (avoids pre-v15 mechanics changes — `old_priority_v2`, city-regen rules. v15 is the largest single version in the current corpus and ~80–85% of replays are v15+. Confirms via filter-counts report.)
  - `modifiers` slot empty
  - All modifier tile arrays empty (swamps, deserts, tunnels, lookouts, observatories, strongholds) — modifier tiles sometimes appear in regular FFA ladder games during weekend events even with the `modifiers` slot empty. The parser doesn't implement modifier-tile mechanics.
  - Custom map slot (`map`) null
  - Chess-clock slot (`chessClockTimingsByMove`) null
  - `generalTrades` slot empty (mutual-general-swap mechanic; ~1.8% of v16+ FFA games per the filter-counts report — cheap drop, mechanic implementation deferred per §11.3)
- A **filter-counts report runs first** against the existing corpus to surface the actual distribution before commitments harden. Want to see the real numbers before locking the filter rules in code.
- Per `compute-considerations.md`: expected ~60% filter survival → ~81k filtered replays.

---

## 5. Perspective selection — **Locked** (parse-time architecture); **Open / TBD** (training-time thresholds)

The parser emits all curated-player perspectives along with rich per-perspective metadata. Quality filtering — picking the skill bar, ablating thresholds — is a training-time concern; threshold tuning happens via config, not via corpus re-parse. The parser itself applies only a conservative random-baseline noise floor as architectural defense against upstream collection changes.

### 5.1 Three-layer filtering pipeline

- **Upstream (collection)** — curated player list, controlled by `replay-collector/`. Determines which usernames we fetch replays for at all.
- **Parse-time (this section)** — emit all curated-player perspectives; apply the conservative noise floor (§5.3); emit rich per-perspective metadata (§5.2).
- **Training-time (config)** — quality thresholds, threshold ablation, trajectory-quality filters (long-game, bad-start, died-quickly, etc.).

### 5.2 Per-perspective metadata emitted

For each curated-player perspective in each game's ranking:

- `player_id`
- `perspective_index` — slot ordinal in this game's ranking.
- `rolling_1st_rate_at_game_time` — 1st-place rate over the player's most recent 200 games strictly prior to this game's start (capped at 200; if fewer prior games available, computed over what's available, subject to the noise-floor minimum in §5.3).
- `rolling_top3_rate_at_game_time` — top-3 rate over the same window.
- `prior_games_count_at_game_time` — count of the player's prior FFA games at this game's start. Drives the noise-floor minimum in §5.3 and is useful at training time for weighting by sample reliability.
- `stars_at_start` — slot 5; captured for training-time use (season-volatile due to 10-week stars resets, so absolute thresholding is unreliable — rolling win-rate and top-3 are the seasonally-immune alternatives).
- `placement` — final placement from `replay_players`.
- `elim_turn` — turn the player was eliminated; `null` if they survived to game-end.

Rolling rates are computed via a per-player chronological walk (`ORDER BY started ASC` per player) so "most recent N games" means "the N games immediately preceding this one in time" — same convention as the filter-counts report.

### 5.3 Parse-time noise floor

Conservative random-baseline thresholds, applied per perspective (all AND-ed):

- `rolling_1st_rate > 12.5%` (random baseline for 8-player FFA win rate)
- `rolling_top3_rate > 37.5%` (random baseline for 8-player FFA top-3 placement)
- `prior_games_count >= 50`

The win-rate cuts sit below where any real serious player lives — they catch sub-random outliers like `kickapp` (rolling 1st-rate ~0.06 per the filter-counts report). The prior-games floor drops each player's first 50 games entirely; rolling rates over very small samples are noise, not signal.

Expected to be a near-no-op on the current curated corpus by design. Its purpose is architectural defense: if the upstream collection pipeline ever broadens (e.g., non-curated sources), the parser still emits only plausibly-real-player perspectives.

### 5.4 Why quality filtering is training-time, not parse-time

The right skill bar (elite-only vs. very-strong vs. broader strong-player slice) depends on questions we cannot answer without training experiments:

- **Some perspective-quality filters need parser output to define.** Trajectory-based filters (long-game, bad-start, died-quickly) operate on the parsed move sequence and per-turn state; they don't exist yet.
- **Sample-volume target depends on compute shape.** At ~170k §4-filtered games and ~268k curated perspectives (filter-counts report), we're well above Strakam's 16k BC anchor under any reasonable cut.
- **The right strictness is empirical.** Stricter cut → cleaner data + smaller corpus; looser → more data + more noise. Which balance trains a better BC agent is an experimental question.
- **Relative leverage vs. architecture / obs-tensor / training hyperparams is unknown.** Without runs, we can't tell whether skill-cut tuning matters more or less than other knobs.

The architecture above makes threshold ablation a config change — no corpus re-parse needed.

### 5.5 Related corpus enhancement work (future / optional)

**Note:** This is *not* part of the parser design, but it came up when discussing various noise and player skill level thresholds.

The curated list can grow via win-rate-based discovery from the cached corpus. We can:

* Find candidate strong players in existing games (using a lax stars pre-filter, which is fine as the real test is win-rate)
* Fetch their full listing histories (metadata-only via the existing collector, cheap) and then calculate win-rate statistics. The exact metric is TBD (career win-rate probably isn't correct)
* The new players with sufficiently high win-rates then get passed to `fetch-gior` , adding their replays to the corpus.

This could be iteratively repeated (like convergence after 2–3 rounds). It's an optional future enhancement, especially if we decide we want more data.

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
- **Wire → typed records and arrays**, with version gating (`generalTrades` ≥ v16, strongholds ≥ v18):
  - **Hot-loop data** (`moves`, `afks`): columnar NumPy arrays at decode time — `moves_index: int8[N]`, `moves_start: int32[N]`, `moves_end: int32[N]`, `moves_is50: uint8[N]`, `moves_turn: int32[N]`. Per-tick simulator loop slices by precomputed turn-range offsets for fast access.
  - **Low-volume data** (game-static metadata, `chat`, `pings`): named-field records.
- **Game simulator** (the hard part): per-turn production (general + each owned city), per-round land tick (every 25 turns, +1 on every owned tile), move resolution with the priority sort + inward-first dependency check (`game-mechanics.md` §6), capture mechanics (army halving, territory transfer, captured general → city), AFK lifecycle driven by the `afks` array (paired kill / neutralize events — often only the kill event when capture or game-end pre-empts the countdown; e.g., a 2nd-to-last-player surrender ends the game immediately. See §11.1).
- **Per-perspective state tracking**: raw simulator state is per-game; action targets are per-perspective.
- **Action extraction**: each `[index, start, end, is50, turn]` is the supervised target for the moving player at that timestep; non-moving frames target "pass". Action space is settled in `network-architecture-design.md` §4: per-cell `[pass + 4 directions × 2 splits]`.
  - **Pass-frame rate** (napkin-math from user experience; verify empirically once parser produces samples): ~30–45% corpus-wide. Decomposes into:
    - ~20% **open-turtle** games — elite-player-specific behavior: deliberate early-game concession when ganged up on by 2–3 attackers, often gathering small armies onto the general while otherwise idling. Distinct from the generic turtling weaker players do. Pass-rate within these games: ~50–80%.
    - ~80% **normal** games — pass-rate < 30%, often much lower.
  - 30–45% sits comfortably inside vanilla-cross-entropy comfort range; no class-weighting expected at v1. Revisit only if empirical rate exceeds ~80%.
- **Metadata enrichment** per MVP §2: player_id, rating-at-game-start, placement, elim turn, perspective index, etc.
- **Robustness**: skip-and-log on decode failures from `replay_collector/wire.py:decode` (the canonical decoder for the DB's `wire_data` BLOB); a malformed wire-data row should never abort a corpus parse.
- **Validation harness** (see §10).

**Simulator orchestration order (per timestep):**

1. Process pending AFK events from the `afks` array (events with `turn <= current_turn`).
2. Buffer pending moves from the `moves` array into per-player queues (preserve array order).
3. Resolve at most one move per player via the priority sort + inward-first dependency check; execute combats and captures.
4. Increment the turn counter.
5. Every 2nd timestep: generals + each owned city produce +1 army.
6. Every 50th timestep: each owned tile gains +1 army (the land tick).
7. Recompute scores; check for game-end (one player remaining, or one of the synthetic fallbacks in `game-mechanics.md` §9 fires).

Stop processing pending AFK events the moment game-end is reached — the second AFK event often never fires in practice (see §11.1).

---

## 8. Sample-volume and training calibration — **inherited from `compute-considerations.md`**

- Per filtered replay: ~530 (state, action) samples (60% × 380 + 40% × 760).
- Total raw at 81k filtered: ~36–50M (state, action) pairs.
- With augmentation (board symmetry ~2–4×, slot permutation ~2–3×) over ~5 epochs: ~1–2B effective sample-views.
- BC compute anchor: Strakam's 3h H100 BC × ~10× sample volume × ~2× per-sample → ~30–60 H100-hours per BC run.
- **Note:** these numbers were computed under the prior stars-threshold perspective filter. Under the v1 plan in §5 (curated-list primary), per-game yield may revise — verify via filter-counts report and update as needed.

---

## 9. Dataloader pattern (informs the intermediate-format decision)

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

The v1 quality gate is **placement-outcome ranking-match** across the filtered corpus. Other signals are useful checks but secondary.

- **Primary: placement-outcome ranking-match.** For each filtered replay, the simulator's deduced final ranking must equal the listings-API ranking stored in `replay_players` in the collector DB. The ranking is computed from end-of-game state via the bundle's `lbSort` tiebreak (players with kills outrank kill-less players regardless of army totals; then alive > dead; then dead-players by reverse death order; then total army desc, tile count desc, player index asc). **Target ≥ 99.9%** match across ~170k filtered games (≤ ~170 mismatches); stretch ≥ 99.99%. The validator runs over the full §4 vanilla-FFA-filtered corpus (~170k games), independent of the §5 perspective-quality filter that produces the smaller ~81k figure in §8 sample-volume math — simulator correctness doesn't depend on which perspectives we ultimately train on. Bundle-independent server-truth signal.
- **Spot checks**: dump rendered timesteps from a few games, compare against the in-browser replay player on generals.io. Cheap qualitative check during parser development.
- **JS bundle diff (deferred until needed).** Node-side harness using the saved JS bundle's `deserialize` (`research/gior-format/generals-main-prod-v31.4.1-d51b92c0.js`) to dump per-tile per-timestep state; diff against parser output. The bundle is our most-documented reference implementation, but it is the replay-viewer's reconstruction — historical bugs exist. Useful for diagnosing per-tile drift if ranking-match plateaus below target, but not a primary gate.
- **Wire-slot cross-check**: first 14 slots vs. legacy JS parser (`vzhou842/generals.io-Replay-Utils`) — useful for slot decoding only, not v18 mechanics.
- **Live-game observation capture (deferred).** Capture per-timestep observations from the live server over the WebSocket during a small set of real games; diff against parser output. Per-timestep server-truth — the gold standard — but a meaningful chunk of infrastructure to build. Spin up only if ranking-match plateaus and per-timestep diagnosis is required.
- **Round-trip parser ↔ live decoder**: deferred to bucket 3 (live deployment work). Per MVP §4: dry-run on a recorded game, compare live-decoded observations to parser-decoded observations of the same game. Catches format mismatches before they show up as bad live play.

---

## 11. Resolved mechanics references

Implementation-relevant mechanics decisions for the simulator. Full player-facing statement of these rules: `generals-io-game-mechanics.md` §6, §7, §9. Bundle line refs: its appendix.

### 11.1 Surrender / AFK countdown — Resolved

The countdown is **50 timesteps** at the standard game speed (= 25 turns / 25 seconds / one full round). However, the replay records both the kill and neutralize moments as paired entries in the `afks` array — drive both directly from the data rather than hardcoding the constant. Empirically only ~37% of AFK'd players have both events in the replay (the rest have only the kill: game ended or capture pre-empted the countdown). The simulator must stop processing AFK events past game-end.

### 11.2 `moves[].turn` timestep-within-turn — Resolved

Array order within `moves` is authoritative. The bundle drains all moves with `turn <= current_turn` into per-player input buffers in array order, then consumes one move per player per timestep. Same-`turn` moves for the same player therefore execute across consecutive timesteps in array order.

### 11.3 `generalTrades` — Resolved (filter, not implement)

Filter out games with non-empty `generalTrades`. ~1.8% of v16+ FFA games (filter-counts report). Added to the §4 hard-filter list. Mutual-general-swap implementation deferred indefinitely.

### 11.4 Tie-resolution edge cases — Resolved

All three (convergent moves, two-attacker, production-on-capture) reduce to the v15+ priority sort + sequential resolution. Within a timestep, moves are sorted by: defensive-first, general-attacks-last, larger-source-army-first, input-order tiebreak. The inward-first rule is applied as a dependency check on top of the sort. For production-on-capture: moves resolve first, then turn counter increments, then production runs against the post-move state — a capture this timestep produces for the new owner immediately if the new turn is a production turn.

Full statement of these rules: `generals-io-game-mechanics.md` §6, §7, §9.

---

## 12. Mechanics hot zones (high blast radius if wrong)

- **Move-priority chain rule** (inward-first) — `game-mechanics.md` §6. Must match server resolution exactly.
- **Half-turn vs. turn semantics** — DB and wire `turn`/`turns` fields are half-turns (= timesteps = one move per player). No ×2 conversion. Notational landmine.

---

## 13. Hard prerequisites still in flight

- **Obs-tensor consolidation** (`5.06-1` §"Obs-tensor consolidation pass"). The per-game intermediate insulates the parser from channel layout, but the dataloader's channel-assembly code is downstream of this consolidation.

(Action space was previously in flight; now resolved per `network-architecture-design.md` §4 — referenced from §7.)

---

## 14. Deferred / lower-level

- Exact on-disk format for the per-game intermediate (gzipped `.npz` per game suffices for v1; HDF5 / Zarr / webdataset are graduation paths if file-count or per-file ops become a bottleneck).
- Chunk-prefetch depth — dataloader-level tuning. Buffer-size floor itself is spec'd in §9.
- Phase 2 self-play simulator (separate effort if Phase 2 happens; likely fork strakam's JAX `game.py`).

---

## Appendix A: Decision provenance

Brief notes on calls where the doc records the conclusion but the reasoning is light. For future readers re-visiting the design.

### A.1 Compact per-game intermediate, over alternatives (§3)

- Initial back-of-envelope used the stale corpus size from `replay-collector/README.md` (~6.7k full-data replays). Corrected mid-discussion to ~140–145k. Recomputed sizes:
  - Full obs tensor materialized per perspective: ≈ 30 TB (infeasible).
  - Raw game state + per-perspective precomputed fog state: ≈ 3 TB (painful at hobby scale).
  - Raw game state only (the chosen format): ≈ 130 GB raw / 50–65 GB gzipped (decisively manageable).
- Beyond storage: the chosen format decouples parser from obs-tensor schema, so any channel-layout revision is a dataloader change only — no corpus re-parse. Compounds across the planned ~5 epochs of multi-experiment iteration.

### A.2 Choice of curated list over stars-based filtering (§5)

- Original v1 plan was stars-at-game-start as the primary skill signal. Rejected after recognizing stars are season-cycle volatile (10-week resets; early-season suppressed across all players, including elite). A fixed threshold either excludes legitimate early-season elite play or admits late-season tier-2 play.
- Curated list is robust because tier-1 *player identity* is reasonably stable across seasons even when their absolute stars are not. Drives upstream collection in `replay-collector/`.
- Rolling win-rate and top-3 rate signals are seasonally-immune by construction (rolling per-player metrics, decoupled from absolute stars) — they feed the parse-time noise floor (§5.3) and training-time quality filtering (§5.4).

### A.3 NumPy parser, PyTorch trainer (§6)

- JAX considered for both pieces. Parser is sequential per-game CPU work — JAX's vmap/scan/JIT pay zero benefit on small-grid sequential ops with control-flow-heavy logic.
- PyTorch trainer driven by user's local-dev preference for Apple Silicon (MPS). PyTorch MPS support is mature; JAX's `jax-metal` is experimental.
- Modal cloud GPU complements rather than replaces MPS — MPS for fast inner-loop dev (no network round-trip), Modal H100 for real training runs (~5–10× faster than M-series on CNN workloads).

### A.4 Threshold and parameter values (§5, §9)

- **Shuffle buffer ≥ 30× mean trajectory length:** ensures ~30+ concurrent trajectories in buffer, comfortably above the ~15 below-which mini-batches show visible correlation. Below 15 concurrent, BC trains but val signal is noisier.
- **200-game rolling win-rate window:** large enough for statistical stability, small enough to track recent skill drift. Capped at the most recent 200; below 50 prior games the perspective is dropped by the §5.3 noise floor.
- **Random-baseline parse-time noise floor (§5.3):** 1st-rate > 12.5% and top-3 > 37.5% (8-player FFA random baselines) — set deliberately below where any real serious player sits, so the cuts catch sub-random outliers without making skill-quality judgments at parse time.
- **Training-time skill threshold (§5.4):** intentionally Open / TBD. Reference points: random = 12.5% (1st), 37.5% (top-3); "consistently above average" ≥ 50% (1st); tier-1 typically ≥ 25% (1st). Resolved empirically via BC runs at different settings.
- **~33–45% pass-frame rate (§7):** napkin estimate from user experience; ~20% of games are open-turtle (~50–80% pass) + ~80% normal (~25% pass). Verify empirically once parser produces samples.

### A.5 Strakam as BC reference (§8, §9)

- Two-stage recipe: BC (3h H100, 16k filtered games) → PPO self-play (36h). Initially framed (incorrectly) as RL-only — corrected to BC + PPO. Their BC stage is our closest scale calibration anchor.
- Paper is silent on batch composition / sample ordering — AlphaStar BC pretraining is the methodological reference for *how* to shuffle.

### A.6 `generalTrades` mechanic (§11.3)

- Initially mistaken for a 2v2 team-mode mechanic. Corrected: it's a recent FFA addition handling simultaneous mutual-general-captures via position-swap. Both players survive, swap all tiles / cities / armies / generals. Replaces older tie-breaker rules.

### A.7 Methodological note

- This doc went through an Opus-agent fresh-perspective review after the initial draft (broad-range design review, returned ~10 substantive flags). Most findings were folded in directly; the rest surface as documented Open / TBD items. Pattern (write design → fresh-perspective review → fold corrections) worth repeating on future load-bearing design docs.
