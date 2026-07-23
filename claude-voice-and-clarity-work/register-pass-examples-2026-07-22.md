# Register-pass examples — 2026-07-22

Raw material for a possible future tells catalog: the diff from a register-focused finishing pass on `docs/2026-07/7.22-2-channel-summary-design-sketch.md`, captured from the working tree before it was committed. Each hunk is a before/after pair. The "before" is agent-drafted prose that had already survived one finishing pass run from the written guidelines alone.

Why this batch is better material than usual: earlier in the same session, the doc's intro went through several live correction rounds with Daniel (agent-noun coinages, relationship-abstraction sentences, section-vocabulary register, stub-sentence choppiness). This pass ran immediately after, with those corrections still in context, and it caught further instances of the corrected patterns that the written guidelines alone had missed. The intro's own evolution is not in this diff, since it had already been committed. Those examples live in the session transcript.

## Tell labels for the hunks below

- **Agent-noun / personification**: "orients a debugger" (misreads as the software tool), "the build isn't surprised", "announcing itself", "which tables lit up".
- **Coined compound standing in for a sentence**: "two-location", "same-writer atomicity", "forensic self-description", "one-rule simplicity", "assumption surface", "earning case".
- **Cute or colorful where standard vocabulary exists**: "coarse by mandate", "painted over the board", "Anatomy", "workhorse magnitude", "glaring", "the cap never binds", "buys orientation", "one design idea at two depths", "Worked exemplars".
- **Term collision**: "marks a boundary the report stays behind" uses a metaphorical "boundary" in a doc family where boundary is precise vocabulary.
- **Construction**: ungrammatical embedded quote ("with the rationale \"which git diffs...\""), gerund-headed subject ("Tracing the actual consumers moved..."), elliptical fragments ("Counts are cells summed...", "(a stable layout, and ...)"), long free-relative subject ("What we believe the values mean lives...").
- **Accuracy error surfaced by unpacking a coined compound**: "only hand-maintained assumption surface" contradicted the list of other hand-maintained assumptions in the same sentence. Expansion was the diagnostic.

## The diff

*Update: the actual diff is saved in full to commit `4ebf24e`. I'm keeping the copy below here just in case.*


```diff
diff --git a/docs/2026-07/7.22-2-channel-summary-design-sketch.md b/docs/2026-07/7.22-2-channel-summary-design-sketch.md
index f29e1dc..e4001ce 100644
--- a/docs/2026-07/7.22-2-channel-summary-design-sketch.md
+++ b/docs/2026-07/7.22-2-channel-summary-design-sketch.md
@@ -14,9 +14,9 @@ The design targets two main use cases. The primary one is bless-time review of a
 
 The recipes were selected against rules inherited from 7.15-2, restated here as the local decision criteria.
 
-- **Detection is fully carried by the frame hashes.** A stat is never justified by what it might catch. Every stat needs a legibility case: it makes a bless diff readable, or it orients a debugger at fire time.
+- **Detection is fully carried by the frame hashes.** A stat is never justified by what it might catch. Every stat needs a legibility case: it makes a bless diff readable, or it helps orient debugging after a fire.
 - **The bless-commit reviewer is the primary consumer.** The design test used throughout: when an intentional change is blessed, does the report's diff let the reviewer confirm that the change's sign, scope, and rough size match intent?
-- **Fire-time magnitude information is coarse by mandate.** Tick-level and cell-level questions belong to the frame hashes and the rerun recipe. The dropped dense stats layer (7.15-2 §5) marks a boundary the report stays behind: whole-game aggregates only, no per-tick or time-windowed stats. Whole-game temporal aggregates such as flip counts are aggregates rather than time series, and are fine.
+- **Fire-time magnitude information is coarse by design.** Tick-level and cell-level questions belong to the frame hashes and the rerun recipe. The dropped dense stats layer (7.15-2 §5) sets a limit the report respects: whole-game aggregates only, no per-tick or time-windowed stats. Whole-game temporal aggregates such as flip counts are aggregates rather than time series, and stay within the limit.
 
 Two standing facts shape what the stats can be:
 
@@ -28,15 +28,15 @@ Two standing facts shape what the stats can be:
 Four principles govern the recipes, and future recipe changes should honor them.
 
 - **Stats are functions of the emitted channel bytes only** (the same domain the hashes cover). Anything computed from the fixture's inputs is frozen by construction: it can never move across a re-bless, so it is fixture documentation, not a stat. The example that produced the rule: "how many cells did player X take" is not computable from the transition channels, because the encoder keys on the older owner and never encodes the gainer.
-- **Structure is observational, and assumptions are demoted to annotations.** The tables record what the channels did. What we believe the values mean lives in legend lines, where a wrong belief can only mislabel a correct count, never misfile one. Assumptions are kept only where being wrong is visible in the output and harmless to the data.
-- **Expected set plus overflow, for categorical values.** Count columns cover the expected value set (a stable layout, and a zero count is itself informative). Observed out-of-set values get appended columns with a `?` marker. Nothing is dropped and nothing crashes: an out-of-set value diffs as a new header plus a populated column.
-- **Point masses plus moments, for continuous values.** Exact-value counts for the point-mass values (0 and -1), distribution stats over everything else. This makes the report one design idea at two depths: categorical channels get a full value histogram, continuous channels get a two-bucket histogram plus moments.
+- **Structure is observational, and assumptions are demoted to annotations.** The tables record what the channels did. Our beliefs about what the values mean live in legend lines, where a wrong belief can only mislabel a correct count, never misfile one. Assumptions are kept only where being wrong is visible in the output and harmless to the data.
+- **Expected set plus overflow, for categorical values.** Count columns cover the expected value set (the layout stays stable, and a zero count is itself informative). Observed out-of-set values get appended columns with a `?` marker. Nothing is dropped and nothing crashes: an out-of-set value diffs as a new header plus a populated column.
+- **Point masses plus moments, for continuous values.** Exact-value counts for the point-mass values (0 and -1), distribution stats over everything else. The two patterns are the same idea at different depths: categorical channels get a full value histogram, continuous channels get a two-bucket histogram plus moments.
 
-One property follows from holding these together: the writer's channel-to-table classification map is the report's only hand-maintained assumption surface, and every assumption in the system (expected sets, legends, table placement) fails visibly rather than silently. To keep the property complete, the writer fails loudly on a channel its map does not cover: no guessed recipe, no silent skip. How that failure is detected and surfaced is worked out at build time.
+One property follows from these principles taken together: the writer carries a small set of hand-maintained assumptions (the channel-to-table classification map, the expected value sets, the legends), and every one of them fails visibly rather than silently. To keep the property complete, the writer fails loudly on a channel its classification map does not cover: no guessed recipe, no silent skip. How that failure is detected and surfaced is worked out at build time.
 
 ## 3. The channel map
 
-The default config point emits 110 channels: 86 base, 10 dense-history at n=5, and 14 player-status. The legacy point emits 96 (no player-status group). Two axes place every channel: its class (7.20-3) and its layout. A *spatial* channel has real per-cell structure, and a *broadcast* channel holds one value painted over the board. Recipes are per (class × layout) cell.
+The default config point emits 110 channels: 86 base, 10 dense-history at n=5, and 14 player-status. The legacy point emits 96 (no player-status group). Two axes place every channel: its class (7.20-3) and its layout. A *spatial* channel has real per-cell structure, and a *broadcast* channel holds one value repeated over the whole board. Recipes are per (class × layout) cell.
 
 | class | spatial | broadcast | total |
 |---|---:|---:|---:|
@@ -51,7 +51,7 @@ The two empty class cells are structural: every int-scaled channel is a scoreboa
 
 One deviation from the docs' census: 7.20-1 counted the 7 zero-stub `city_inference` channels as binary (giving binary 58). This report records them as `inactive` instead, because a constant-zero channel has no behavior to classify (§5.6).
 
-Anatomy the recipes rely on:
+Properties the recipes rely on:
 
 - The 28 step channels (21 binary broadcast flags, 7 `captured_by`) are step functions, today with at most one transition per walk.
 - The dense-history channels (5 `ownership_transition`, 5 `army_delta`) are lagged copies of one signal each: the encoder computes each (t, t-1) pair once, and the five lag channels re-read the same stored arrays at different offsets. Whole-game stats of the five rows are near-identical, differing only by edge truncation at the walk's start and end.
@@ -69,16 +69,16 @@ Anatomy the recipes rely on:
 
 ## 4. The report's format
 
-**Markdown, one report per (fixture, config point).** 7.15-2 specified JSON, with the rationale "which git diffs and deltas well". Tracing the actual consumers moved the format off JSON: the machine path (the checker's comparisons, regen's printed fire extent) runs on the hash references, not on the report, and the report's one real consumer is a human reading a git diff. Markdown tables serve that consumer better: one row per channel, so a change diffs as exactly the changed rows, no repeated key names, and the file renders as actual tables. If a future tool ever wants the stats programmatically, it parses the strictly formatted markdown, or a machine-format twin is added at that point (the standard revival-trigger shape).
+**Markdown, one report per (fixture, config point).** 7.15-2 specified JSON on the grounds that it diffs and deltas well in git. We moved off JSON after tracing the actual consumers: the machine path (the checker's comparisons, regen's printed fire extent) runs on the hash references, not on the report, and the report's one real consumer is a human reading a git diff. Markdown tables serve that consumer better: one row per channel, so a change diffs as exactly the changed rows, no repeated key names, and the file renders as actual tables. If a future tool ever wants the stats programmatically, it parses the strictly formatted markdown, or a machine-format twin is added at that point (the standard revival-trigger shape).
 
 **Layout: a provenance header, then seven tables.**
 
-- The header records the hash algorithm and digest size, the canonical serialization (C-order bytes in the emitted dtype), and the unpadded-region stats rule (below). It is forensic self-description: the checker uses a code constant, and the recorded copy answers "which algorithm produced these hashes" for a future investigation.
-- One table per recipe. The class record is table membership: each heading carries the full class and layout name once, and no row carries a class label. A reclassification diffs as a row deleted from one table and added to another, which is loud and two-location, matching the significance of the event.
+- The header records the hash algorithm and digest size, the canonical serialization (C-order bytes in the emitted dtype), and the unpadded-region stats rule (below). It exists for the record rather than for the machine: the checker uses a code constant, and the recorded copy answers "which algorithm produced these hashes" for a future investigation.
+- One table per recipe. The class record is table membership: each heading carries the full class and layout name once, and no row carries a class label. A reclassification diffs as a row deleted from one table and added to another: a loud change that matches the significance of the event.
 
 **Shared columns.**
 
-- Every table carries the channel's hash (16 hex chars), a display duplicate of the `[C]` reference kept honest by same-writer atomicity. Its earning case: a mid-walk change can leave whole-game aggregates visually identical while the hash flips, and the reviewer still sees which channels changed in the same diff.
+- Every table carries the channel's hash (16 hex chars), a display duplicate of the `[C]` reference that cannot drift from it, because both come from the same regen pass. The case that justifies it: a mid-walk change can leave whole-game aggregates visually identical while the hash flips, and the reviewer still sees which channels changed in the same diff.
 - Spatial tables carry `com`, the whole-game center of mass of active cells, formatted `(r̄, c̄)` in board coordinates (rationale in §5.7).
 
 **Alignment and printing.**
@@ -89,7 +89,7 @@ Anatomy the recipes rely on:
 - A legend line above a table carries its semantic annotations (a value-to-meaning map, or the per-sub-group reading of `mean`).
 - An empty stats domain prints a dash, never NaN. The known case is an all-sentinel BFS channel for a general the perspective never discovers.
 
-**Padding.** The hashes cover the padded emitted bytes ("hash what the config emits" extends to shape), so detection has no padding blind spot. Stats are computed over the unpadded H×W board region only, declared once in the header. The zero counts force this rule (they are not padding-immune), and it is adopted uniformly for all recipes for one-rule simplicity.
+**Padding.** The hashes cover the padded emitted bytes ("hash what the config emits" extends to shape), so detection has no padding blind spot. Stats are computed over the unpadded H×W board region only, declared once in the header. The zero counts force this rule (they are not padding-immune), and it is adopted uniformly across the recipes so there is a single rule.
 
 **Config points share recipes.** Recipes are class-driven, so the legacy point reuses them wholesale: it differs only in channel set (96) and emitted dtype (fp32).
 
@@ -108,7 +108,7 @@ The 21 binary flags (`opp_N_contacted`, player-status) and the 7 `opp_N_captured
 | opp_3_captured_by  | 2b94d1c07f6e8a43 | 0       | 842→5            |
 ```

-For a step signal with few transitions, the initial value plus the transition list is lossless at two or three numbers, which beats any aggregate. The transitions field is a list rather than a single slot because "at most one transition per walk" is a fact about current behavior, not a schema guarantee: if a latch-once channel ever re-flips, the list records it as a glaring second entry (evidence in the diff) instead of forcing the writer to error or to keep only the first. The list is capped (first N entries plus a total count) against pathological cases. The cap never binds today.
+For a step signal with few transitions, the initial value plus the transition list is lossless at two or three numbers, which beats any aggregate. The transitions field is a list rather than a single slot because "at most one transition per walk" is a fact about current behavior, not a schema guarantee: if a latch-once channel ever re-flips, the list records it as a second entry that stands out in the diff, instead of forcing the writer to error or to keep only the first. The list is capped (first N entries plus a total count) against pathological cases. The cap is never reached today.
 
 ### 5.2 Numeric broadcast (24 channels)
 
@@ -134,7 +134,7 @@ Legend: -1 = self lost · 0.5 = neutral lost · 1+k/8 = loss by canonical opp k
 | ownership_transition_t-1 | 3fa8c91d22e07b56 | 41  | 388 | 12    | 96   | 30    | 0   | 55    | 8    | 0     | (14.2, 17.8) |
 ```

-Counts are cells summed over the whole walk. The 0 bucket is deliberately absent: it is the overwhelming bulk (almost every cell, every frame, is "no change"), it carries no reviewable signal, and dropping it makes every reported count padding-immune. Per-opponent columns are kept rather than a lumped `opp_lost` because attribution is what makes an encoding change readable: a recode diffs as counts migrating between columns. The five rows are near-identical by the lagged-copy structure (§3). That redundancy is left visible as a standing invariant rather than deduplicated away.
+Each count is the number of cells in that category, summed over the whole walk. The 0 bucket is deliberately absent: it is the overwhelming bulk (almost every cell, every frame, is "no change"), it carries no reviewable signal, and dropping it makes every reported count padding-immune. Per-opponent columns are kept rather than a lumped `opp_lost` because attribution is what makes an encoding change readable: a recode diffs as counts migrating between columns. The five rows are near-identical by the lagged-copy structure (§3). That redundancy is left visible as a standing invariant rather than deduplicated away.
 
 ### 5.4 Binary masks (30 channels)
 
@@ -146,7 +146,7 @@ The spatial masks: `fog_cells`, ownership, map knowledge, `last_seen_owner`, `hi
 | historically_seen  | 4c80be2217aa9035 | 214.6    | 407       | 382   | 0      | (15.1, 16.4) |
 ```

-`mean_nnz` is the per-frame average count of 1-cells (the workhorse magnitude), and `final_nnz` the end state. `gains` and `losses` are whole-game counts of 0→1 and 1→0 cell flips, split into two columns rather than one flips total: cumulative masks then display `losses = 0` as a free monotonicity check, read off the output rather than assumed by the writer. The expected value set is {0, 1}, so a "binary" channel emitting anything else spawns an overflow column, which is the observational class check.
+`mean_nnz` is the per-frame average count of 1-cells (the main magnitude stat), and `final_nnz` the end state. `gains` and `losses` are whole-game counts of 0→1 and 1→0 cell flips, split into two columns rather than one flips total: cumulative masks then display `losses = 0` as a free monotonicity check, read off the output rather than assumed by the writer. The expected value set is {0, 1}, so a "binary" channel emitting anything else spawns an overflow column, which is the observational class check.
 
 ### 5.5 Log-scaled (16 channels)
 
@@ -180,7 +180,7 @@ Counts and moments are blind to one realistic bug class: multiset-preserving spa
 
 "Active" means the cells in each table's stats domain: the 1-cells for masks, the nonzero cells for the histograms, and the non-point-mass cells for the log table.
 
-The trade: the class is hash-detected regardless, and the rerun recipe diagnoses it on demand in about 0.2s. `com` buys orientation (the diff itself says "positional"), and with it the report's remaining blind spot narrows to timing-only changes (§6).
+The trade: the class is hash-detected regardless, and the rerun recipe diagnoses it on demand in about 0.2s. `com` adds orientation (the diff itself shows the change is positional), and with it the report's remaining blind spot narrows to timing-only changes (§6).
 
 ## 6. Reading the report
 
@@ -188,17 +188,17 @@ This section collects the reading patterns the recipes were designed around. The
 
 **Fire signatures.**
 
-- The first read of an accidental fire is breadth: which tables lit up. Cross-channel patterns (fog plus ownership plus transitions) usually name the subsystem faster than any single row.
+- The first read of an accidental fire is breadth: which tables changed. Cross-channel patterns (fog plus ownership plus transitions) usually point at the responsible subsystem faster than any single row.
 - Hashes flipped broadly while stats stay quiet: a serialization- or padding-layer change (stats never see the padding).
 - Hash flips confined to log-scaled rows with stats quiet: the environment-churn signature. Any flip outside the log table is a behavior change.
 - Counts move while magnitudes hold (or the reverse): the two axes separate rule changes from encoding changes. The BFS rows are the clean example: a passability-policy change moves `n(-1)` and `n_active` with magnitudes stable, and a distance-encoding change does the opposite.
 - A row whose hash flipped but whose aggregates are identical: a timing or positional change. `com` separates the two (positional moves it, timing does not).
-- Worked exemplars against project history: a 6.18-style tick-seam change diffs as one step-trace row, `842→0` becoming `843→0`. A divisor change scales all four numeric-broadcast stats by the ratio, uniformly across the affected rows. A transition-encoding recode diffs as histogram counts migrating between columns.
+- Worked examples from project history: a 6.18-style tick-seam change diffs as one step-trace row, `842→0` becoming `843→0`. A divisor change scales all four numeric-broadcast stats by the ratio, uniformly across the affected rows. A transition-encoding recode diffs as histogram counts migrating between columns.
 
 **Free invariants** (all read off the output, none assumed by the writer):
 
 - The five `ownership_transition` rows and the five `army_delta` rows are each near-identical. Divergence beyond edge effects means the lag-buffer logic itself broke.
-- `losses = 0` on cumulative masks. A nonzero value is a monotonicity violation announcing itself.
+- `losses = 0` on cumulative masks. A nonzero value there is a monotonicity violation, visible directly in the diff.
 - The nine `last_seen_owner` rows' nnz roughly partitions `historically_seen`'s nnz.
 - `turns_since_seen`'s derivable zero count approximates the visible-cell count, cross-checking `fog_cells`' `mean_nnz`.
 - `n(-1) = 0` on log rows with no -1 convention.
@@ -226,7 +226,7 @@ The decisions whose reasoning sets reusable patterns, compressed. Chat-level alt
 
 ## 8. Build-time items
 
-Deferred mechanics, listed so the build isn't surprised. None of them block drafting the writer.
+Deferred mechanics, listed here so the build session doesn't have to rediscover them. None of them block drafting the writer.
 
 - The value-range sweep over eligibility-gated fixtures that sizes the column-width buffers.
 - The hash algorithm pick (blake2b vs sha256), recorded in the provenance header either way.
```
