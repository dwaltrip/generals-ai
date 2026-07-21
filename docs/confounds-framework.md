# The confounds framework: how behavior changes affect checkpoints and recorded numbers

*Date: 2026-07-08. Status: evergreen reference.*

*Update 2026-07-20: the impact vocabulary was renamed (formerly coded E1 / E2 break / E2 repair / E3 ruler / E3 task). The rename map is in [7.20-3](2026-07/7.20-3-goldens-naming-reference.md).*

Some code changes draw a boundary in time: checkpoints and recorded numbers from before the change were produced under the old code, so their validity for any later use has to be checked rather than assumed. The code in question is whatever trains, runs, or measures the model (obs building, forward semantics, targets, loss, metric definitions, eval protocol).

This doc classifies these changes. For any use of an old artifact (a checkpoint or a recorded number), the classification gives a verdict: clean, confounded, incomparable, or valid with a caption. Classifying takes two steps, one per section below:

1. **Impacts** — what the change does to checkpoints and numbers, read off the surface table's three tests (§1).
2. **Verdict** — cross the change's impacts with the *activity*: what is being done right now, with which artifacts (§2).

## The two contracts

Every confound is a break of one of two implicit promises.

**The model contract**: a checkpoint behaves the same today as it did at training time. The contract holds as long as input construction and the forward pass still compute what they did when the checkpoint was trained. A break shows up when an old checkpoint runs on current code: the checkpoint plays below its real strength. Any head-to-head against a newer checkpoint then mixes ability with the mismatch. Re-measuring can't remove the confound, because the damage is upstream of measurement. The cures are structural: gate the old behavior, retire the checkpoint, or fine-tune it (§3).

**The measurement contract**: two recorded numbers with the same name mean the same thing. The contract holds as long as everything that produces the number is stable: metric definitions, label definitions, eval protocol, and the game itself. A break shows up when a number recorded before the change is compared with one recorded after: the models on both sides may be perfectly healthy, but the gap between the numbers mixes model difference with measurement difference. The cure is re-measurement: run the old checkpoint under current conditions and compare fresh numbers (§3 notes the limits).

A change can also break neither contract: one that alters only how new checkpoints are trained (a new learning rate, a different data mix), touching neither the runtime code nor anything that produces a number. Later checkpoints are simply different models, and comparisons across the change stay valid. In §1's terms: a `recipe-change` without a `contract-break`.

## Artifacts: run versus read

The two artifact types differ in how they are used after the boundary, and that difference is what pairs each with a contract.

A **checkpoint** is stored weights that later code re-executes. Whatever code is current at use time does the running, so a checkpoint is governed by the model contract.

A **recorded number** is a stored measurement that is only ever read back. No code re-runs it, and its meaning was fixed by the pipeline that produced it, so a number is governed by the measurement contract. This project records three kinds:

- **Training curves**: per-epoch losses and head metrics. Pipeline: the model run on val data, label definitions, loss and metric definitions.
- **Gameplay-eval numbers**: win rates, ratings, head-to-head results. Pipeline: the game sim, the live model stack, sampling and protocol, the map set, the opponent pool.
- **Analysis numbers**: probe scores, head-vs-rule gaps, dump-derived metrics. Pipeline: dumps or fresh forward passes, label definitions, metric definitions.

Every pipeline splits into two parts. The **subject** is the checkpoint and its runtime path (input construction plus the forward pass): the thing the number is about. The **apparatus** is everything else that shapes the value: label and metric definitions, protocol, the game. Together they fix the number's meaning: how does this subject score under this apparatus? A subject change moves the value, which is what the number is for. An apparatus change alters the question, and that is what breaks comparability.

Recorded dumps sit between the two artifact types: data that is read, but that new analysis code then executes against. Wrinkle (d), at the end of §2, covers them.

## 1. Surfaces — what changed, and which impacts follow

Each surface is a kind of code or configuration that can change. The three test columns say what a change to it alters, and each column feeds one family of impacts: a change's impact set is its set of ✓s.

- **Alters training?** Holding the replay corpus and the seed fixed, would a training run launched after the change produce different weights (systematically, not as run-to-run noise)? A ✓ means checkpoints trained after the change are genuinely different models: `recipe-change`. The test covers everything from data selection through the optimizer, and excludes code that executes during training without shaping the weight updates (logging, dump cadence).
- **Alters the model's runtime path?** With weights held fixed, does the change alter the function from game state to network outputs (input construction plus the forward pass)? A ✓ means running a checkpoint across the boundary changes its behavior: `contract-break` or `contract-repair`, depending on direction (below). The endpoint matters: sampling temperature turns network outputs into a chosen action, downstream of the network, so it is part of the apparatus and belongs to the third column.
- **Alters what numbers mean?** Does the change alter the measurement apparatus of some kind of recorded number: its ruler (metric definitions, label definitions, eval protocol) or its world (game rules, or the information the game exposes to the model)? A ✓ means same-name numbers of the affected kinds stop comparing across the boundary: `metrics-change` for ruler changes, `mechanics-change` for world changes. The affected kinds are the ones whose pipeline contains the changed code (cells note them where it isn't obvious).

One conditional cuts across the third test: obs code is part of the subject, but it also sets how much of the world the model can see. An obs change therefore alters number meaning only when it changes the information available (a leak fix, a staleness fix), not when it re-encodes the same information. Wrinkle (a), at the end of §2, works through this case.

| Surface | Alters training? | Alters the model's runtime path? | Alters what numbers mean? | Example |
|---|---|---|---|---|
| Obs build, shared path | ✓ | ✓ | only if information changes (wrinkle (a)) | incident 1 (§4) |
| Obs build, live path only | – | ✓ (live path only) | only if information changes (gameplay-eval numbers) | incident 2 (§4) |
| Forward semantics (same weight shapes) | ✓ | ✓ | – | (hypothetical: changing a normalization constant) |
| Targets / labels | ✓ | – | ✓ (ruler: label-scored metrics and curves) | incident 3 (§4) |
| Loss definition | ✓ | – | ✓ (ruler: loss-type metrics) | soft targets, loss-weight changes |
| Eval protocol / metric definitions | – | – | ✓ (ruler) | new map set, sampling temperature |
| Hyperparameters / data mix | ✓ | – | – | learning rate, data-mix change |
| Arch shape / channel count | ✓ | fails loudly at load | – | ungated channel additions |
| The game itself | ✓ | – | ✓ (world) | sim-rule changes |

Two rows are special:

- **Shape changes** fail loudly at `load_state_dict(strict=True)`. An old checkpoint cannot silently degrade on mismatched arch code, because it refuses to load at all. The change instead poses a support decision: which old checkpoints stay loadable. The rest of this doc concerns confounds of the silent kind.
- **Game changes** are the pure world case: the game itself changed, so recorded numbers from the two eras describe different games and stay incomparable. Comparable numbers require re-measuring both sides on today's game.

The two model-contract impacts differ only in direction:

- **`contract-break`**: the change pushes the runtime path *away* from training conditions. Old checkpoints get inputs or computation they never trained with.
- **`contract-repair`**: the change brings the runtime path back *into* line with training. The contract was broken all along, and the fix ends the breach. The damage lands on the old era's numbers instead: they measured the broken behavior.

**Gating** is what makes a runtime-path change safe: the old behavior stays available behind a config switch keyed to the checkpoint, so each checkpoint's stored config selects the semantics it trained with. An in-place, ungated change is the kind that breaks the contract.

One note on the measurement side: `mechanics-change` rarely means generals.io itself changed. More often it is our implementation of the game (a sim-rule fix, a parity correction), which changes the world the bot experiences all the same.

**The row easiest to misread is targets / labels.** A label change feels like it should corrupt old checkpoints, but the tests say otherwise. Walking the three cells:

- Training ✓: checkpoints trained after the change are genuinely different models. The gradients flowed differently, including through any trunk the heads share (wrinkle (e)).
- Runtime path –: a frozen old checkpoint plays exactly as it always did, because label code never executes at inference. No contract-break.
- Numbers ✓ (ruler): the label definition is part of the measurement apparatus. Scoring an old head against the new labels is valid but answers the new question (caption it), and recorded label-scored curves stop comparing across the boundary.

## 2. Activities × impacts — the verdict table

Six activities exhaust the ways artifacts get used: checkpoint vs checkpoint, a checkpoint analyzed alone, number vs number, a checkpoint rerun against its own record, a checkpoint continued in training, and one artifact read or deployed alone. "Old" and "new" mean before and after the boundary: an artifact's *vintage*. A single change often carries several impacts at once (its set of ✓s in §1): apply each impact's column, and the strictest verdict governs.

| Activity | recipe-change | contract-break | contract-repair | metrics-change | mechanics-change |
|---|---|---|---|---|---|
| **1. Old vs new ckpt, both run now** | clean — a genuine model difference, including the recipe change's effect | **confounded** — the old side mixes ability with mismatch | clean | clean | clean |
| **2. Old ckpt analyzed now** | – | **confounded** if the analysis walks the changed runtime path | clean | the number is valid but measures the *new* definition — caption it | valid (today's game) |
| **3. New number vs old recorded number** | comparable — and measures the recipe change | see wrinkle (a) | **incomparable** — the old numbers were recorded during the breach | **incomparable** — curable by re-measuring | **incomparable** — permanently; only re-measured numbers compare |
| **4. Old ckpt rerun vs its own old number** | eval numbers reproduce; rerunning *training* is a new experiment | fails to reproduce — expected | fails — expected, and the new number is the truer one | fails — expected | fails — expected |
| **5. Resume / fine-tune old ckpt now** | the result is a cross-recipe hybrid — legitimate, label it | starts out-of-distribution; continued training *adapts* it (a cure) | fine | its curves kink at the resume point for definition reasons, not learning reasons | trains on today's game |
| **6. Read or deploy one artifact alone** | caption: which recipe | caption: contract status on current code | caption: that era's numbers are systematically off | caption: which definition | caption: which game era |

Row 4 carries the biggest practical trap: a reproduction gap has four possible causes in this table (five if the rerun is a training run under a changed recipe), so a surprising "this doesn't reproduce" must be attributed to a known boundary before being read as a regression or as nondeterminism.

### Wrinkles that don't fit in cells

- **(a) Native-vs-native numbers under an input change.** If each era's number was measured in-contract (old checkpoint on old code, new on new), both faithfully measure "how good was this model in its environment." They compare as recipe quality *only if* the input change didn't alter the information available. A leak fix does alter it (the old game was easier), which pushes this case into the mechanics-change column.
- **(b) Same-side comparisons during a shared breach.** Two old checkpoints compared today are *equally* out of contract, so the relative read plausibly survives, but only on the unverified assumption that both are equally sensitive to the mismatch. (Head-to-head evals run while incident 2's live-path bug (§4) was still in place have exactly this status: the pre-fix era of a contract-repair is such a breach.)
- **(c) Boundary-spanning checkpoints.** A run resumed across a boundary is neither vintage. A checkpoint census needs a "hybrid" category, not just pre/post.
- **(d) Frozen artifacts, not just frozen numbers.** Recorded dumps (e.g. per-epoch val dumps) bake the era's label definitions into *data*. Recomputing metrics on an old dump with new analysis code mixes eras inside a single pipeline — a metrics-change that hides in storage.
- **(e) Trunk coupling.** A targets or loss change (a recipe-change with measurement-side residue only, per §1) still moves *gameplay* across the boundary, because aux-head gradients shape the shared trunk that the policy head reads. Still not a confound, but "only the labels changed" does not imply "gameplay is identical across the boundary."
- **(f) The mismatch is symmetric.** Running a *new* checkpoint on *old* code (e.g. checking out an old commit) breaks the same contracts in the other direction. What matters is vintage mismatch, not age.

## 3. Cures — one set per impact

- **recipe-change:** nothing to cure. Record it so model differences stay attributable.
- **contract-break:** three cures — gate the old behavior (§1's gating note), declare the affected checkpoints degraded or unsupported (a supported-checkpoints policy call), or fine-tune them into the new contract. Re-running measures the confound but does not remove it.
- **contract-repair:** nothing to cure going forward. The old era's numbers inherit a measurement-contract impact.
- **metrics-change / mechanics-change:** re-measure the old artifact under current conditions, which works whenever its model contract holds. Where re-measuring is impossible, the old number keeps a permanent caveat.
- **All of them:** recording the boundary (date, surface, impacts) cures nothing by itself, but every verdict in §2 is computable only when the boundary is known to exist. An unrecorded boundary silently turns the confounded cells above into wrong conclusions.

## 4. Worked examples — three boundaries from this project (2026)

1. **Obs-content fixes (2026-05-21 and 2026-05-27).** Three ungated fixes changed existing obs channel values in place, in the shared path: "Close two obs-tensor fog-of-war leaks" (`2e6b2df`) and "Subtract expected production from army_delta channels" (`5a2beb0`) on 05-21, then "Replace game_progress channel with timestep (fix future-info leak)" (`e04cad6`) on 05-27, which replaced a channel that leaked the true game length with a fixed-divisor timestep.
2. **Stale-live-obs fix (2026-06-07).** "Fix one-tick-stale obs in live inference path" (`bc4bb19`, with same-day companion `49b335d` fixing the eval-bot's world model): the live path double-seeded tick-0 history, so every live encode saw the previous tick's board. The training-data path never had the bug.
3. **Elim-target tick-seam fix (2026-06-18).** "Add bc/player_status; fix the obs/alive tick seam" (`4aace8b`): the alive/present masks moved from `death > t` to `death >= t`, ungated, on the targets side only (obs channels untouched). The elim head's death-frame target became `dt = 0`.

| Incident | recipe-change | model-contract impact | measurement-contract impact |
|---|---|---|---|
| 1. Obs-content fixes | yes — later models train on cleaner obs | **contract-break** — earlier checkpoints see different inputs on current code | mechanics-change — the leaks made the earlier game easier |
| 2. Stale-live-obs fix | no — the training path was never wrong | **contract-repair** | mechanics-change — earlier gameplay-eval numbers measured a blinder game (the companion eval-bot fix adds a metrics-change: the opponent is part of the protocol) |
| 3. Elim-target fix | yes — labels, and gameplay via trunk coupling | none — label code never runs at inference | metrics-change — elim metrics, training curves, and old heads scored on the new labels |

## Completeness

The enumeration is believed complete in this sense:

- the six activities exhaust the ways the artifact types can be used (alone, paired, or continued),
- the impacts derive from §1's three tests, one column per impact family, and
- each wrinkle is a composition of base cases rather than a new one.

A change that resists classification here is a reason to extend this doc.
