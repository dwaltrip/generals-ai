# The confounds framework: how behavior changes affect checkpoints and recorded numbers

*Date: 2026-07-08. Status: evergreen reference.*

*Update 2026-07-20: the impact vocabulary was renamed (formerly coded E1 / E2 break / E2 repair / E3 ruler / E3 task). The rename map is in [7.20-3](2026-07/7.20-3-goldens-naming-reference.md).*

Some code changes draw a boundary in time: checkpoints and recorded numbers from before the change were produced under the old code, so their validity for any later use has to be checked rather than assumed. The code in question is whatever trains, runs, or measures the model (obs building, forward semantics, targets, loss, metric definitions, eval protocol).

This doc classifies these changes. For any use of an old artifact (a checkpoint or a recorded number), the classification gives a verdict: clean, confounded, incomparable, or valid with a caption (a few cells carry close variants). Classifying takes two steps, one per section below:

1. **Impacts** — what the change does to checkpoints and numbers, read off the surface table's three tests (§1).
2. **Verdict** — cross the change's impacts with the *activity*: what is being done right now, with which artifacts (§2).

## Terminology note

- **`labels`** — This term is used widely in this doc to refer to the training label values and how they are applied. In the code or other docs, this is often referred to as "targets".

## The two contracts

Every confound is a break of one of two implicit promises.

**The model contract**: a checkpoint behaves the same today as it did at training time. The contract holds as long as the checkpoint's **runtime path** (input construction, the forward pass, and the decode of outputs into moves) still computes what it did when the checkpoint was trained. A break shows up when an old checkpoint runs on current code: the checkpoint plays below its real strength. Any head-to-head against a newer checkpoint then mixes ability with the mismatch. Re-measuring can't remove the confound, because the damage is upstream of measurement. The cures are structural: gate the old behavior, retire the checkpoint, or fine-tune it (§3).

**The measurement contract**: two recorded numbers with the same name mean the same thing. The contract holds as long as everything that produces the number is stable: metric definitions, label definitions, eval protocol, and the game itself. A break shows up when a number recorded before the change is compared with one recorded after: the models on both sides may be perfectly healthy, but the gap between the numbers mixes model difference with measurement difference. The cure is re-measurement: run the old checkpoint under current conditions and compare fresh numbers (§3 notes the limits).

A change can also break neither contract: one that alters only how new checkpoints are trained (a new learning rate, a different data mix), touching neither the runtime path nor anything that produces a number. Later checkpoints are simply different models, and comparisons across the change stay valid. In §1's terms: a `recipe-change` without any other impacts.

## Artifacts: run versus read

The two artifact types differ in how they are used after the boundary, and that difference is what pairs each with a contract.

A **checkpoint** is stored weights that later code re-executes. Whatever code is current at use time does the running, so a checkpoint is governed by the model contract.

A **recorded number** is a stored measurement that is only ever read back. No code re-runs it, and its meaning was fixed by the pipeline that produced it, so a number is governed by the measurement contract. This project records three kinds:

- **Training curves**: per-epoch losses and head metrics. Pipeline: the model run on val data, label definitions, loss and metric definitions.
- **Gameplay-eval numbers**: win rates, ratings, head-to-head results. Pipeline: the game sim, the live model stack, sampling and protocol, the map set, the opponent pool.
- **Analysis numbers**: probe scores, head-vs-rule gaps, dump-derived metrics. Pipeline: dumps or fresh forward passes, label definitions, metric definitions.

Every pipeline splits into two parts. The **subject** is the checkpoint and its runtime path: the thing the number is about. The **apparatus** is everything else that shapes the value: label and metric definitions, protocol, the game. Together they fix the number's meaning: how does this subject score under this apparatus? A subject change moves the value, which is what the number is for. An apparatus change alters the question, and that is what breaks comparability.

The split cuts through the translation of outputs into a move. Output decoding mirrors the target encoding (fixed at training time), so it is part of the subject's runtime path. Move selection has no such mirror, as training doesn't choose individual moves (the loss consumes the whole scored distribution). Thus the selection policy (e.g. sampling temperature) is considered part of the apparatus. The underlying test, for any piece of inference-side code: does training have a counterpart that this code must stay consistent with? Code with such a counterpart is part of the subject, while code without one is part of the apparatus.

Recorded dumps sit between the two artifact types: data that is read, but that new analysis code then executes against. Wrinkle (d), at the end of §2, covers them.

## 1. Surfaces — what changed, and which impacts follow

Each surface is a kind of code or configuration that can change. Code changes are, of course, not limited to a single surface. In terms of the table below, the full impact of a change is the union of ✓s across the surfaces (rows) that it touches. An ungated obs-channel addition, for example, touches two rows: obs build (new channel content) and model shape (the input layer grows).

The three test columns say what a change to a given surface alters:

- **Test 1: Alters training?** Holding the replay corpus and the seed fixed, would a training run launched after the change produce different weights (systematically, not as run-to-run noise)? A ✓ means checkpoints trained after the change are different models: `recipe-change`. The test covers everything from data selection through the optimizer, and excludes code that executes during training without shaping the weight updates (logging, dump cadence).
- **Test 2: Alters the model's runtime path?** With weights held fixed, does the change alter the function the runtime path computes (game state in, the model's scored moves out)? A ✓ means running a checkpoint across the boundary changes its behavior: `contract-break` or `contract-repair`, depending on direction (below). One limiting case also earns a ✓: a change that makes the old weights unloadable, leaving no function to run at all (the model-shape row).
- **Test 3: Alters what numbers mean?** Does the change alter the measurement apparatus of some kind of recorded number: its ruler (metric definitions, label definitions, eval protocol) or its world (game rules, or the information the game exposes to the model)? A ✓ means same-name numbers of the affected kinds stop comparing across the boundary: `metrics-change` for ruler changes, `mechanics-change` for world changes. The affected kinds are the ones whose pipeline contains the changed code (cells note them where it isn't obvious).

One conditional cuts across the third test: obs code is part of the subject, but it also sets how much of the world the model can see. An obs change therefore alters number meaning only when it changes the information available (a leak fix, a staleness fix), not when it re-encodes the same information. Wrinkle (a), at the end of §2, works through this case.

| Surface | Alters training? | Alters the model's runtime path? | Alters what numbers mean? | Example |
|---|---|---|---|---|
| Obs build, shared path | ✓ | ✓ | only if information changes (wrinkle (a)) | incident 1 (§4) |
| Live inference path (obs build, action decode) | – | ✓ (live path only) | only if information changes (gameplay-eval numbers) | incident 2 (§4) |
| Forward semantics (same model shape) | ✓ | ✓ | – | (hypothetical: changing a normalization constant) |
| Training targets / labels | ✓ | – | ✓ (ruler: label-scored metrics and curves) | incident 3 (§4) |
| Loss definition | ✓ | – | ✓ (ruler: loss-type metrics) | soft targets, loss-weight changes |
| Eval protocol / metric definitions | – | – | ✓ (ruler) | new map set, sampling temperature |
| Training-only code/config | ✓ | – | – | learning rate, regularization (e.g. dropout), data-mix change |
| Model shape | ✓ | ✓ (loud: fails at load) | – | trunk width or depth change |
| The game itself | ✓ | – | ✓ (world) | sim-rule changes |

Two rows are special:

- **Shape changes** are contract-breaks with a built-in safety: the break is "loud". `load_state_dict(strict=True)` refuses mismatched weights, so an old checkpoint fails at load instead of silently degrading. The break still poses the support decision (which old checkpoints stay loadable), but it cannot produce misleading numbers (§2's note on loud breaks).
- **Game changes** are the pure world case: the game itself changed, so recorded numbers from the two eras describe different games and stay incomparable. Comparable numbers require re-measuring the old side on today's game. Like most impacts we expect to encounter, this mechanics-change is felt immediately.

  The row's training ✓ arrives later. Training uses the on-disk output of the replay-parser module (the `intermediate` directory), and a sim change has no effect on training until that data is regenerated. Upon regeneration, the earlier code change delivers its remaining impacts: the `recipe-change` (later runs train on the corrected data), and the `mechanics-change` now also reaches the data-derived numbers (curves and analysis numbers, whose val data and dumps describe the corrected world from then on). In practice the regeneration is its own boundary event, recorded with its own date.

The two model-contract impacts differ only in direction:

- **`contract-break`**: the change pushes the runtime path *away* from training conditions. Old checkpoints run under inputs, computation, or output decoding they never trained with. In the loud case, they simply get nothing at all, as the weights refuse to load.
- **`contract-repair`**: the change brings the runtime path back *into* line with training. The contract was broken all along, and the fix ends the breach. The damage lands on the old era's numbers instead: they measured the broken behavior.

**Gating** is what makes a runtime-path change safe: the old behavior stays available behind a config switch keyed to the checkpoint, so each checkpoint's stored config selects the semantics it trained with. An in-place, ungated change is the kind that breaks the contract.

One note on the measurement side: `mechanics-change` rarely means generals.io itself changed. More often it is our implementation of the game (a sim-rule fix, a parity correction), which changes the world the bot experiences all the same.

**The row easiest to misread is training targets / labels.** A label change feels like it should corrupt old checkpoints, but the tests say otherwise. Walking the three cells:

- Training ✓: checkpoints trained after the change are different models. The gradients flowed differently, including through any trunk the heads share (wrinkle (e)).
- Runtime path –: a frozen old checkpoint plays exactly as it always did, because label code never executes at inference. No contract-break.
- Numbers ✓ (ruler): the training label definition (targets) is part of the measurement apparatus. Scoring an old head against the new labels is valid but answers the new question (caption it), and recorded label-scored curves stop comparing across the boundary.

## 2. Activities × impacts — the verdict table

Six activities exhaust the ways artifacts get used: checkpoint vs checkpoint, a checkpoint analyzed alone, number vs number, a checkpoint rerun against its own record, a checkpoint continued in training, and one artifact read or deployed alone. "Old" and "new" mean before and after the boundary: an artifact's *vintage*. A single change often carries several impacts at once (its set of ✓s in §1): apply each impact's column, and the strictest verdict governs.

| Activity | recipe-change | contract-break | contract-repair | metrics-change | mechanics-change |
|---|---|---|---|---|---|
| **1. Old vs new ckpt, both run now** | clean — a genuine model difference, including the recipe change's effect | **confounded** — the old side mixes ability with mismatch | clean | clean | clean |
| **2. Old ckpt analyzed now** | – | **confounded** if the analysis walks the changed runtime path | clean | the number is valid but measures the *new* definition — caption it | valid (today's game) |
| **3. New number vs old recorded number** | comparable — and measures the recipe change | comparable only if the input change didn't alter the information available (wrinkle (a)) | **incomparable** — the old numbers were recorded during the breach | **incomparable** — cure: re-measure the old side under the current definitions | **incomparable** — cure: re-measure the old side on today's game |
| **4. Old ckpt rerun vs its own old number** | eval numbers reproduce; rerunning *training* is a new experiment | fails to reproduce — expected | fails — expected, and the new number is the truer one | fails — expected | fails — expected |
| **5. Resume / fine-tune old ckpt now** | the result is a cross-recipe hybrid — legitimate, label it | starts out-of-distribution; continued training *adapts* it (a cure) | fine | its curves kink at the resume point for definition reasons, not learning reasons | trains on today's game |
| **6. Read or deploy one artifact alone** | caption: which recipe | caption: contract status on current code | caption: that era's numbers are systematically off | caption: which definition | caption: which game era |

Row 4 carries the biggest practical trap: a reproduction gap has four possible causes in this table (five if the rerun is a training run under a changed recipe), so a surprising "this doesn't reproduce" must be attributed to a known boundary before being read as a regression or as nondeterminism.

For a loud break (the model-shape row), the contract-break column's confounded cells are unreachable rather than hazardous: the old checkpoint cannot run at all, so the comparisons those cells warn about can never be performed. What remains is the support decision recorded at the boundary.

### Wrinkles that don't fit in cells

- **(a) Native-vs-native numbers under an input change.** If each era's number was measured in-contract (old checkpoint on old code, new on new), both faithfully measure "how good was this model in its environment." They compare as recipe quality *only if* the input change didn't alter the information available. A leak fix does alter it (the old game was easier), which pushes this case into the mechanics-change column.
- **(b) Same-side comparisons during a shared breach.** Two old checkpoints compared today are *equally* out of contract, so the relative read plausibly survives, but only on the unverified assumption that both are equally sensitive to the mismatch. (Head-to-head evals run while incident 2's live-path bug (§4) was still in place have exactly this status: the pre-fix era of a contract-repair is such a breach.)
- **(c) Boundary-spanning checkpoints.** A run resumed across a boundary is neither vintage. A checkpoint census needs a "hybrid" category, not just pre/post.
- **(d) Frozen artifacts, not just frozen numbers.** Recorded dumps (e.g. per-epoch val dumps) bake the era's label definitions into *data*. Recomputing metrics on an old dump with new analysis code mixes eras inside a single pipeline — a metrics-change that hides in storage.
- **(e) Trunk coupling.** A targets or loss change (per §1: a recipe-change plus a metrics-change, no contract impact) still moves *gameplay* across the boundary, because aux-head gradients shape the shared trunk that the policy head reads. Still not a confound, but "only the labels changed" does not imply "gameplay is identical across the boundary."
- **(f) The mismatch is symmetric.** Running a *new* checkpoint on *old* code (e.g. checking out an old commit) breaks the same contracts in the other direction. What matters is vintage mismatch, not age.
- **(g) Checkpoints inside the apparatus.** An opponent pool puts checkpoints on the apparatus side, and the model contract governs them there too. A contract-break then reaches gameplay-eval numbers by a hidden route: the pool's spec is unchanged, but its checkpoints play below strength on current code, so numbers recorded after the boundary are measured against weakened opponents. That is a metrics-change no surface row generates (nothing in the apparatus's code changed), the same shape as (d): an artifact of one type hiding inside the other's pipeline. Gating closes the route, because gated opponents keep the semantics they trained with. An ungated break leaves ratings incomparable across the boundary, like any other metrics-change.

## 3. Cures — one set per impact

- **recipe-change:** nothing to cure. Record it so model differences stay attributable.
- **contract-break:** three cures — gate the old behavior (§1's gating note), declare the affected checkpoints degraded or unsupported (a supported-checkpoints policy call), or fine-tune them into the new contract. Re-running measures the confound but does not remove it.
- **contract-repair:** nothing to cure going forward. The old era's numbers inherit a measurement-contract impact.
- **metrics-change / mechanics-change:** re-measure the old artifact under current conditions, which works whenever its model contract holds. Where re-measuring is impossible, the old number keeps a permanent caveat.
- **All of them:** recording the boundary (date, surface, impacts) cures nothing by itself, but every verdict in §2 is computable only when the boundary is known to exist. An unrecorded boundary silently turns the confounded cells above into wrong conclusions.

## 4. Worked examples — three boundaries from this project (2026)

### 1. Obs-content fixes

- What changed: existing obs channel values changed in place, in the shared path (three ungated fixes).
- Commits:
    - "Close two obs-tensor fog-of-war leaks" (`2e6b2df`, 2026-05-21)
    - "Subtract expected production from army_delta channels" (`5a2beb0`, 2026-05-21)
    - "Replace game_progress channel with timestep (fix future-info leak)" (`e04cad6`, 2026-05-27)

The third commit replaced a channel that leaked the true game length with a fixed-divisor timestep.

### 2. Stale-live-obs fix

- What changed: a staleness fix in the live-only obs path.
- Commit: "Fix one-tick-stale obs in live inference path" (`bc4bb19`, 2026-06-07), with same-day companion `49b335d` fixing the eval-bot's world model.

The live path double-seeded tick-0 history, so every live encode saw the previous tick's board. The training-data path never had the bug.

### 3. Elim-head target tick-seam fix

- What changed: a one-tick shift in the labels' definition of death.
- Commit: "Add bc/player_status; fix the obs/alive tick seam" (`4aace8b`, 2026-06-18)

On a player's death tick, the alive/present masks now mark them still alive (`death > t` became `death >= t`, matching the obs's pre-event board snapshot), ungated, on the targets side only (obs channels untouched). The elim head's target derives from those masks, so its values shifted with the change.

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
