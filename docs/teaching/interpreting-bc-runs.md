# Interpreting BC training/probe runs — a worked example

A practical guide to reading the numbers from a behavioral-cloning training or probe run in this project. The framework — what each metric measures, what baselines to compare against, what patterns in the train/val curves signal — is general; the worked example is specific.

## What you'll know after working through this

- What cross-entropy loss measures and what the numbers mean intuitively
- Why we report top-1 accuracy alongside CE, and how the two can decouple
- Three baselines that anchor "did the model learn anything"
- The train-vs-val pattern as a generalization signal
- Why parameter count interacts with effective sample size to drive overfitting

The math is light. The interpretive habits are the point.

## The example we'll work from

A Mode B value-head probe of the `epoch_005.pt` checkpoint. The probe freezes the trunk and trains fresh value-head variants on cached features, asking *does the trunk's representation encode placement-relevant signal?*

| Artifact | Location |
|---|---|
| Probe script | `training/scripts/value_head_probe.py` |
| Probe run dir | `training/data/probes/20260521-110841-value-probe/` (gitignored; numbers reproduced below) |
| Probe summary | `<run dir>/probe_summary.json` (per-epoch curves per variant) |
| Probe plot | `<run dir>/probe_curves.png` (val loss + val top-1 over epochs) |
| Underlying training run | `training/data/runs/20260521-005930/` |
| Split manifest | `training/data/splits/poc_2kish_perspecs.json` |

Motivation context: the probe was triggered by the value-head dead-ReLU collapse described in `docs/2026-05/5.21-1-debugging-value-head.md`.

## Metric 1: cross-entropy loss

We report CE in **nats** (natural-log units; the default in PyTorch's `F.cross_entropy`). Bits would just divide by ln(2); the unit choice doesn't change anything substantive.

For a single sample with a one-hot target, CE reduces to `-log(p_correct)` — the negative log of the probability the model assigned to the true class. Lower is better; 0 is perfect.

Worked numerics for one sample, 8-class problem:

| Probability on true class | CE (nats) | Interpretation |
|---:|---:|---|
| 1.0 | 0.000 | perfect |
| 0.6 | 0.511 | reasonable hit |
| 0.125 | 2.079 | uniform — = `ln(8)` |
| 0.01 | 4.605 | confidently wrong |
| 0.001 | 6.908 | very confidently wrong |

The asymmetry matters: being confidently wrong is *much* worse than being slightly uncertain. A model that puts 0.01 on the truth one frame out of a hundred can have a worse average CE than a model that just predicts uniform every time. We'll see this exact pattern in our run.

## Metric 2: top-1 accuracy

For each sample, take the argmax of the predicted distribution; compare to the true class; average over the val set.

Top-1 ignores probability magnitude — a barely-confident right answer (0.13 on the true class, 0.12 on every other) counts equal to a confident right one (0.95 on the true class). It only cares which class had the highest score.

Why we report both:

- A model can have **good top-1, bad CE**: usually picks the right mode but is overconfident when wrong.
- A model can have **bad top-1, decent CE**: well-calibrated but stuck close to the prior (predicts something near the class frequencies regardless of input).
- The gap between the two tells you about *calibration*.

## Three baselines

You can't read numbers without anchors. Three "doing nothing" reference points, in order of increasing sophistication:

### Uniform head

The dumbest classifier: every class gets probability 1/N. For our 8-class placement: probability 0.125 for each.

- CE = `ln(N)` = `ln(8)` ≈ **2.08 nats**
- top-1 = 1/N = **0.125**

A randomly-initialized head sits approximately here before training. Our probe's "init" rows hover here: `gelu` init was 2.07, `status_quo` init was 2.09. Good sanity check that the random init isn't broken.

### Marginal head

Predicts the *class frequencies* observed in the data — the same fixed distribution every time, no input dependence.

The CE of this head equals the **entropy** of the class distribution, often called the *marginal entropy*. For a class distribution `p`, this is `-sum(p_i * log(p_i))` over classes with non-zero frequency.

For our val_probe (class fractions `[0.317, 0.293, 0.204, 0, 0.053, 0.133, 0, 0]`):

- Marginal H = **1.47 nats**

This is *the* most important baseline for a classification head. If your model's val CE is worse than the marginal H, it's doing worse than a model that ignores its input entirely and predicts class frequencies. That's the absolute floor for "the model is using its input usefully."

The top-1 of the marginal head equals the frequency of the **mode** — because argmax always picks the most-probable class, and the mode appears with frequency equal to its mass. For our val_probe: mode = 1st place, frequency 0.317. So marginal top-1 = **0.317**.

### Always-mode head

Deterministic: probability 1.0 on the mode, 0 elsewhere.

- top-1: same as marginal head, **0.317**
- CE: infinite in the pure form (`-log(0)` on any non-mode frame)

In practice we don't use this as a CE baseline; the marginal head's CE is the realistic floor. But "always-1st" is the simplest top-1 baseline to beat, and worth sanity-checking against.

A reproducible recipe for these baselines on any val split is `training/scripts/marginal_entropy.py` — it computes frame-weighted and perspective-weighted entropies from any manifest. The probe's val_probe sub-split has a *different* marginal than the full val set (1.47 vs 1.52) because random sub-sampling changes the class distribution.

## The generalization signature: train vs val

A run's interesting story is told by what happens to *both* curves, not either one alone.

Four patterns to recognize:

| Train | Val | Diagnosis |
|---|---|---|
| Drops to ≈0 | Stays high | **Memorization** — fit the training data, learned nothing transferable |
| Stuck near marginal | Stuck near marginal | **Under-capacity** or **under-trained** |
| Drops together | Drops together | **Working** — representation is generalizing |
| Drops fast | Rises | **Overfit** — should have early-stopped at the val minimum |

The actual numbers from our run:

```
status_quo:  train 2.09 → 0.001    val 2.09 → 5.61
no_relu:     train 2.19 → 0.001    val 2.10 → 4.47
gelu:        train 2.08 → 0.007    val 2.07 → 3.95
wide_conv:   train 2.00 → 0.0001   val 2.06 → 6.04
gap_linear:  train 2.26 → 1.24     val 2.19 → 1.57
```

The first four are textbook **overfit/memorization**: train collapses to near-zero, val gets *worse* than the uniform baseline (2.08). The val curve actively climbs across epochs — visible in `probe_curves.png`.

How can val CE exceed the uniform baseline? Because the model isn't just *wrong* — it's *confidently wrong*. Putting 0.99 on the wrong class costs you 4.6 nats per sample; a handful of those drag the average way up.

`gap_linear` is the outlier. Train loss doesn't collapse — it's still at 1.24 after 5 epochs. Val loss sits near the marginal floor (1.57 vs 1.47). It hasn't memorized. The next section is why.

## Capacity vs effective sample size

A head with N parameters trained on M independent samples is roughly OK when N << M. When N ≈ M or N > M, the model has the budget to memorize each training point individually.

The catch: "independent samples" isn't the same as "frames." In our run:

- Total training frames: **31,444**
- Distinct perspectives: **81**
- Frames per perspective (average): ~388

Frames within the same perspective are highly correlated — the board state at timestep t+1 differs from t by one move. The *effective* sample size sits closer to "81" (distinct trajectories) than "31,444" (raw frames).

Now look at param counts:

| Variant | Params |
|---|---:|
| `status_quo`, `no_relu`, `gelu` | 9,353 |
| `wide_conv` | 74,768 |
| `gap_linear` | **1,032** |

A head with 9k-75k params has more than enough budget to memorize 81 distinct perspective signatures — and it does. It learns "perspective X's trunk features look like *this* → predict class C", which is a perfectly valid explanation of the training data but generalizes to nothing.

`gap_linear` physically can't do that. With 1,032 params — a single linear layer over 128 globally-pooled channels (`128*8 + 8 = 1032`) — the head is forced to find an explanation that compresses across perspectives. Forced generalization.

This is why **linear probes** are the standard tool for measuring what a frozen representation knows: they remove the head's ability to cheat with raw capacity. If a linear probe over your frozen features extracts useful signal, the features encode that signal. If even a linear probe can't extract it, your representation didn't learn what you wanted.

## Putting it together: our run, fully read

| variant | params | val_loss | Δ marginal (1.47) | val_top1 | Δ always-1st (0.317) |
|---|---:|---:|---:|---:|---:|
| status_quo | 9,353 | 5.61 | +4.14 | 0.226 | −0.091 |
| no_relu | 9,353 | 4.47 | +3.00 | 0.260 | −0.057 |
| gelu | 9,353 | 3.95 | +2.48 | 0.199 | −0.118 |
| wide_conv | 74,768 | 6.04 | +4.57 | 0.283 | −0.034 |
| **gap_linear** | **1,032** | **1.57** | **+0.10** | **0.340** | **+0.023** |

Variant-by-variant:

- **`status_quo`** (architecture matching `epoch_005.pt`'s value head): val CE 5.61, much worse than uniform. Top-1 (0.226) is worse than always-guess-1st. Pure memorization; no transferable signal.
- **`no_relu`, `gelu`**: same shape, different activations. Same memorization signature. Removing or changing the ReLU doesn't "fix" anything here — they're all blown apart by the same overfit pattern.
- **`wide_conv`**: more capacity (75k params); even more aggressive memorization. Train loss reaches 0.0001 by epoch 5.
- **`gap_linear`**: doesn't memorize. Val CE 1.57 (only 0.10 above marginal) and val top-1 0.340 (+0.023 above always-1st). The +2.3 percentage points on top-1 is the only positive signal in the entire run.

The honest one-line read: *the linear probe (`gap_linear`) shows a barely-positive trunk signal; the spatial-head probes are uninterpretable in this run because they're overfitting before they can show what they'd extract. To get a clean answer for the spatial heads we'd need stronger regularization or early stopping.*

## A practical checklist when reading any BC training/probe run

In order, ask:

1. **Where does train loss end up?** Near 0 → memorized. Stuck near marginal → under-capacity or under-trained. Smoothly descending → working as expected.
2. **Where does val loss end up relative to the marginal floor?** Below = model is using its input usefully. Above = it's worse than predicting class frequencies (confidently wrong somewhere).
3. **Where does val top-1 end up relative to always-mode?** Catches "weak signal hiding behind bad calibration."
4. **What's the train-val gap?** Big gap = overfit. Small gap with both bad = under-capacity or wrong representation.
5. **Does param count look reasonable vs effective sample size?** Especially for probes: too many params will memorize before they generalize.

Apply this order every time and you'll get a usable read in 30 seconds.

## Reproducing the baselines

If you want to compute marginal entropy on a different split (or sub-sample of one), the recipe is in `training/scripts/marginal_entropy.py`. Core formula in plain Python:

```python
import math

# counts[i] = number of frames with placement class i (0=1st, 7=8th)
total = sum(counts)
p = [c / total for c in counts if c > 0]
H = -sum(pi * math.log(pi) for pi in p)
```

For top-1 baselines, the always-mode head's top-1 equals `max(counts) / total`. The frame-weighted vs perspective-weighted distinction (every frame counts once vs every perspective counts once) shows up in the script's CLI — for value-head purposes, frame-weighted is the apples-to-apples comparison with the model's loss.

---

## For the Claude session picking up this thread

If you're a Claude Code agent picking this up later, what follows is the handoff.

### What the user has internalized

- CE in nats; the `-log(p_correct)` intuition
- Top-1 vs CE decoupling (the confidently-wrong pathology)
- The three baselines (uniform / marginal / always-mode) and how to compute them
- Train-vs-val patterns (memorize / under-capacity / working / overfit)
- Capacity vs effective sample size as the lens for "why is this memorizing"
- Why linear probes are the standard tool for representation interrogation

### User context (carry-over)

The user is a lapsed Python expert with prior fine-tuning experience (YOLO-v8 on a puzzle-parsing task) — solid on the training loop, gappier on custom architecture design and ML theory. Plays generals.io at top-50 level so the game-side mental model is strong. Prefers discussion before implementation. They're learning AI/ML alongside the project; explanations can lean educational without being condescending.

### Concepts likely worth a deeper dive next

- **Weight decay** as a generalization knob: L2 in the loss vs decoupled (AdamW), what "high" weight decay looks like, when to reach for it.
- **Dropout**: mechanism, eval-mode interaction, where in a classifier head it typically goes.
- **Early stopping**: best-epoch model selection vs final-epoch reporting; how it functions as a regularizer; the patience knob.
- **Calibration** as a distinct property from accuracy: temperature scaling as the simplest fix; expected calibration error (ECE).
- **Linear probes in representation learning** more broadly: the CLIP / MoCo / SimCLR literature is the canonical home of this idiom; the same playbook applies here.
- **Bias-variance tradeoff** in classical statistical-learning terms; how it shows up in the train-vs-val gap and motivates regularization.

### Open methodology questions left from the run that grounded this doc

- The probe script's `VAL_MARGINAL_ENTROPY = 1.5195` constant is the *full* val-set marginal, not the val_probe sub-split's marginal. The sub-split has a different class distribution and a different floor (1.47 in this run). Should be computed per-run from the actual val_targets, not hardcoded.
- The probe summary reports only **final-epoch** val numbers. Best-epoch val is worth tracking too; for most variants in this run the best val happens at epoch 1, before overfit takes hold.
- No regularization knobs exposed in the probe CLI. Adding `--probe-weight-decay`, `--probe-dropout`, or honoring early-stop semantics would let the user empirically explore the "can a regularized spatial head beat `gap_linear`?" question.
- Sub-split missed 4th, 7th, 8th place classes in val entirely. Class-stratified sub-split would fix this. Currently the sub-split is random over perspectives.

### Pointers — code and data

| Path | What's there |
|---|---|
| `training/scripts/value_head_probe.py` | The probe harness. Variant classes at the top, `train_head_minibatched` for the loop, `evaluate_head` for the metric computation. |
| `training/scripts/marginal_entropy.py` | Computes the marginal-entropy baseline for any manifest. Worked example of how to derive a floor for a specific dataset. |
| `training/scripts/plot_run.py` | Plots for full *training* runs (not the probe). Same interpretive framework applies — useful comparison. |
| `training/data/runs/20260521-005930/` | The PoC training run that produced `epoch_005.pt`. Has `epochs.jsonl`, `batches.jsonl`, plots. Reading these is a good exercise. |
| `training/data/probes/20260521-110841-value-probe/` | The specific probe run analyzed in this doc. |
| `training/data/splits/poc_2kish_perspecs.json` | The split manifest used. |
| `docs/2026-05/5.21-1-debugging-value-head.md` | Diagnostic context for why this probe exists. The marginal-H = 1.52 number first appears there. |
| `docs/network-architecture-design.md`, `docs/2026-05/5.20-3-model-architecture-implementation.md` | Model architecture context — useful when "why is the trunk 128 channels" comes up. |
| `docs/papers/` | DeepNash and Strakam papers; the architectural lineage for this project. Not directly needed for the interpretive framework but worth knowing about. |

### What the user is likely to want next

They've been operating in a "session-by-session, fast iteration" mode. They just learned a framework rather than building a thing. Plausible next directions:

- Apply the framework to the *training run* `epoch_005.pt` came from — exercise to internalize.
- Discuss + possibly implement the methodology fixes for the probe (regularization, best-epoch tracking, per-run baselines).
- Continue the Phase 4 planning thread (value-head fix in spike vs Phase 4; cloud-retrain options).
- Deeper dive on any of the "concepts likely worth a deeper dive next" list.

Don't auto-assume; ask which direction interests them.
