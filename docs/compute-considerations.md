# Compute Considerations

**Status:** Topical reference doc. Updated in place as the project's compute picture sharpens.
**Updated:** 2026.05.07
**Companion:** [`deepnash-tpu-analysis.md`](./deepnash-tpu-analysis.md) (the v3 baseline argument for DeepNash).
**Pulls together:** material previously scattered across `2026-05/5.02-3-initial-notes-and-decisions.md` §5, `2026-05/5.02-6-strakam-paper-takeaways.md` §2.5, and `2026-05/5.02-7-deepnash-summary-and-implications.md` §1.8.

---

## TL;DR

- Hobby-budget cloud H100 rental ($2–10/hr) is the default for runs that need real artifacts. Local M1 is for development, not full training runs.
- BC at current data scale (~135k raw replays) is estimated at **~30–60 H100-hours per run**, ~$60–600 cloud rental per run. Mini-sweep variants (e.g. width sweep) run at ~5–10% of full duration.
- Phase 1 budget posture is exploratory and decided as the project progresses — see §4. Phase 2 (PPO self-play) is a separate downstream decision, not a committed plan.
- **Channel count is not the binding compute lever.** First-conv FLOPS are 1–3% of total. The actually-binding levers are inner pyramid width, spatial extent, and dataset size. Channel count matters for engineering complexity and inference latency, not training compute.
- DeepNash trained on roughly 10⁴ × Strakam's compute (estimated). Architecture transfer is sound; what's uncertain is how much of the architecture's capability we can extract on a tiny fraction of that compute.

---

## 1. Purpose & scope

Reference doc for compute decisions on this project. Consolidates reference points (prior art compute scales, hardware specs, cost estimates) and captures the reframing the project arrived at: that channel count is not the binding compute constraint we initially thought it was.

This doc is intended to be **evergreen-ish** — updated in place when the picture changes. It currently predates any real training runs, so all numbers are estimates with explicit reasoning.

**Update triggers:**
- First BC training run completes → real wall-clock + throughput numbers replace estimates
- M1 MPS throughput is empirically benchmarked → §3 recommendation gets sharper
- Phase 2 (self-play) design lands → §6 gets concrete instead of extrapolated
- Major architecture decisions (e.g. inner pyramid width) shift the per-sample cost model

---

## 2. Reference points from prior art

### Hardware peak throughput (BF16, no sparsity)

| Hardware | TFLOPS / chip | Notes |
|---|---|---|
| TPU v3 | ~123 | DeepNash baseline (per `deepnash-tpu-analysis.md`) |
| TPU v4 | ~275 | |
| H100 SXM5 | ~990 | Cloud rental target |
| M1 Max GPU (32-core) | ~22 | FP16; 64GB unified memory variant |
| M1 Ultra GPU (64-core) | ~42 | FP16 |

Implied conversions: **1 H100 ≈ 8 TPU v3 chips ≈ 25–45 M1 (Max/Ultra) GPUs** at peak. Real-world throughput is typically 30–50% of peak depending on architecture, batch size, and data loading overhead.

### Strakam (1v1 generals)

- 1× H100, ~36 hours total wall clock
- ~3 hours behavior cloning, ~33 hours self-play
- ~16k filtered 1v1 replays, ~25 input channels, ~18×18 board
- The paper's published agent reaches top 0.003% of the 1v1 ladder

### DeepNash (Stratego)

- 768 + 256 = 1,024 TPU "nodes" (paper §42)
- Per Sebulba convention, "node" = host of 4 chips, so this is **~4,096 chips total**
- 7.21M learner steps × 768-trajectory batch (paper §32)
- TPU generation **baseline v3** (~80% confidence; see companion analysis doc)
- Wall-clock duration not reported in paper; presumably multi-week given scale
- 10×10 board, 82-channel input

### Compute asymmetry

DeepNash's ~4,096 v3 chips ≈ **~510 H100-equivalents of peak compute**, multiplied by an unknown but presumably multi-week wall clock. Order-of-magnitude estimate: DeepNash's full training was on the order of **10⁴× Strakam's full 36-hour H100 run**.

This matters for architecture transfer: we're inheriting DeepNash's network design (Pyramid Module, 256/320 inner widths) but training on a fraction of a percent of their compute. The architecture is sound, but capability extraction at our compute scale is the real question.

---

## 3. Hardware options for this project

### Cloud H100 rental ($2–10/hr)

The default for runs that need to count. Major cloud providers (Lambda, RunPod, Vast.ai, AWS) all rent H100s at the lower end of that range; AWS sits at the high end. Per-run costs depend on architecture and training duration; see §4 for the project budget posture and per-run estimates.

### Local Apple Silicon (64GB M1 Max or Ultra)

- ~25–45× slower than 1 H100 in raw peak FLOPS
- Implies **~30–110 days per full BC run** at our current dataset size — *theoretically* feasible, but iteration loop is brutal: a single hyperparameter sweep would take a year, and recovering from a bug discovered partway through a run wipes out a month
- PyTorch MPS coverage has improved substantially but isn't 100%; some custom ops still fall back to CPU and tank throughput
- **Pre-investment check recommended**: a 30-min PyTorch MPS benchmark on a small U-Net before designing local-dev workflow around it

### Recommended split

- **Local M1**: development, debugging, parser sanity checks, training-loop validation, small-subsample iteration
- **Cloud H100**: any run intended to produce a real model artifact

This minimizes spend (only pay for runs that count) while keeping the iteration loop fast (most work happens locally on whatever subsample fits in single-digit hours).

---

## 4. BC compute estimate (Phase 1)

### Budget posture

Phase 1 BC budget is exploratory and decided as the project progresses. Working baseline: **~$500 total cloud spend on BC as a soft upper-bound**, hoping for less if achievable. The intent is to leave room for Phase 2 (PPO self-play, see §6) without committing to it, and to scale up only if early results suggest deeper investment is warranted.

In practice:
- Mini-sweeps and small partial training runs are cheap (tens of dollars each). Run freely.
- Full BC runs are the meaningful budget items — be deliberate about when to commit one.
- If first results are exciting and suggest more iteration is worth doing, willing to expand.
- If early results are underwhelming, scale back rather than throwing more compute at it.
- Per-run cost estimates below are factual planning inputs; total spend depends on how many runs end up being warranted.

If budget pressure shows up after early sweeps, look for ways to squeeze more out of available compute — see §7 for levers (smaller widths, mixed precision, dataset subsampling for iteration runs).

### Sample-volume math (current ~135k raw replays)

The replay DB's `turns` field stores half-turns (one half-turn = one move = one (obs, action) sample). Each filtered replay yields one or two strong-player trajectories depending on cohort.

Real game-length distribution for the 8-player FFA cohort (from `replay-collector/scripts/analyze_turn_stats.py`):

| Stat | Half-turns |
|---|---|
| Mean | 632 |
| Median | 563 |
| p25 / p75 | 433 / 743 |
| p10 / p90 | 342 / 974 |

Per-replay sample count derivation:

```
Game length (mean):       ~632 half-turns
Strong-player trajectory: ~60% of game length [^trajsurv] = ~380 half-turns
Per replay (mixed):       60% × (1 trajectory × 380)
                        + 40% × (2 trajectories × 380) [^twostrong]
                        ≈ ~530 samples per filtered replay
```

[^trajsurv]: Working assumption — strong-player trajectories tend to span ~60% of game length on average across various final placements. Easily validatable post-parser by joining trajectory length with placement metadata. Plausible range: 50–70% across the strong-player cohort.

[^twostrong]: Cohort statistic — ~40% of filtered 8-player FFA games have ≥2 qualifying strong-player trajectories.

Total raw samples at current 135k:

```
135k raw × ~60% filter survival = ~81k filtered replays
× ~530 samples/replay
≈ ~36–50M raw (state, action) pairs
```

With augmentation (effective independence multipliers, not theoretical maxima):
- **Board symmetry [^bs]:** ~2–4× effective (theoretical 4× for rectangular maps, 8× for square)
- **Slot permutation [^sp]:** ~2–3× effective

[^bs]: Board symmetry: rectangular maps have D₂ symmetry (4 transformations: identity, horizontal flip, vertical flip, 180° rotation); square maps have D₄ (8 transformations including 90° rotations and diagonal flips). The theoretical multiplier is exactly determinable once we measure the dataset's map-aspect distribution. The *effective independence* multiplier — how much new training signal augmented samples actually provide — is smaller because augmented samples are correlated.

[^sp]: Slot permutation theoretical max is 7! = 5040, but most permutations produce highly-correlated samples. The effective multiplier is hard to measure precisely and is mostly about how much "opponent identity confusability" the network exploits during training. Likely only validatable indirectly via held-out eval performance.

Net effective sample-views during ~5 epochs of training: **~1–2B** at current 135k. At the achievable ~270k upper end (continued scraping is straightforward), roughly doubles to ~2–4B.

### Per-sample compute vs Strakam

| Factor | Ratio | Notes |
|---|---|---|
| Spatial extent | ~2× | 25×25 vs ~18×18 |
| Input channels | ~4× | ~99 vs ~25 — but first conv is small fraction of total FLOPS |
| Inner pyramid width | 1× (assumed) | Inherited 256/320; see "Softness in the 30–60 anchor" subsection below |
| **Net per-sample** | **~2×** | Dominated by spatial; channels matter little |

### Headline estimate

**~30–60 H100-hours per BC run, ~$60–600 at $2–10/hr.** Mini-sweep variants run at ~5–10% of full duration each — a few hours and tens of dollars per variant.

**Calibration check**: Strakam's 3-hour BC × ~10× sample volume (135k vs 16k filtered, with our larger per-game sample yield) × ~2× per-sample ≈ 60 hours, the upper end of our planning range. Order-of-magnitude only.

### Softness in the 30–60 anchor

The 30–60 figure is calibrated to Strakam's 3-hour BC at *presumed* inherited widths (256/320). Strakam's paper is silent on actual widths used; the published `network.py` is a 4×4 toy scaffold, not production code. So this calibration has a circularity worth flagging.

**Two scenarios:**

- **Scenario A — Strakam used inherited widths.** Our extrapolation is solid. Real H100 utilization for this architecture is ~20–30% of peak BF16 throughput. 30–60 H100-hrs/run is a reasonable estimate.

- **Scenario B — Strakam quietly used smaller widths (e.g. 128/192).** Their 3 hours achieved ~4× fewer FLOPs per sample than our calibration assumes. We'd be overestimating. Real wall-clock at 256/320 could be 60–120 H100-hrs/run; at smaller widths from the planned width sweep (see `network-architecture-design.md` §3.3), 8–30 H100-hrs/run.

We don't know which is true. **The first real BC run replaces estimate with measurement** — at that point we know actual H100 utilization, throughput, and per-epoch wall-clock for our specific architecture.

What the budget posture is robust to (within ±2× of the anchor):
- Order-of-magnitude budget viability — a few hundred dollars covers BC iteration, regardless of which scenario is true
- Mini-sweep cost as a small fraction of full-run cost (the *ratio* doesn't depend on the calibration)
- The width-sweep approach itself (since it explicitly sweeps over the widths in question)

What's *less* robust:
- Precise count of full BC runs achievable within any specific dollar ceiling
- Exact dollar cost of the mini-sweep

### Implications for dataset sizing

At ~135k raw replays we're already in a comfortable BC range. The achievable upper end is ~270k+ via continued scraping (community channels, deeper archive work). The ~1–2B effective sample-views derived above is already past the threshold where pure BC tends to saturate — at this scale, doubling the dataset typically yields ~5–15% validation improvement, not transformative gains.

The strategic value of *more raw replays* isn't in raising the sample count further — it's in enabling stricter quality filtering. With 135k raw, "≥1 strong player in lobby" is about as restrictive a bar as we can comfortably afford. With 270–400k raw, we could demand "all 8 players above threshold" or other aggressive cuts and still end up with a comparable filtered training set. Beyond ~400k raw, training time starts to bind before learnability does, so the marginal return on additional scraping diminishes.

For ballpark planning, the project currently baselines a filtered training set in the **~36–50M raw sample range, ~1–2B effective sample-views with augmentation**. Wall-clock estimates in this section scale roughly linearly with total sample-views — re-estimate if the baseline shifts substantially (much stricter filter producing fewer samples, or a much larger raw corpus enabling more aggressive subsetting).

---

## 5. What actually scales with our design — the key reframe

The instinct from earlier obs-tensor design sessions was that channel count was a major compute concern — that adding the 9-channel `last_seen_owner` block, or 7 `opp_N_has_seen` channels, or extending dense history N from 7 to 12, would meaningfully eat compute budget.

A back-of-envelope on the U-Net's per-pass FLOPS shows this isn't true.

### Why channel count is a small lever

A U-Net torso with the DeepNash configuration (outer C=256, inner C=320, ~10 conv layers, 3×3 kernels) does most of its work in the inner layers, where channel widths are fixed regardless of input. The first conv layer is the only place where input channel count materially affects FLOPS:

```
First conv: spatial × C_in × C_out × kernel² = 625 × 99 × 256 × 9 ≈ 142M FLOPS
Each inner conv: spatial × 256² × 9 ≈ 740M FLOPS at full resolution
Total torso: ~10-15 GFLOPS per forward pass
```

The first conv accounts for roughly **1–3% of total FLOPS**. Cutting one input channel saves ~1% of the first conv, which is ~0.01–0.03% of total compute. Cutting the 9-channel `last_seen_owner` block saves ~0.1–0.3%. Halving N (the dense history window) saves ~1–2%. These are real but minor.

### What actually binds

In rough priority order:

1. **Inner pyramid width** (256/320, inherited from DeepNash). This dominates per-sample compute — total network FLOPS scale roughly with width-squared, so cutting to 128/192 yields ~4× savings. **This is the lever the project hasn't pushed on.** Inheriting DeepNash's widths was a defensible starting point given the architecture transfer, but those widths were tuned for an environment with effectively unlimited compute. Generals.io has structurally simpler local interactions than Stratego (no piece types, simpler combat), so smaller widths may suffice. Worth examining empirically before scaling up.

2. **Spatial extent** (~25×25 = 625 cells). Mostly fixed by the game — 90% of FFA boards fall in the 600–900 tile range. The 10% outside this range could be filtered (or padded), but the savings are minor and would constrain map diversity.

3. **Dataset × perspectives × turns** (sample volume). Total wall-clock scales linearly. Levers: stricter filters (smaller corpus), fewer perspectives per game, or sub-sampling within trajectories (e.g. every 2nd timestep). The first two are also quality levers; the third is purely a compute trade.

4. *Channel count* — small compute lever; matters for other reasons (see below).

### What channel count actually matters for

Not training compute. But still:

- **Engineering complexity.** A 99-channel observation tensor with persistent memory state requires a non-trivial parser. The parser is already the highest-risk component of the project (per `5.02-3` §3) and additional channels expand the bug surface.
- **Dataloader throughput.** With more channels per observation, the dataloader does more work per sample. For BC at our scale, GPU compute is fast enough that loader throughput can become the bottleneck before the GPU does.
- **Live inference latency.** Currently de-prioritized for Phase 1 (the bot is being evaluated on output-move quality, not live ladder play), but if deployment becomes a goal, channel count materially affects per-inference latency on consumer hardware.
- **Marginal generalization risk.** More input parameters at the first layer to fit. Likely fine at our sample size, but a real consideration for tiny datasets.

### Implication for design discussions

When obs-tensor changes are being evaluated, the relevant cost questions are about engineering complexity and (later) inference latency, not training compute. The training-compute question is settled by the inner pyramid width and dataset choices, not by adding or dropping a few channels.

---

## 6. Phase 2 (self-play) — extrapolated

Self-play not yet planned in detail. Forward-extrapolated estimate from Strakam's numbers, scaled per `5.02-6` §2.5:

- Strakam self-play: ~33 H100-hours for 1v1
- FFA estimated 2–5× more compute. Drivers: more agents per game step (8 inferences per timestep vs 2 for 1v1), longer episodes (1800 vs ~500 turns), sparser reward signal (placement vs binary win/loss makes credit assignment harder), and a larger opponent pool space to converge against
- → **~70–170 H100-hours per Phase 2 run, ~$140–1,700 cloud rental**

**Whether to do Phase 2 at all is an open decision**, made post-Phase-1. Factors that will weigh into it:

- Phase 1 outcomes — does BC produce a model worth investing further in?
- Remaining budget appetite — tighter if Phase 1 chewed through allotment; looser if early Phase 1 results were exciting and motivated expansion
- Compute optimizations identified during Phase 1 — real H100 utilization, smaller widths from the sweep, more efficient data loading, etc.
- Broader project timing and motivation

The cost figures above are planning inputs to that decision, not a commitment. Numbers will sharpen substantially once Phase 2 design lands; treat the above as an anchor, not a target.

---

## 7. Cost-control levers (tentative, only if needed)

If compute pressure shows up, these are the levers worth considering, ordered roughly by impact:

- **Reduce inner pyramid width** (256/320 → 128/192): ~4× compute savings, biggest single lever
- **Mixed-precision training** (BF16 / FP16): typically free 1.5–2× on H100
- **Subsample dataset for iteration runs**: full data only for milestone runs
- **Reduce inner pyramid depth**: smaller saving, more architectural risk
- **Reduce history window N**: small compute savings, but simplifies the parser

These are not currently on the path; listed for reference if compute becomes binding.

---

## 8. Possible follow-up items

In rough priority order:

- **Inner pyramid width sensitivity** — the largest unexplored compute lever. Inheriting DeepNash's 256/320 was a defensible default; whether 128/192 (or even smaller) suffices for FFA's structurally simpler local interactions has never been examined. Could pay for itself many times over in iteration speed.
- **30-min PyTorch MPS benchmark** on a small U-Net to validate the M1 dev workflow before designing around it
- **Real BC throughput measurement** on the first training run — backs out actual TFLOPS realized vs peak, replaces the ~50% peak assumption baked into §4 with a measured number
