# Network Architecture Design — v1 Working Spec

**Date:** 2026.05.06
**Status:** **Skeleton.** Major decisions tentative; many mid-/low-level details deferred. Meant to grow as decisions land.
**Companion docs:**
- `observation-tensor-design.md` — obs tensor v1 working spec (delta + pointer to 5.05-1)
- `distance-features-design-space.md` — design rationale for distance-feature obs channels
- `compute-considerations.md` — compute reframe; channel count is not the binding lever
- `2026-05/5.02-5-strakam-paper-summary.md` — Strakam paper summary (architecture lineage)
- `2026-05/5.02-7-deepnash-summary-and-implications.md` — DeepNash architecture details (Pyramid Module spec)
- `2026-05/5.02-3-initial-notes-and-decisions.md` — load-bearing project decisions; §5 (Architecture) is partially stale, consult `2026-05/5.02-6-strakam-paper-takeaways.md` for U-Net decision
- `2026-05/5.02-6-strakam-paper-takeaways.md` — U-Net decision lives in §2.1

---

## 1. Status legend

Decisions in this doc are tagged with one of:

- **Locked** — Decision made, unlikely to change without specific reason.
- **Tentative** — Decision made, but might be revisited based on training signal or further analysis. Treat as the working baseline.
- **Default (inherited)** — Punted; using the inherited choice (Strakam → DeepNash) without project-specific scrutiny yet.
- **Open / TBD** — Explicitly unresolved.

---

## 2. Context and inheritance

We're building on the architectural lineage:

```
Perolat 2022 (DeepNash, Stratego)  →  Strakam 2025 (1v1 generals.io)  →  us (FFA generals.io)
```

DeepNash's "Pyramid Module" — a U-Net torso with residual blocks at each level, plus heads-as-Pyramid-Modules — is the architectural foundation. Strakam ported this to 1v1 generals with one notable change: a single H×W×9 policy head (per-cell action logits) instead of DeepNash's two-pass piece-selection-then-destination decomposition.

### 2.1 The Strakam existence proof (sharpened)

A reframing landed in this session: the relevant compute comparison is **Strakam → us**, not DeepNash → us.

- Strakam's full training: ~36 H100-hours total (3 BC + 33 self-play)
- Our BC budget: ~30–60 H100-hours per run

We're at parity, not ~10⁴× below. So *something* at roughly Pyramid-Module shape achieved top-25 1v1 ladder performance with compute we have access to. This is a strong signal that the inherited architecture is broadly trainable at our scale.

**Caveat:** Strakam's paper is genuinely vague on architecture details — "same architecture as Perolat 2022" doesn't commit to specific widths. The repo's `network.py` exists but is flagged as "not the paper's actual training code." So the existence proof is **qualitative on shape, not quantitative on widths.**

### 2.2 Where FFA differs from 1v1

The most relevant deltas for architecture:

| Dimension | 1v1 (Strakam) | FFA (us) | Implication |
|-----------|--------------|----------|-------------|
| Per-cell semantics | armies + ownership | armies + ownership | same |
| Spatial extent | ~18×18 | ~25×25 (~2× cells) | more spatial reach needed (favors more contraction) |
| Opponents to model | 1 | up to 7 | more channels, more memory bandwidth |
| Long-term strategic state | modest | significant (capture graph, contact, staleness) | drives obs-tensor design more than architecture |
| Game length | comparable | comparable | same |
| Combat semantics | numeric magnitudes | numeric magnitudes | unchanged |

**Net read:** FFA wants more spatial reach and more memory bandwidth, **not** necessarily more local capacity. This maps to contraction depth and obs-tensor decisions, not width.

---

## 3. Tentative decisions

### 3.1 Contraction depth = 2 — **Tentative**

**Decision:** Use 2 contraction levels in the U-Net torso (vs. DeepNash's 1).

```
25×25  →  13×13  →  7×7  →  13×13  →  25×25
        contract  contract  expand    expand
```

**Reasoning:**

DeepNash's single contraction (10×10 → 5×5 → 10×10) gives a bottleneck **theoretical receptive field of ~33 cells in input space** — about 3× coverage of a 10×10 board. Comfortable for global reasoning.

Applied unchanged to our 25×25 input, the bottleneck RF is still 33 (architecture-only quantity). 33 vs. board side 25 is barely larger; vs. board diagonal (~35.4) it's just under. **Single contraction on 25×25 is right at the edge for global RF.**

Effective receptive field (Luo et al. 2017) is typically much smaller than theoretical — often Gaussian-shaped, growing on the order of √(N_layers). This compounds the concern: if theoretical RF is borderline, ERF likely covers significantly less of the board with meaningful weight.

A second contraction restores comfortable coverage:

```
RF formula: RF_L = RF_{L-1} + (k_L − 1) × jump_{L-1};   jump_L = jump_{L-1} × s_L
```

Tracing through 2 contractions of the DeepNash Pyramid Module on 25×25 input: bottleneck RF ≈ 77, restoring the ~3× coverage ratio DeepNash had on Stratego.

**Important caveat: this does not fix mountain dead-ends fundamentally.** The graph-reasoning limitation is separate from RF (covered in `distance-features-design-space.md` §2). Distance-feature channels (the v1 baseline of point-source BFS) directly address the graph-reasoning side. RF expansion and distance features compound — both are needed to address the dead-end pathology.

**Compute envelope:** see §5 — the cost is modest (~2–10% per-sample FLOPs, ~50% more parameters). This is well within budget per the compute reframe.

**Channel widths at the new middle level:**
- Default proposed: **256 → 256 → 320** (keep widening only at the bottleneck)
- Alternatives considered: 256 → 320 → 320 (more middle capacity), 256 → 320 → 384 (progressive widening)
- For v1: **256 → 256 → 320**. Smallest delta from inherited; easy to revisit.

Status: tentative pending an empirical sensitivity sweep (1 vs. 2 vs. 3 contractions on small training runs). Listed in §8.

### 3.2 Value head output shape: categorical placement distribution — **Tentative**

**Decision:** The value head outputs a categorical distribution over predicted final placement (8-way softmax over placements 1–8). Loss: cross-entropy. Per-sample target: one-hot at the player's actual final placement in this game.

**Fallback option (held in reserve):** scalar binary win/loss (sigmoid + BCE). See "Fallback consideration" below.

**Reasoning:**

DeepNash and Strakam default to scalar binary win/loss. For 1v1 zero-sum, this is fine — 50% positive class on average, every position genuinely uncertain.

In FFA the target is sparser:
- Uniform skill: ~12.5% positive class average (one winner of 8)
- Our strong-player corpus: **25–40% win rate** for top players (some 40+%) — sparser than 1v1 but not catastrophically so

Categorical placement gives ~3 bits of signal per sample (log₂8) vs. ~1 bit for binary, even at the favorable balanced case. More importantly:

- **Captures multimodality** — FFA positions can have genuinely bimodal outcomes ("I'll come either 2nd outright or get eliminated next"). Scalar regression can't represent this; the distributional head can.
- **Distributional value head literature** (Bellemare et al. 2017, C51; QR-DQN follow-ups) shows distributional outperforms scalar even when only the mean is used at decision time — the richer target shapes representations better.
- **Phase 2 alignment** — placement is the natural FFA reward signal (the ladder ranks players by placement). A placement-aware value head transfers cleanly to PPO with placement-based reward.

**Phase 1 vs. Phase 2 role:**
- Phase 1 (BC): value head is auxiliary supervision. BC needs only the policy head for gradient; the value head's role is trunk shaping. Richer auxiliary signal is better.
- Phase 2 (PPO): value head is operationally needed for advantage estimation.

**Fallback consideration:** binary win/loss remains on the table because the user's strong-player win rates (25–40%, top players 40+%) make the binary signal denser than the uniform "12.5%" framing suggests. ~1 bit/sample at 30–40% balance is decent training signal, and binary is simpler. If implementation pressure favors it, or if placement targets turn out empirically harder to learn than expected, binary is an acceptable v1 choice.

Status: tentative. Sensitivity check on small training runs would settle this empirically.

---

## 4. High-level structure

```
Input obs tensor (~95+ channels @ 25×25)
    │
    ▼
┌──────────────────────────────────────────────┐
│ Torso: 2-contraction Pyramid Module          │
│   25² → 13² → 7² → 13² → 25²                 │
│   widths 256 → 256 → 320 (→ symmetric)       │
│   N=2 outer, M=2 middle, M=2 inner ResBlocks │
│   skip connections at each level             │
└──────────────────────────────────────────────┘
    │ (256-ch spatial embedding @ 25×25)
    ├──────────────────┐
    ▼                  ▼
┌──────────────┐   ┌─────────────────┐
│ Policy head  │   │ Value head      │
│ (Pyramid M.  │   │ (Pyramid M.     │
│  N=1, M=0)   │   │  N=0, M=0)      │
│ → 25×25×9    │   │ → 8-way softmax │
│ + masking    │   │ over placement  │
└──────────────┘   └─────────────────┘
```

(Optional auxiliary heads — see §6.)

Action space (policy head): per-cell `[pass + 4 directions × 2 splits]` = 9 channels. Inherited from Strakam. Output is `H×W×9` masked softmax over legal actions (owned cell + army > 1 + dest passable).

---

## 5. Compute envelope

Anchored to the compute reframe in `compute-considerations.md`: channel count is not the binding compute lever (1–3% of FLOPS); the actually-binding levers are inner pyramid width, spatial extent, and dataset × perspectives × turns.

### 5.1 Per-sample forward FLOPs (rough estimate)

Standard formula: Conv FLOPs ≈ 2 × H × W × C_in × C_out × K². For DeepNash-style ResBlocks (C → C/2 → C structure, two convs each): 18 × H × W × C² FLOPs per ResBlock at width C and resolution H×W.

| Component | 1-contraction | 2-contraction |
|-----------|---------------|---------------|
| Outer (25², 256ch, N=2 RBs each side) | ~3.0 G | ~3.0 G |
| First strided conv/deconv | ~1.2 G | ~1.2 G |
| Middle (13², 256ch, M=2 each side) | — | ~0.8 G |
| Second strided conv/deconv | — | ~0.4 G |
| Inner (13² or 7², 320ch, M=2 each side) | ~1.2 G | ~0.4 G |
| **Torso forward total** | **~5.7 G** | **~5.8–6.3 G** |

Note: outer-resolution (25²) layers dominate total compute. The added middle level at 13² is cheap; the inner level moves to 7² (cheaper than at 13²), partially offsetting.

### 5.2 Translated to training budget

Anchored to session 5.05-3's BC estimate of 30–60 H100-hours per training run:

| Metric | 1-contraction | 2-contraction | Delta |
|--------|--------------|---------------|-------|
| Per-sample FLOPs (forward) | ~5.7 G | ~5.8–6.3 G | +2–10% |
| BC run wall-clock | 30–60 H100-hrs | 31–66 H100-hrs | +1–6 hrs |
| Cloud rental cost | $60–600 | $62–660 | +$2–60 |
| Params (torso) | ~14M | ~20–22M | +50–60% |

Compute cost is small. Param-count increase (~50%) is the bigger qualitative cost, but ~22M params still fits comfortably on an H100 (80GB) with reasonable batch sizes.

**Caveat:** these are back-of-envelope numbers. Real numbers come from profiling the actual implementation. They're reliable on order of magnitude, useful for decision-making, not load-bearing for budget planning.

---

## 6. Defaults inherited (tentative pending review)

These are choices we've explicitly punted to "use the inherited Strakam/DeepNash choice" without project-specific scrutiny. They are **not locked** — any could be revisited if specific signal warrants.

| Decision | Inherited choice | Notes |
|----------|------------------|-------|
| Inner width | 256 outer / 320 bottleneck | Per the compute reframe, this is the actually-binding compute lever; never explored. Sensitivity sweep is on the open list. |
| Normalization | LayerNorm | Strakam paper doesn't specify. LayerNorm is the safe default for RL — BatchNorm interacts poorly with policy gradient. Worth inspecting Strakam's `network.py` to confirm. |
| Heads-as-Pyramid-Modules | Yes (per DeepNash) | Heads have their own conv structure rather than being linear projections. Strakam doesn't confirm they kept this; we're defaulting to inherited because the DeepNash justification (heads need their own spatial reasoning) holds for us. |
| Skip-connection geometry | Symmetric across encoder/decoder at each level | Standard U-Net pattern. With 2 contractions, 2 skip levels. |
| ResBlock structure | C → C/2 → C with 1×1 residual on stride | Per DeepNash spec (5.02-7 §1.3). |
| Activations | ReLU | Per DeepNash spec. |

---

## 7. Failure modes to track in evaluation

Ranked by likely relevance:

- **Mountain dead-ends.** Strakam's diagnosed pathology (`2026-05/5.02-5-strakam-paper-summary.md` §7.5). Mitigated partially by the v1 distance-feature channels (`distance-features-design-space.md` §6) and by the 2-contraction RF expansion (§3.1). **Not fully solved without GNN-style architecture.** Track via: agent gets stuck moving against impassable terrain; agent fails to route around mountains.

- **Mode-locking.** Strakam's other diagnosed pathology — agent fixates on attack OR defense OR castle-taking, ignoring others. Auxiliary-head territory if it surfaces (§8). Track via: held-out positions where agent ignores critical context (e.g., capturing castles while being attacked).

- **Output-class collapse in value head.** The categorical placement head is risk for collapse to a single mode (e.g., "always predict 4th place"). Track via: distribution entropy on held-out positions, calibration metrics.

- **Spatial blind spots from limited ERF.** Effective RF can fall short even when theoretical RF is adequate. Track via: position-based move review on positions requiring corner-to-corner reasoning.

- **Slot-permutation augmentation insufficiency.** If the network is still latching onto slot indices despite augmentation. Track via: held-out games where opponent slot identity is shuffled at inference.

---

## 8. Open / TBD

| Item | Notes |
|------|-------|
| Channel widths at the new middle level | Default 256→256→320; could be 256→320→320 or 256→320→384. Sensitivity sweep would settle. |
| Auxiliary heads | Open from `5.05-3-session-notes.md` §3.5. Candidates: predict eliminations, predict army-count-in-N-turns, predict who-eliminates-whom. Multi-task regularizer; FFA's strategic complexity strengthens the case. |
| Optimizer / LR / batch size / schedule | Default to Adam + standard PPO defaults until profiled. Strakam paper doesn't specify. |
| Sensitivity sweeps | Tentative decisions in §3 should be validated empirically on small training runs: contraction depth (1 vs. 2 vs. 3), value head shape (categorical vs. binary), middle-level channel widths. |
| Output-projection hardening | DeepNash's fine-tuning replaces softmax with thresholded discretization to prevent blunder accumulation in long games (their §1.9). Worth knowing as a backstop if BC produces a "occasionally blunders catastrophically" failure mode in live play. |
| Player/skill conditioning embedding | Open from `5.05-3-session-notes.md` §3.5. Lets us trade dataset size against filter restrictiveness. |
| Player-slot canonicalization edge cases | Slot-permutation augmentation is locked; verifying replay-data slot assignments is folded into parser/dataset prep work. |

---

## 9. Pointers and what's NOT in this doc

This doc covers network shape and head structure. It does **not** cover:

- **Obs tensor channel layout** — see `observation-tensor-design.md` (working spec) and `2026-05/5.05-1-observation-tensor-design-part2.md` (current load-bearing baseline).
- **Distance-feature design rationale** — see `distance-features-design-space.md`.
- **Compute reframe / where compute actually goes** — see `compute-considerations.md`.
- **Training pipeline (data → parser → loss → eval)** — not yet specified; flagged in `5.05-3-session-notes.md` §4 as a future dedicated planning session.
- **Slot-permutation augmentation, board-symmetry augmentation** — locked in `5.05-3-session-notes.md` §3.3.
- **Reward shaping / PPO specifics** — Phase 2 work.
