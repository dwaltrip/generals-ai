# Assumptions & Key Calcs (Grab-Bag)

**Status:** Cross-cutting reference for assumptions and derived numbers that are load-bearing across multiple project decisions.
**Updated:** 2026.05.07
**Posture:** Grab-bag, not exhaustive. Consult before locking decisions; verify if relying on a specific value. Updates happen opportunistically — items go stale and that's expected. Topical docs (`network-architecture-design.md`, `compute-considerations.md`, etc.) are canonical; this doc is a cheat sheet.

---

## How to read

Each item tagged:
- **F** — Fact: from data, paper, or measurement
- **A** — Assumption: working estimate or guess
- **C** — Calculated: derived from F + A

Section IDs (`A1`, `B3`, etc.) are stable references for use elsewhere.

---

## A. Data & samples

- **A.F1** — Current dataset: **135,769** raw 8-player FFA replays with `.gior` fetched. From `replay-collector/scripts/analyze_turn_stats.py`.
- **A.F2** — Game-length distribution (half-turns): mean **632**, median **563**, p25/p75 **433/743**, p10/p90 **342/974**.
- **A.F3** — Unit-of-counting: DB `turns` field stores **half-turns**. One half-turn = one move = one (obs, action) sample for the moving player.
- **A.F4** — Cohort: ~40% of filtered games have ≥2 strong-player trajectories.
- **A.A1** — Strong-player trajectory survival: **~60%** of game length (range 50–70%). See §E for validation path.
- **A.A2** — Filter survival rate: **~60%** of raw replays (range 50–70%). Depends on concrete filter definition.
- **A.A3** — Effective independence: **~30–60%** of raw samples (adjacent timesteps in a trajectory are highly correlated).
- **A.A4** — Board-symmetry effective multiplier: **~2–4×** (theoretical 4× rectangular D₂, 8× square D₄; effective independence is smaller).
- **A.A5** — Slot-permutation effective multiplier: **~2–3×** (theoretical max 7! = 5040; most permutations are correlated).
- **A.C1** — Strong-player trajectory length: **~380 half-turns** average (range 315–440 across A.A1 uncertainty).
- **A.C2** — Per-replay sample count: **~530** (state, action) pairs average (range 440–620). Derivation: 60% × (1×380) + 40% × (2×380).
- **A.C3** — Filtered replay count at 135k: **~81k** (range 68k–95k).
- **A.C4** — Total raw samples at 135k: **~36–50M** (state, action) pairs.
- **A.C5** — Effective sample-views per ~5 epochs with augmentation: **~1–2B** at 135k; **~2–4B** at the achievable 270k upper end.

Full discussion: `compute-considerations.md` §4.

---

## B. Compute & training budget

- **B.F1** — Strakam BC reference: **3 hours on 1 H100**, ~16k filtered 1v1 games, top-25 1v1 ladder.
- **B.F2** — H100 SXM5 peak BF16: **~990 TFLOPS**. Real-world utilization typically **30–50% of peak**.
- **B.A1** — Training default: **5 epochs** assumed for sample-view math (could plausibly be 3–10).
- **B.A2** — Per-sample compute (2-contraction at 256/320): **~17–19 GFLOPs** forward+backward.
- **B.A3** — Strakam used DeepNash's inherited widths (256/320). **Uncertain** — paper silent; published `network.py` is a 4×4 toy. Calibration anchor depends on this.
- **B.A4** — Phase 1 budget posture: **~$500 total cloud spend on BC** as soft upper bound. Phase 2 is a separate downstream decision.
- **B.C1** — Planning anchor for BC wall-clock: **~30–60 H100-hours per run**. Calibrated to Strakam under B.A3.
- **B.C2** — Cloud cost per BC run at $2–10/hr: **~$60–600**.
- **B.C3** — Mini-sweep cost: **~12–25 H100-hrs total** (4 variants × ~7.5% of full duration).
- **B.C4** — Strakam-circularity range: real BC wall-clock could plausibly be **0.5–2× the 30–60 estimate** (~15–120 H100-hrs/run) depending on B.A3.

Full discussion: `compute-considerations.md` §4 (including "Softness in the 30–60 anchor" subsection).

---

## C. Architecture & params

- **C.F1** — ResBlock structure: 2 convs (3×3, 3×3) at C → C/2 → C, ReLU, residual sum. Per DeepNash spec.
- **C.F2** — ResBlock FLOPs: **~18 × H × W × C²** per block (forward).
- **C.F3** — First-conv FLOPs share: **1–3% of total** network FLOPs. Input channels are not the binding compute lever.
- **C.F4** — Inherited DeepNash widths: **256 outer / 320 bottleneck**.
- **C.F5** — 1-contraction torso params at 256/320: **~14M**.
- **C.F6** — 2-contraction bottleneck spatial extent: **7×7** at 25×25 input. Theoretical RF at bottleneck: **~77 cells**.
- **C.A1** — Width scaling: both params and FLOPs scale as **C²** per ResBlock; a single multiplier captures the delta when scaling all levels proportionally.
- **C.C1** — 2-contraction torso params at 256/320: **~18–22M** (~50% more than 1-contraction from the added level).
- **C.C2** — Width-sweep variant param counts (proportional scaling at 2-contraction):

| Multiplier | Outer / Mid / Bot | Approx params |
|---|---|---|
| 0.5× | 128 / 128 / 160 | ~5M |
| 0.75× | 192 / 192 / 240 | ~10M |
| 1.0× (baseline) | 256 / 256 / 320 | ~18M |
| 1.5× | 384 / 384 / 480 | ~40M |

Full discussion: `network-architecture-design.md` §3.1, §3.3, §5.

---

## D. Heuristics & rules-of-thumb

- **D.A1** — Data:params heuristic for BC at our scale (directional, not a hard constraint):

| sv/p (with aug) | Regime |
|---|---|
| < 1 | Under-trained |
| 1–5 | Regularization-dependent; can work with care |
| 5–30 | Comfortable for BC at our scale (validated by Strakam at ~23 raw sv/p) |
| 30–100 | Very comfortable |
| 100+ | Probably under-using capacity |

- **D.F1** — Strakam existence proof: ~14M params, ~23 raw sv/p (~180 with aug), top-25 1v1 ladder. Strongest empirical anchor for "what works at our compute scale."
- **D.A2** — BC tolerates lower data:params than image classification — per-sample info density is lower, but task structure compensates. CV-derived "20 sv/p comfortable" framings are too tight for BC; treat as directional.

**Default regularization stack** (paired with width sweep, per `network-architecture-design.md` §3.3):
- AdamW with `weight_decay ≈ 1e-4`
- Trunk dropout 0.05–0.10
- Train/val loss tracking with early-stopping backstop
- Board-symmetry + slot-permutation augmentation enabled from run 1

---

## E. Validatable / pending

Items flagged for empirical validation when the right data exists. Mostly cheap to pin down once available.

- **A.A1** strong-player trajectory survival → placement-aware DB query post-parser. ~1 hour of work; tightens sample math materially.
- **A.A2** filter survival rate → depends on concrete filter definition; pin down once filter is finalized.
- **A.A4** board-symmetry theoretical multiplier → measurable exactly from map-aspect distribution.
- **A.A5** slot-permutation effective multiplier → only via held-out eval; may never get a clean number.
- **B.F2** real H100 utilization → measured during first BC training run.
- **B.A2** per-sample wall-clock → measured during first BC training run.
- **B.A3** Strakam architecture (widths) → likely never confirmed externally; circularity flagged in `compute-considerations.md` §4.
- **`last_seen_owner` 9-channel utility** → empirical check post-parser; affects obs-tensor design.
- **Map-aspect distribution** → quick query once parser surfaces map metadata; pins down A.A4 exactly.
