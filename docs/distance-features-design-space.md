# Distance Features Design Space

**Date:** 2026.05.06
**Status:** Design exploration. V1 baseline is locked (§6); richer schemes are deferred but preserved here for later iteration.
**Companion docs:**
- `network-architecture-design.md` — where the v1 baseline appears as an obs-tensor input and the contraction-depth decision sits.
- `observation-tensor-design.md` — the obs-tensor v1 working spec; lists the v1 distance channels in its delta section.
- `2026-05/5.02-5-strakam-paper-summary.md` — diagnoses the "mountain dead-ends" failure mode (§7.5).
- `2026-05/5.02-7-deepnash-summary-and-implications.md` — flags policy-independent inference as input features as a transferable pattern (§2.3).

---

## 1. Why this matters

Generals.io maps have non-trivial topology. ~20% of cells are mountains, which are impassable; cities are passable but cost armies to traverse; fog-of-war hides large portions of the board. This produces frequent mismatches between **Euclidean distance** (how far apart two cells are on the grid) and **traversable distance** (how far an army actually has to move to get from one to the other).

The Strakam paper diagnoses **mountain dead-ends** as one of two prominent failure modes of their SOTA 1v1 agent (§7.5):

> "[the agent] gets stuck. The CNN's inability to compute distances — it 'thinks' it can pass through impassable regions."

Our FFA project inherits the same architecture and operates on larger boards (~25×25 vs Stratego's 10×10) with more mountains and more strategic targets to navigate around. So the issue is at least as relevant for us, and probably more so.

This doc explores what we can do about it — at the input-feature level — without changing the core architecture.

---

## 2. The actual problem (it's not just receptive field)

"CNNs can't compute distances" hides an important subtlety. The deeper issue is that **path planning around obstacles is a graph-reasoning problem**, and CNNs are not graph-reasoning machines.

Two properties of reachability make it hard for CNNs:

1. **Topological distance ≠ Euclidean distance.** Two cells that are 5 grid-cells apart but separated by a long mountain ridge might be 30+ traversable steps apart. Convolution weights are spatial — they don't natively understand "blocked vs. open."

2. **Reachability is iterative.** "Can I reach cell C from A?" requires propagating reachability through the graph: A reaches its neighbors, those reach theirs, etc. **Each conv layer can extend reachability propagation by exactly one step.** Fully resolving reachability on a board with paths up to length ~50 would need ~50 sequential conv layers acting as BFS steps.

So even with infinite receptive field, a finite-depth CNN approximates BFS poorly. This is why GNNs (which the Strakam paper flags as future work) handle the problem natively — they explicitly propagate over edges.

What CNNs *do* learn is a heuristic: "in patterns where there's an obvious detour, prefer that." This works for short detours and fails for long ones. Hence Strakam's dead-ends.

### 2.1 Receptive field is a separate (also-real) limitation

For completeness: even Euclidean global awareness requires sufficient receptive field. DeepNash's torso has theoretical RF ≈ 33 in input space, comfortably covering its 10×10 board (~3× margin). Applied to our 25×25 it's barely larger than the side and just under the diagonal (~35.4). Effective RF (Luo et al. 2017) is typically much smaller than theoretical, compounding the concern.

The contraction-depth decision in the architecture doc (2 contractions, RF≈77) addresses this RF question. The graph-reasoning question covered here is **distinct and not solved by more contractions** — they are separable concerns that compound.

Full RF derivation: see `network-architecture-design.md` §3.1.

---

## 3. The general pattern: pre-computed distances as input features

The cleanest mitigation is to compute distances **offline in the parser** and feed the results to the network as input channels. The network learns to *use* the feature instead of having to *compute* it.

This is an instance of the broader pattern called out in DeepNash (their public-information tensor, our 5.02-7 §2.3 (a)): any time you can compute a policy-independent function from history to inferred state, doing it at the input is essentially free. The network learns to use the feature, not to derive it.

For BFS-style features specifically, the parser cost is microseconds per frame at 25×25 — negligible.

### 3.1 Literature pointers

For background on related approaches:

- **Distance / coordinate channels** — folklore in gridworld RL; routinely used in Atari/navigation papers without explicit citation. The simplest version of this entire family.
- **CoordConv** (Liu et al. 2018, "An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution") — adds x/y coordinate channels. Tangential to graph distances; addresses absolute spatial position.
- **Value Iteration Networks** (Tamar et al. 2016) — embeds value iteration as a recurrent CNN module. K iterations simulates K BFS-like Bellman backups. Demonstrated on 2D gridworld navigation. More powerful than passive distance channels; significant architectural addition.
- **Gated Path Planning Networks** (Lee et al. 2018) — VIN follow-up.
- **Dilated / atrous convolutions** (DeepLab et al.) — expand RF without losing resolution. Addresses the *receptive-field* limitation but not the graph-reasoning one.
- **Self-attention / transformer blocks** (ViT, hybrid CNN-attention) — global mixing in one operation. Same caveat: doesn't natively understand "blocked vs. open."
- **Graph Neural Networks** — principled solution to the graph-reasoning limitation. Strakam paper flags as future work. Major architectural change.

For our class of problem — extending a working CNN architecture without ground-up redesign — pre-computed distance channels are the right entry point.

---

## 4. Design space we explored

Within "pre-compute BFS, feed as channels," the key design lever is **what to compute distances from**. Three flavors emerged in the discussion, plus hybrids.

### 4.1 Strategic point sources (curated)

Pick strategically important positions and BFS from each. Each channel has a single, named source.

Candidate sources, in rough order of likely value:

| Source | Why useful |
|--------|------------|
| My general | Defense distance, retreat planning |
| Each known opponent general | Attack-distance to each target (sentinel when not yet revealed) |
| Nearest unexplored frontier | Where to scout |
| Nearest known enemy army > threshold | Where the immediate threat sits |
| Nearest neutral city | Where economic expansion lives |
| My nearest army stack > threshold | Where reinforcements come from |

**Channel count:** ~5–10 typical.

**Pros:**
- Stable identity across games (every channel always means the same thing)
- Dense per-channel signal (every cell has a meaningful value)
- Strategically curated — every channel maps to a known strategic motif

**Cons:**
- Engineer pre-decides what distances matter
- Less flexibility if there's a strategic axis we don't anticipate
- Anchor positions can move (e.g., "my nearest army stack") which slightly dilutes channel stability

### 4.2 Block-rep middle ground (systematic, fixed grid)

Divide the map into an N×N grid of blocks (e.g., 5×5 grid → 25 blocks on a 25×25 map). Pick one representative cell per block (most-passable cell, or block center if passable, else nearest passable). BFS from each rep.

**Channel count:** equal to block count (25 for a 5×5 grid).

**Pros:**

- Stable identity (block position never changes across games)
- Fixed channel count, no variable-region issues
- Captures global topology systematically — each block-rep produces a full distance field through mountains
- Trivial implementation (no edge cases beyond "block is fully blocked")

**Cons:**

- Loses within-block fragmentation distinctions (treats each block as a single "place")
- Wastes some capacity on pairs that don't matter strategically (e.g., far-corner-to-far-corner)

### 4.3 Block-region (denser, multi-region per block)

For each block, identify connected components **within the block in isolation** (artificially cut at boundaries, by design). A real-world region that physically spans blocks A and B becomes two separate block-regions, one anchored in each. Each block-region gets its own channel.

Per-cell channel value = BFS distance from the cell's containing block-region to the target block-region. This is piecewise-constant within each block-region (all cells in the same block-region see the same distance values for any given target).

**Channel count:** ~75 for a 5×5 block grid with up to K=3 regions per block (K=3 is a guesstimate, needs to be empirically validated by analyzing the maps in our dataset)

**Pros:**

- Captures multi-region topology where blocks are split by mountains
- Stable spatial anchor (channel identity tied to block position + canonical within-block ordering)
- "Feed the network exactly what it can't compute" applied most fully

**Cons:**

- Variable region count requires padding + edge-case handling for the long tail (blocks with >K regions). What happens to a 4th region in a block?
- More channels (~3× the block-rep scheme) — sparse channels in the tail risk being underutilized.
- Per-block-region piecewise-constant resolution loses some intra-region detail (bounded by block size).

**Open Questions:**

* What block configs work well? Dividing the map into a block-grid with 3x3, 4x4, 5x5, etc. feels natural configs to try first. What dstribution of `max(block-region counts)` do we see in the data (8-player FFA maps in our replay database)?
  * This should reveal what values of K would actually be needed (the k=3 value is my current guess). And then we can know the actual channel count needed as well.

* How well does the "standardization" of region ID assignment within each block work out? How good is the spatial correlation for each fixed region across our dataset?
  * e.g. Look at region-1 tiles for all maps in the data. Compute a metric measuring how well these regions "line up". They should be in relatively the same geographic position for each map. Ideally, the position is very similar.
  * There may be fudge room — the channels may still highly useful even if they are only approximate. 80% "accurate" for distance measurements would still be a massive improvement over the native "navigational" capabilities of the CNN.
* Can the  network learn to make effective use of these approximate distance signals?

### 4.4 Hybrid combinations

E.g., strategic point sources for known-critical things (my general, opponent generals) **plus** a block-rep grid for systematic global coverage. Captures both the curated and the systematic axes.

**Channel count:** ~30–40 typical.

**Tradeoff:** compounds engineering complexity but gives both robustness (network can find what it needs from systematic coverage) and named-channel signal density (curated channels carry interpretable meaning).

### 4.5 Tradeoff summary

| Scheme | Channels | Captures | Main risk |
|--------|----------|----------|-----------|
| Strategic point sources | ~5–10 | Curated strategic distances | Engineer pre-decides what matters |
| Block-rep grid | ~25 | Systematic global topology | Loses intra-block fragmentation |
| Block-region | ~75 (with 25 blocks) | Full multi-region topology | Sparse channels, tail edge cases |
| Hybrid | ~30–40 | Curated + systematic | Compounded complexity |

### 4.6 Other design axes (orthogonal to source selection)

These apply regardless of which scheme is chosen:

| Axis | Options |
|------|---------|
| Distance encoding | Raw, log-scaled (`log(1+d)`), Gaussian-smoothed (`exp(-d/τ)`), binary thresholds ("within k") |
| Mountain handling | Hard ∞ vs. large sentinel constant |
| Fog handling | BFS treats fogged cells as opaque (plan from what we know) vs. optimistically passable |
| Update cadence | Every frame vs. only when terrain knowledge changes (cheap optimization, not load-bearing) |
| Multi-scale | Feed both `d` and `log(1+d)` to give the network choice of scale |

---

## 5. Channel utilization at our scale (a real concern, but easy to overstate)

A natural worry about denser schemes: at our data scale (~100M training samples) and inherited model capacity (~14–22M params), will the network actually learn to use 25–75 distance channels effectively?

**The first conv layer absorbs correlated channels via implicit linear projection.** If two channels are highly redundant, the layer learns small weights for one and routes signal through the other — they don't both have to be "used independently." This compresses correlated inputs into a lower-rank effective representation. The cost of redundant input channels is a few thousand extra parameters in the first conv (e.g., 75×256 ≈ 19k vs. 25×256 ≈ 6k weights), trivial compute, and *not* "70% of channels wasted."

The remaining concern: **truly null channels** — slots that are sentinel-dominated in most training samples (e.g., "block 6, region slot 3" being empty in 95% of games). Those weights see weak intermittent gradient and don't pull weight, but they're not actively harmful either. This is a smaller fraction than naive worst-casing implies.

**Net for v1:** at ~100M samples, even the densest scheme (~75 channels) is plausibly fine. The "20–40% utilization" framing was too pessimistic. Where the line falls precisely is empirical; we can revisit if obs-tensor changes are warranted post-training.

---

## 6. V1 baseline (locked)

For v1, we adopt a minimal version of §4.1: **point-source BFS to each general (self + each opponent).**

| Property | Choice |
|----------|--------|
| Source set | Self general + each opponent general (8 sources, 8 channels) |
| BFS scope | Through cells known to be passable (mountains as deleted nodes; fogged cells treated as opaque) |
| Sentinel | -1 for unreachable cells and for unrevealed opponent generals |
| Encoding | Log-scaled: `log(1 + d)`, with sentinel applied *after* the log (don't `log(1 + (-1))`) |
| Update cadence | Every frame (cheap; can be optimized later) |

**Why this minimal set:**

- Every channel has stable identity (per-opponent slot, identical to other per-opponent broadcast channels)
- Strategic relevance is direct and well-understood (defense, attack-target)
- 8 channels is well below the "load-bearing fraction of obs tensor budget" line
- No edge cases beyond handling a single, straight-forward sentinel
- Composes naturally with the slot-permutation augmentation already locked from session 5.05-3

**What this v1 baseline does NOT include but could later:**

- Distance from nearest unexplored frontier (scouting target)
- Distance to nearest neutral city (economic target)
- Distance from my nearest army stack > threshold (reinforcement distance)
- Block-rep grid (systematic global coverage)
- Block-region multi-region encoding (denser coverage)
- Encoding variants (Gaussian-smoothed, binary thresholds)

Adding any of these is cheap from an obs-tensor budget perspective per the compute reframe. The reason to hold them out of v1 is **parsimony** — start with the channels we can most clearly justify, evaluate utilization, then add.

The integration of these v1 channels into the full obs-tensor channel list is documented in `observation-tensor-design.md` §2.

---

## 7. Juicy threads for future iteration

These were considered seriously and deferred. Preserving them here so they don't have to be rederived.

### 7.1 Block-rep grid as an additive layer

The middle-ground scheme from §4.2. Adds ~25 channels of systematic global topology without the variable-region complexity of block-region. Plausibly the highest-value extension if we ever see "agent makes locally good moves but globally incoherent ones" pathology — it would give the network a fixed reference grid for global topology that the curated point-source set might miss.

Cheapest sensible test: train v1, then a v1+block-rep variant, compare output-move quality on held-out positions.

### 7.2 Block-region multi-region encoding

The denser scheme from §4.3 (the user's idea, defensible after pushback). Adds ~75 channels. Captures multi-region topology where mountains split blocks, which the block-rep scheme conflates. The most expressive of the three, with the most edge-case engineering. Worth considering if simpler schemes plateau.

Empirical question to settle before doing this: at our data scale, can the network extract signal from sparse-tail channels (e.g., "block 6, region 3" being empty in 95% of games)?

### 7.3 Encoding variants

The log-scaled encoding from v1 is a default, not an investigation. Worth trying:
- **Gaussian-smoothed** (`exp(-d/τ)`): produces a soft "neighborhood of source" mask that may be easier for the network to use than raw distance for some tasks.
- **Binary thresholds**: "within 3 steps," "within 7 steps," etc. — gives the network discrete categorizations.
- **Multi-scale** (raw + log + Gaussian): redundant but cheap, lets the network pick.

### 7.4 VIN-style modules

If pure passive distance channels turn out insufficient, Value Iteration Networks (Tamar et al. 2016) embed an iterative planning module. Genuinely changes the architecture but addresses the iterative-BFS limitation directly.

### 7.5 GNN as future-work

The principled fix to the graph-reasoning limitation. Strakam paper flags it as future work for similar reasons. Major architectural change; out-of-scope for v1 but worth knowing as the right escalation if dead-ends persist post-training.

### 7.6 Distance-aware auxiliary losses

If we add auxiliary heads (open question per session 5.05-3 §3.5), one natural auxiliary task is "predict distance to the nearest enemy general" or "predict whether cell C is reachable from my general." These are policy-independent, exact targets — ideal for auxiliary supervision.

---

## 8. Methodology note: working with LLMs on complex spatial schemes

LLM spatial reasoning is a known weak spot — particularly involving partitioning, connectivity, multi-scale aggregation, and region semantics. The discussion that produced this doc bumped into this several times. Concrete examples from the session:

- **Misread of block-region semantics.** Initial critique assumed a real-world region spanning two blocks would create boundary attribution issues. The scheme was actually defined so block-regions are connected components *within each block in isolation* — by construction. The misread invalidated a "block-boundary artifacts" objection that, on closer inspection, didn't apply.
- **Conflation of region variability with channel-identity variability.** Initial critique conflated "regions differ game-to-game" with "channel meaning differs game-to-game." These are distinct: canonical block-position assignment cleanly resolves the second even when the first remains.
- **Overstated edge cases.** Several engineering complications named in the initial critique (centroid in a mountain, fully-blocked block, boundary-spanning regions) turned out to be trivial or non-issues in the actual scheme.
- **Too-pessimistic channel utilization estimate.** Initial framing of "20–40% of 75 channels effectively used" undercounted. First-layer linear projection compresses correlated inputs, so dense schemes degrade more gracefully than the framing suggested.

These corrections came from user pushback during the discussion; this doc reflects the revised understanding rather than the initial reactions.

**General lesson:** when LLM-assisted analysis is part of the loop and the topic involves complex spatial reasoning, treat first-pass LLM reactions as one input among several rather than as authoritative analysis. Budget time for back-and-forth. Domain spatial intuition (especially the user's) typically catches errors the LLM misses on first pass.

This applies more broadly than this doc: anywhere complex spatial reasoning is load-bearing in the project (obs-tensor designs, parser logic, attention patterns, GNN topology, hierarchical aggregation), expect LLM-assisted analysis to need correction and don't accept first-pass takes as settled.

**Note for future LLMs reading this doc:** if you're being asked to evaluate or propose schemes involving spatial topology, partitioning, connectivity, or hierarchical aggregation in this project, allocate more reasoning effort than you might default to, and hold initial conclusions a bit more tentatively. The session that produced this doc reached good outcomes — and the model did grok the user's schemes properly once it engaged carefully — but several first-pass takes were overstated or wrong, and only got corrected after substantive user pushback. The failure mode isn't an inability to reason about these schemes; it's overconfident initial reactions that don't survive a second look. Spending the extra thought up front converges faster than relying on the user to catch errors after the fact.

---

## 9. Status notes

- **Locked for v1:** §6 baseline (point-source BFS to each general)
- **Deferred:** all of §7
- **Document is evergreen:** will be updated if v1 baseline changes or richer schemes are adopted in future iterations.
