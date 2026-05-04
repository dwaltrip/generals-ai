# FFA star decay — empirical analysis

Reverse-engineered from generals.io season 42 leaderboard snapshots
(`replay-collector/data/leaderboards/season-42.json`). The goal was a
foundation for a player-skill ranking that distinguishes "actively played
and ranked high" from "ranked high once and decayed downward without
playing." This required understanding how rank decay actually works.

## TL;DR

- **Star deltas from games are continuous; decay deductions are quantized
  to multiples of 0.5 stars.** Therefore *any exact 0.5-multiple in a
  weekly delta means the player played zero FFA games that week.* This is
  the core activity classifier.
- A fully inactive week produces a delta of **-3.5 stars** (~10% of all
  consecutive-week deltas). Smaller-magnitude exact half-stars (-2.0,
  -2.5, -3.0) come from inactive weeks that overlap the grace period
  and/or "big team event" windows where decay is suspended.
- Decay events appear to fire roughly once per day past the grace period,
  each removing -0.5 stars (or -1.0 for stars > ~45, per a known-bug doubling).

## Background

generals.io's FFA leaderboard ranks players by `stars`. To prevent dormant
high-rated players from sitting on the leaderboard forever, stars decay
when a player is inactive. Per community / dev sources:

- Decay triggers when fewer than 3 games have been played in a recent
  lookback window (~2-3 days).
- Decay is paused entirely during "big team event" weekends, when FFA
  mode is replaced by big-team mode (typically 1-3 days, occasionally
  longer).
- A known bug applies double decay (-1.0/event) to players above ~45
  stars; halved decay (-0.25) for low-stars players was intended but
  appears to have been rarely active.

## Data

- 1,684 unique FFA players across 10 weekly snapshots in season 42.
- ~6,000 consecutive-week stars deltas total.
- Each leaderboard entry: `{username, stars}`. Stars are stored to 2
  decimal places; rank is implicit from list order.

## Methodology

Two complementary techniques anchored the analysis.

**1. Bucketing deltas at exact half-star values.** Every consecutive-week
star delta was checked for being an exact multiple of 0.5 (within
floating-point tolerance). The frequencies are radically non-uniform: a
single value (-3.5) accounts for ~10% of all transitions, with several
other half-star values each holding 1-5%. Continuous distributions cannot
produce this kind of discrete spike, so the decay process must be the
source.

**2. Sequence-pattern analysis.** For each player, every consecutive
3-week run was reduced to a 2-tuple of labels — each delta becomes either
its exact half-star value (e.g. `"-3.5"`) or `"x"` for any continuous
value. The same was done for 4-week runs (3-tuples). This collapses the
messy continuous tails into a small set of discrete patterns whose
counts can be inspected by hand.

Counts were also segmented by the calendar week each sequence began, to
reveal which patterns are concentrated on specific weeks (a fingerprint
of "big team event" timing).

Implementation: `replay-collector/scripts/analyze_sequence_patterns.py`.

## Findings

### 1. Continuous game deltas + discrete decay

Single-game star changes are continuous floats determined by opponent
strength and placement. Decay deductions, by contrast, fall on the 0.5
grid. So:

- A week with **at least one game played** produces a continuous net
  delta (essentially never an exact half-star multiple).
- A week with **zero games played** produces a net delta that is the
  sum of half-star decay events — itself a half-star multiple.

This gives a clean binary classifier:

| observed delta | meaning |
|---|---|
| not on the 0.5 grid | played ≥1 game that week |
| exact 0.5-multiple | played 0 games that week |

### 2. The decay model

The pattern data fits **one decay event per day past a 2-3 day grace
period, each event = -0.5 stars** (cap: 7 events / -3.5 per week).

Predicted weekly delta for an inactive player (no events suspending
decay):

| context | grace days | decay days | weekly delta |
|---|---|---|---|
| second+ consecutive inactive week | 0 | 7 | **-3.5** (cap) |
| first inactive week | 2-3 | 4-5 | **-2.0 to -2.5** |
| inactivity onset mid-week | 3+ | 1-3 | -0.5 to -1.5 |

Big-team events suspend decay for their duration:

| event days during week | effective decay days | typical delta |
|---|---|---|
| 0 (no event) | 7 | -3.5 |
| 1-2 (typical weekend event) | 5-6 | -2.5 to -3.0 |
| 3 (extended event) | 4 | -2.0 |
| 4 ("forgot to turn it off") | 3 | -1.5 (rare) |

The observed frequency ordering of half-star values matches this
ordering, including the rarity of -1.5 and the near-absence of -1.0 / -0.5.

### 3. The cohort-progression signature

The week-segmented view shows decay-related patterns concentrated on
*specific* starting weeks rather than uniformly distributed. Examples
from season 42 (3-delta patterns):

| pattern | total | almost entirely starting at |
|---|---|---|
| (-2.5, -2.5, -3.0) | 48 | week 4 |
| (-2.5, -3.0, -3.5) | 56 | week 5 |
| (-3.0, -3.5, -3.5) | 49 | week 6 |
| (-3.5, -3.5, -3.5) | 43 | week 7 |

This is a single *cohort* of players going inactive around weeks 3-4,
whose decay then ramps from partial → cap as the grace consumes early
weeks. The patterns shift right by one starting-week index for each
"step" along the ramp. Different ramp shapes ((-2.5, -2.5, -3.0) vs
(-2.5, -3.0, -3.5)) reflect different durations of big-team events
during specific weeks rather than different player behavior.

### 4. High-stars decay scaling

Players with stars > ~45 show a modal decay of **-5.0 per inactive
week** rather than -3.5, consistent with the known-bug doubling. Counts
of exact -5.0 deltas across season 42:

| metric | -5.0 count |
|---|---|
| ffa | 46 |
| ffawin | 55 |
| ffacombat | 26 |
| ffakills | 87 |

Quarter-but-not-half multiples (e.g. -0.25, -1.25) appear at ~1% of all
deltas, somewhat concentrated in the <50 stars range — consistent with
*some* halved-decay activity but at a frequency close to what
random-chance from continuous game deltas would produce. Halving was
not broadly active in season 42.

### 5. The 0.0 anomaly

About 2% of weekly deltas, and 2% of two-week sequences `(0.0, 0.0)`,
show *exactly* zero change. Continuous game deltas should hit zero with
probability ≈ 0, so something else is going on — probably leaderboard
floor clamping, or a peak-protection rule preventing current stars from
falling below some prior anchor. **Not yet investigated.** Players
showing `(0.0, 0.0)` should be inspected for what stars values they sit
at before applying any classifier in production.

## Activity classifier (recommended)

For computing per-player, per-week activity:

```
delta = stars(week_N+1) - stars(week_N)

if delta is not on the 0.5 grid:
    active that week (high confidence)
elif delta == 0.0:
    edge case — investigate before classifying
else:
    inactive that week (high confidence)
```

This is the foundation for an activity-aware persistence metric: rather
than counting "weeks ranked in top N" (which credits players coasting on
prior peak via decay), count "weeks the player actually played AND was
ranked in top N."

## Open questions

- **The 0.0 cases** — what mechanism produces them? Worth checking the
  stars distribution of affected players.
- **Exact form of high-stars decay scaling** — is it strictly doubled
  above 45, or a smoother function? The d1-window filter we used
  systematically biases against high-stars players, so the existing
  data doesn't pin this down.
- **How "good" is half-star detection in practice** — could rare
  coincidences (game delta + decay netting to a half-star) inflate the
  inactive count? Unlikely with continuous games but should be sanity-
  checked against players known to be active.

## References

- `replay-collector/scripts/analyze_stars_decay.py` — earlier
  exploration: delta histograms, additive-vs-multiplicative test,
  second-week-followup analysis. Produces two PNGs in `scripts/out/`.
- `replay-collector/scripts/analyze_sequence_patterns.py` — the
  primary tool. Produces 4 tables: overall counts and starting-week
  segmentations for both 2-delta and 3-delta sequences.
- `replay-collector/data/leaderboards/season-42.json` — the source
  data analyzed here.
