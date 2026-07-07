# Writing-preference calibration exercise — 2026-07

*Sessions of 2026-07-06/07. Part of the writing-style arc in this directory — see `tells-enumeration-v2.md` and `inventory.md` for the earlier collection work.*

**Outcomes, in brief:**

- Global CLAUDE.md gained a `Well-constructed` writing section (three sentence-level failure classes, a repair order, a paragraph rule), a fourth finishing-pass entry, and an "X, not Y" rule — built almost entirely from examples validated blind in this exercise.
- The content-reviewer agent got two corrections: its active-voice example (which ranked the topic-breaking form first) and a new "claim outruns its shown backing" anti-pattern.
- Two meta-findings shaped both: Daniel's value hierarchy (§3) and the rules-as-control-system principle (§6).

## 1. What this was

A blind preference survey: 22 items over 4 rounds. Claude built minimal-pair variants of sentences and paragraphs (mostly from real repo docs), committed predictions of Daniel's picks to a file before posting each round, and Daniel picked winners with reasons. The disagreements and prediction misses were the payoff — each one sharpened a tell, fixed an example, or exposed a boundary.

Standing: this is the first artifact in the writing arc whose content went through deliberate alignment. The tells-enumeration doc collected candidate tells but never got a validation pass — that was part of the ambitious categorization plan that petered out. Every finding here is backed by at least a blind pick with a stated reason. The reliability note below grades the evidence tiers.

## 2. Method and reliability

Mechanics: minimal pairs on realistic repo material, with predictions written to a scratch file before each round posted (the tool-call ordering in the transcript makes the commitment verifiable). The rounds evolved: multi-variant sentence items (rounds 1–2), paragraph-level items (round 3), then a wide net of two-variant pairs at one tell each (round 4). Each round was salted: items where the naive rule gives the wrong answer, "fine as written" plants, and masked overclaims riding on smooth constructions.

Prediction record: roughly 14 of 21 scored items. The primary misses were register-side — items 8, 15, 19, and 20 each expected a register rule to decide a pick it didn't. The held-parse underweighting shows in the secondary rankings rather than the primary misses: on items 3 and 5, the prediction file ranked the paired-dash and late-verb variants above where Daniel placed them (noted in those appendix entries). Both directions are markers for where drift will re-enter.

**Reliability caveat.** The error source a survey like this can't remove is the rater: fatigue, misreads, typos, ordinary human error, over a session that ran long. Two compensations were in place. Themes were covered by multiple items where possible, and cross-item consistency — plus the predictions being largely right — corroborates that the picks weren't noise. Observed noise instances, for honesty: the item-3 ranking was walked back a round later, item 17 was abstained on, and item 8 came with "we shouldn't look too closely at this one." The working rule: pattern-level findings (several items plus stated reasons) are solid, and single-item findings are the weakest tier. The four single-item findings that each changed a rule, a boundary, or the seasoning queue (14, 15, 19, 20) came with articulated reasoning and in-chat discussion, which upgrades them above a bare pick — but they're still first in line for re-testing if a future round runs.

## 3. Daniel's preference model

When values collided across the exercise, they resolved in a consistent order:

1. **Completeness and concreteness.** Vague labels, dangling implications, and incomplete thoughts are the top killer — "The ambiguity / vagueness is absolutely a killer." A cute compressed label standing in for the concrete fact ("the tight one") loses to plain specifics every time. (Items 14, 15, the round-1 jargon catch, the fragmenting-floor refinement.)
2. **Information hierarchy.** Important first, support immediately behind it. (Items 10, 11, 19.)
3. **Plain construction.** Topic as subject, no held parses, flow. Held-parse pain — a verb kept waiting by a long subject — is the single most-penalized construction fault: it decided items 1 and 5 and outranked every other sin it appeared with.
4. **Calibration.** Claim strength matches shown backing. (Items 7, 18, 21, 22, plus the item-10 overstate-then-walk-back catch.)
5. **Register (anti-punch).** Real but weakest: overridden three times in round 4 — a routing "worth noting" kept (15), an owned "much" kept (20), a bold verdict lead-in kept (19).

Two riders:

- **Register's low rank describes taste, not rule priority.** Daniel's call, endorsed here: the register rules guard the strongest LLM tendency (leading with punch the content can't back), and the scarcity of register failures in recent output is at least partly those rails working. Re-weighting the rules by this hierarchy without that correction would re-open the door.
- **Convergent corroboration.** Daniel independently re-derived two of the principles from his own reactions mid-exercise: given-before-new on item 9 ("handle loads is the topic I'm already aware of… RawCheckpoint is a new thing, and I have to wait to find out what it's for") and the topic caveat on item 12. Convergence of that kind is stronger evidence than agreement with a stated rule.

## 4. Findings promoted to rules — evidence map

Authoritative homes: global CLAUDE.md (the `Well-constructed` section, plus the "X, not Y" bullet in Plainly-written) and the content-reviewer agent file. This section only maps each rule to its evidence — it does not restate the rules.

- Topic-as-subject base rule, with the passive sanctioned when the topic is the thing acted on ← items 2, 9, 12.
- Late-verb / gerund-headed-subject tell ← items 1, 5 (both decided on it).
- Nominalization tell ← item 12.
- Compound-coinage tell, with the established-terms boundary ← item 4 (the "fine as written" plant that failed) and item 5 (the boundary).
- Cleverness/personification tell ← item 19-B ("a concern that isn't its own") and pre-exercise reviewer findings on the 7.06-2 doc (both reviewers flagged its "survive … does not survive" echo).
- Aside rules — single clause only, parens over paired dashes, a trailing single-clause dash is fine ← item 3 (including its walk-back), item 6, item 10 (B won carrying a trailing dash), item 11-C.
- Two-announcements split ← item 3.
- Fragmenting floor and the subordinate-clause demotion option ← item 4's follow-up discussion, item 6.
- Paragraph rule — topic strings ← item 9; conclusion-first with the verdict cashed immediately ← items 10, 11, 19; afterthought qualifier ← item 10-A (Daniel's catch).
- "X, not Y" — manufactured contrast cut ← item 13; anchor requirement ← item 14.
- Existing rules reinforced, no change needed: fragments and semicolons (items 16, 7-C), certainty vocabulary (items 18, 21, 22, 7-A).
- Agent-file style-grounding correction (subject carries the topic) ← item 2, plus the hook/call discussion that exposed the original example as miscalibrated (§6).
- Agent-file "claim outruns its shown backing" anti-pattern ← the pre-exercise survey of six reviewer reports (calibration findings near-absent, with no named entry to file them under), with items 7 and 10 demonstrating the smoothness-masks-overclaim mechanism it targets.

## 5. Findings held back — the seasoning queue

Deliberately not promoted. Each entry names what would promote it.

- **Em-dash global stance.** Paired mid-sentence dashes nearly always lost, and Daniel's current lean is "the dash needs to almost always be removed." The aside rules landed in CLAUDE.md, and the broader stance waits for another session's confirmation.
- **Intensifier floor (item 20).** An intensifier on an explicitly-owned subjective judgment is fine — the target is intensifiers posing as measurement. Promotion would mean first adding an intensifier rule to CLAUDE.md, since none exists there today.
- **Importance-marker boundary (item 15).** "Worth noting" earned its keep as a routing marker in an operational doc — the tell's target is the reflexive use, where the marker substitutes for the point. Same situation as above: the tell exists only in the tells list, so the boundary is recorded here until reconciliation.
- **Reflexive ", so" (item 17).** Abstained — the item's context was too thin to judge. The existing CLAUDE.md rule was kept and is the one construction rule with no validation behind it.
- **"Breathing room" words (item 8).** "In existence" survived as gentle reinforcement plus pacing — a floor under "omit needless words." Held loosely at Daniel's request.

## 6. Design principles

- **Rules are a control system, not a taste mirror.** A rule's strength should track the strength of the tendency it counteracts, not the rank of the value it protects. Corollary: a prevalence survey of current output measures rules-plus-tendency together, so a quiet failure mode is evidence about the rails as much as about the tendency.
- **Examples are the spec.** Models imitate examples harder than they obey surrounding prose, so a miscalibrated example actively trains the wrong preference. Observed instance: the agent file's active-voice example ranked "the hook denies the call" over "the call is denied by the hook," and reviews absorbed the ranking. Blind validation is the standard examples should meet — which is most of why this exercise existed.

## 7. Changes landed, and open threads

Landed this session (named by description, no SHAs):

- Global `~/.claude/CLAUDE.md`: the `Well-constructed` section, the fourth finishing-pass entry, the "X, not Y" bullet, a one-word touch to the Writing voice intro.
- `.claude/agents/content-reviewer.md`: the style-grounding correction and the "claim outruns its shown backing" anti-pattern.
- `docs/agent-tooling/content-review/notes.md`: a dated entry recording both agent changes and the review-survey finding behind them.

Open threads:

- **Reconcile `tells-enumeration-v2.md` with this doc.** The corrections it is owed:
  - Recode the "Bold lead-in pre-announcing the conclusion" entry. Daniel blind-picked that entry's own example as a winner in item 19 — the real sin is a verdict the following text can't back, which Well-calibrated already states correctly.
  - Extend the "X, not Y" legit-use boundary with the anchor requirement (item 14).
  - Add overstate-then-walk-back as a tell (item 10).
  - Refine the em-dash entries to the paired-vs-trailing distinction.
  - Annotate that the list's register-heavy proportions reflect collection-time focus rather than the §3 hierarchy.
- **The passes-architecture question.** Does the separate-passes model (read-clean / plainly-written / well-constructed / well-calibrated) hold up, or want restructuring? Raised this session and deliberately not pulled.
- **The optimization pass on the new CLAUDE.md section.** It shipped deliberately detailed, to see how the high-calibration version performs before trimming.
- **Watch the next content-review runs** for whether the reviewer picks up `Well-constructed` and the new anti-pattern — its CLAUDE.md auto-load makes this free to observe.
- **Round-5 candidates**, if wanted: reflexive ", so" with real context, elimination-dressed-as-discovery, triadic lists, jargon density, hooky flattery — the tells still untested.

## Appendix — full item data

Format per item: what it tested, context where it mattered, variants verbatim, Claude's pre-committed prediction, Daniel's pick with the reason compressed, and the lesson.

### Round 1

**Item 1 — heavy gerund subject (convolution).** From a motivation list in 7.06-2. Predicted B; picked **B**.

- A: "Prevent the sprawling accretion of additive-gated variant code — the scattered if/elses that supporting every historical model variant in one codebase produces."
- B: "Prevent variant-code sprawl: supporting every historical model variant in one codebase scatters if/elses through the train loop, metrics, and logging."
- C: "Prevent the sprawling accretion of additive-gated variant code — the scattered if/elses that accrue from supporting every historical model variant in one codebase."

Reason: A worst by far — the "produces" arriving at the very end is "horrendous… very painful to read." B vs C close, B wins in isolation; context might flip it. Lesson: late-verb pain is top-severity.

**Item 2 — topic choice and the passive.** Constructed; a reference doc tracking what happens to a tool call. Predicted B; picked **B** ("too easy — good grounding check").

- A: "A hook may deny the call before it runs."
- B: "The call may be denied by a hook before it runs."
- C: "Denial by hook may occur before the call runs."

Lesson: topic stays subject; passive is the plain form when the topic is acted on.

**Item 3 — overload: two announcements plus asides.** Session-note status line. Predicted B; picked **B**.

- A: "The co-resident load test landed — v0 and v1 side by side, the main open risk from the triage — and the census found exactly two v1 checkpoints in existence, both smoke artifacts."
- B: "The co-resident load test landed, closing the triage's main open risk: v0 and v1 checkpoints now load side by side. Separately, the census found exactly two v1 checkpoints in existence, both smoke artifacts."
- C: "The co-resident load test (v0 and v1 side by side — the triage's main open risk) landed, and the census found exactly two v1 checkpoints, both smoke artifacts."

Reason: Daniel initially ranked A the clear worst, then walked it back next round — A and C are closer, parens beat dashes because their edges are visible, and an aside holding a compound clause is the real fault. (The prediction file's secondary ranking had C below A — the reverse of the initial read.) Lesson: the aside rules, and the doc's cleanest example of single-item noise.

**Item 4 — the "fine as written" plant that failed.** Ledger-spec line from 7.06-2 (A is the untouched original). Predicted A; picked **B** — the exercise's first correction of Claude's floor.

- A: "Bug fixes included — an in-place behavior-changing fix is precisely the class of change that goes unrecorded."
- B: "Bug fixes are included. An in-place fix that changes behavior is the class of change that tends to go unrecorded."
- C: "Bug fixes included, since in-place behavior-changing fixes are the class of change that goes unrecorded most often."

Reason: "behavior-changing" is invented compound jargon, "highly distracting, little to no gain." Follow-ups produced two more findings: a connected subordinate clause can beat two sentences when the words are plain (C's shape), and over-short sentences strand the reader's question across the period (the fragmenting floor). C also carried a masked overclaim salt ("most often"), partially caught. Lessons: compound-coinage tell, fragmenting floor, and B's quiet demotion of "is precisely the class" to "tends to go unrecorded" was preferred too.

### Round 2

**Item 5 — compound-modifier boundary.** Same arc; "co-resident" is established across the docs. Predicted A; picked **A**. Secondary ranking: C > B, "B's 'landed' at the end is… you know."

- A: "The co-resident load test landed: v0 and v1 checkpoints load side by side in one process."
- B: "The test that loads v0 and v1 checkpoints side by side in one process landed."
- C: "Co-resident checkpoint loading (v0 and v1 in one process) is now verified by a test."

Lesson: established-plus-glossed compounds are fine (the coinage tell needs the boundary), and the late verb again outranks other sins — the prediction file had ranked B above C.

**Item 6 — choppiness, and dash vs parens as a minimal pair.** Results line; facts: fp16 obs fixed n=20 starvation, training now GPU-bound at 2709 samples/sec, +30%. Predicted D; picked **toss-up A or D**.

- A: "The n=20 starvation is fixed. The fix is fp16 observations, now the default. Training is GPU-bound at 2709 samples/sec."
- B: "fp16 observations (now the default) fix the n=20 starvation, and training is now GPU-bound at 2709 samples/sec, a 30% gain."
- C: "Making fp16 observations the default fixed the n=20 starvation. Training is now GPU-bound at 2709 samples/sec — up 30%."
- D: same as C with "(up 30%)".

Lesson (with the later clarification): complete-thought choppiness has a low cost, not zero — the heavy penalty is reserved for fragments that strand a question.

**Item 7 — overclaim on a smooth ride, plus a semicolon probe.** Toolkit-validation wrap-up; only the one join was checked. Predicted B; picked **B**.

- A: "The join on `persp_val_index` is validated against real dumps, so the toolkit's outputs can be trusted going forward."
- B: "The join on `persp_val_index` is validated against real dumps. The toolkit's other paths are still unexercised."
- C: "The join on `persp_val_index` matches real dumps in the cases tested; the other paths are still unexercised."

Reason: caught A's overclaim (this round was after the round-1 salt was revealed, so the warned condition), and separately noted A reads smoothest if you ignore the facts — the masking mechanism observed first-hand. C came in close behind B, with semicolons rejected on prior ("misused very often" in AI writing). Lessons: overclaims are catchable when warned, and the semicolon ban holds even against a small precision gain.

**Item 8 — needless-words floor.** Census status line. Predicted B; picked **A** by a small edge, with "we shouldn't look too closely at this one."

- A: "A census found exactly two v1 checkpoints in existence, both smoke artifacts."
- B: "A census found exactly two v1 checkpoints, both smoke artifacts."
- C: "A census found that only two v1 checkpoints exist, and both are smoke artifacts."

Reason: "in existence" gently reinforces the point and gives the sentence breathing room. Lesson, held loosely: "omit needless words" has a floor.

### Round 3 (paragraph level)

**Item 9 — topic strings.** Module-landing paragraph; facts about the RawCheckpoint module. Predicted C; picked **C**, "mostly due to flow."

- A: "The RawCheckpoint module now wraps every handle load. Format facts moved into typed properties. One file read is all a handle load costs now. The old model_key fold-out code is gone."
- B: "The RawCheckpoint module now wraps every handle load. It exposes the format facts as typed properties, and it costs one file read per load. It replaces the old model_key fold-out code."
- C: "Handle loads now go through the RawCheckpoint module: one file read, with the format facts exposed as typed properties. The module replaces the old model_key fold-out code."

Reason: after deliberation, Daniel derived the principle himself — "handle loads" is the already-known topic, RawCheckpoint is news the reader must wait to resolve. Also flagged B's "costs" as mildly flourishy. Lesson: given-before-new, independently corroborated.

**Item 10 — accretion order vs constructed order.** Eval-results paragraph. Predicted C; picked **B**, "clear winner — best hierarchy of information and flow."

- A: "The eval pool used the frozen map set. TrueSkill was run over 400 games. The new checkpoint beats the previous best by a clear margin. Ratings were noisier than expected in early games, though, so the margin estimate is rough."
- B: "The new checkpoint beats the previous best, though the margin estimate is rough — early-game rating noise was higher than expected. The eval: TrueSkill over 400 games on the frozen map set."
- C: "On the standard eval (TrueSkill, 400 games, frozen map set), the new checkpoint beats the previous best. The margin estimate is rough: early-game rating noise ran higher than expected."

Bonus catch by Daniel: A "overstates the facts, and then self-corrects" ("by a clear margin" walked back a sentence later) — the overstate-then-walk-back tell, not deliberately planted. Lessons: conclusion-first with the qualifier integrated. The colon-fragment spec register ("The eval: …") is acceptable, and B won while carrying a trailing dash clause.

**Item 11 — journey vs destination.** Investigation summary. Predicted B; picked **B** — "strong flow and information structure; its first sentence captured both main points." A "very clear last place" (buried lede).

- A: "We first tried raising the worker count, which didn't help. Profiling then showed the obs-build step dominating. Batching the obs builds cut step time by 40%. So the pipeline was obs-build-bound, not IO-bound as assumed."
- B: "The pipeline was obs-build-bound, not IO-bound as we'd assumed. Profiling showed the obs-build step dominating, and batching the builds cut step time by 40%. Raising the worker count, the first fix tried, did nothing."
- C: "The pipeline was obs-build-bound: batching the obs builds cut step time by 40%. We'd assumed IO-bound — raising workers did nothing — until profiling showed obs-build dominating."

Refinements from Daniel: B's "the first fix tried" insertion is slightly awkward (subject-verb split), and C's paired-dash aside wants rewriting. Lesson: journey content is fine as demoted support; placement is the issue.

### Round 4 (wide net, minimal pairs)

**Item 12 — nominalization.** Predicted B; picked **B**, adding the topic caveat unprompted ("if 'validation' was the key topic, maybe A could win").

- A: "Validation of the join happens after each dump completes."
- B: "The join is validated after each dump completes."

**Item 13 — manufactured contrast.** Context: no one ever suggested an architecture explanation. Predicted A; picked **A** "by a mile" ("X, not Y when Y was completely out-of-band… *vomit*").

- A: "The starvation was an obs-pipeline issue."
- B: "The starvation was an obs-pipeline issue, not a fundamental architecture flaw."

**Item 14 — the contrast-anchor boundary (salt that went against the plan).** Context: the sibling endpoint really does fuzzy-match and the confusion is real, so A was planted to survive. Predicted A; picked **B**.

- A: "`validateUsername` checks byte-exact existence, not fuzzy matching."
- B: "`validateUsername` checks byte-exact existence."

Reason: a bare "not fuzzy matching" is "a dangling clause / incomplete thought" — worse than silence, because the reader must guess why it's said. Lesson: a real contrast still needs its anchor named ("unlike `replaysForUsername`, which fuzzy-matches") — this rewrote the tells list's "definitional contrast is fine" boundary.

**Item 15 — importance-signaling marker.** Operational README line. Predicted B; picked **A**.

- A: "Worth noting: the collector rate limit is per-endpoint, and the replay endpoint is the tight one."
- B: "The collector rate limit is per-endpoint, and the replay endpoint is the tight one."

Reason: "worth noting" does real routing work in operational context. Bigger finding: both variants fail on "is the tight one" — a cute compressed label hiding the concrete facts the reader needs (the replay endpoint hits the live prod game server, while the other endpoint is a cheap auto-scaling S3 bucket). Lessons: the marker tell targets reflexive use only, and vague-label compression is the deeper sin.

**Item 16 — staccato fragments.** Predicted B; picked **B** ("A might win at a spoken-word poetry jam").

- A: "The failure is silent. No exception. No log line. The dump just comes back empty."
- B: "The failure is silent: no exception, no log line, just an empty dump."

**Item 17 — reflexive ", so".** Predicted A; **abstained** — "the given context was too vague for me." The existing CLAUDE.md rule stands unvalidated.

- A: "Only two v1 checkpoints exist, and both are disposable smoke artifacts, so v1 is safe to change."
- B: "Only two v1 checkpoints exist, so v1 is safe to change."

**Item 18 — certainty vocabulary, subtle form.** One experiment, alternatives not ruled out. Predicted A; picked **A** ("B is way over-confident").

- A: "The gap is consistent with a missing army ranking."
- B: "The gap is explained by the missing army ranking."

**Item 19 — bold verdict lead-in (the designed uncertainty).** Review-style finding. Predicted B at low confidence; picked **A** — "lead with important info, cleaner sentence structure, better flow."

- A: "**It bundles an unrelated concern.** The dump path opens the store directly, so the loader's cache policy leaks into analysis code."
- B: "The dump path opens the store directly, so the loader's cache policy leaks into analysis code. The dump path is bundling a concern that isn't its own."

The pick contradicts the tells list's own example (A's lead-in is quoted there verbatim as a bad case). Resolution, confirmed with Daniel: the original sin was the "catchy banger that can't be backed up" — a calibration failure miscoded as a form tell. Conclusion-first stands when the next sentence cashes the verdict. Also flagged in B: "a concern that isn't its own" personifies the dump path — distracting. Lessons: the tells entry needs recoding, and B supplied the personification tell.

**Item 20 — intensifier floor.** No measurement exists; the judgment is honestly the author's. Predicted A; picked **B** ("if it's clear from context this is the author's judgment — minor for me").

- A: "The cleanup made the module easier to work with."
- B: "The cleanup made the module much easier to work with."

Lesson (seasoning queue): owned subjective intensifiers are fine; the target is intensifiers posing as measurement.

**Item 21 — over-definite "the real X".** Several fixes plausible, none decided. Predicted B; picked **B** ("A is overconfident").

- A: "The real fix is consolidating the config."
- B: "One candidate fix is consolidating the config."

**Item 22 — concessive reflex.** One-sided observation, no counterevidence collected. Predicted A; picked **A**.

- A: "Sonnet reviewers found real issues Fable missed."
- B: "Sonnet reviewers found real issues Fable missed. That said, both models have strengths."

Reason: B "sounds like vague hedging for unexplained reasons… there needs to be a reason for the statement." Lesson: hedges and balance must be earned, same as claims.
