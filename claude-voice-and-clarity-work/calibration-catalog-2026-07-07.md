# Calibration catalog — 2026-07-07 session

Date: 2026-07-07

## Scope

This is a session-isolated catalog and synthesis of the writing-preference calibrations that emerged from the 2026-07-07 working session, in which Daniel and Claude put the global `~/.claude/CLAUDE.md` under git tracking and reworked it end to end. Where the 2026-07-06 calibration exercise elicited preferences through controlled blind items, this session produced naturalistic evidence: preferences revealed live through what Daniel accepted, rewrote, rejected, and conceded across roughly fifteen rounds of propose-shape-apply. The catalog records those decisions with their evidence and reasoning. It is source material, deliberately upstream of curation — repetition with CLAUDE.md is expected and fine, because CLAUDE.md is the selected, operationalized artifact and this is the evidence base behind it.

The primary consumer is the planned integration task that will compare this catalog against the 2026-07-06 exercise doc. Secondary consumers are future curation passes over CLAUDE.md or other guidance.

## Methods

**Evidence strength.** Each entry carries a tag with two dimensions. How the preference surfaced: `stated` (Daniel said it outright), `edit` (revealed by his own rewrite of concrete text), `framing` (he accepted a proposal of Claude's — the weakest tier, since it may be Claude's preference standing unopposed), or `observed` (behavioral, never commented on). And how often: `×1`, `×2`, or `recurring`.

**Isolation.** The 2026-07-06 exercise doc was not opened during this work, and no comparison or relation-tagging against it was attempted. Known contamination, declared: the author read a paragraph-length summary of that doc (in a memory file) earlier the same session, and CLAUDE.md itself — which both sessions worked on — is shared ancestry. The item-level content of the prior doc was never in view.

**Sources.** The session conversation and the `~/dotfiles` diff. Quotes are Daniel's exact words with typos preserved — they are the evidence, so they stay verbatim.

**Discipline.** Entries state observations at the strength their evidence supports. A single-instance preference is recorded as one, not as a law.

## Session context

The session, in one arc: dotfiles repo created and global CLAUDE.md adopted verbatim under git, then a pure restructure (Daniel's own spike), then a long shaping phase — seam fixes, a self-consistency pass, a full scrub of the doc's nine prose semicolons with repairs sized to their context, example pairing, and promotion of the timing model into the finishing pass. The branch tells the story in four commits: skeleton, verbatim adoption, pure restructure, and one squashed pass Daniel titled "Big pass updating CLAUDE.md to follow its own writing rules." Three memories were updated along the way (commit-message content, commit cadence, the sober-writing arc record). The dotfiles repo is the diff-level record of everything that landed.

## Writing-style catalog

### Rules and strictness

**W1 — Strict semicolon scrub over rule softening.** `stated ×1`
Claude offered two resolutions for the doc's rule-vs-practice contradiction (scrub the nine prose semicolons, or soften the rule to allow balanced pairs) and leaned toward softening. Daniel chose the scrub: "yeah i'd like to scrub all the semi-colons." No rationale was stated at that moment — see W2 for the rationale he later gave in a parallel case. Landed: the doc's prose is semicolon-free except the deliberate specimen in the semicolon rule itself.

- Before: "The `-S` flag enables env's split-string mode; without it the kernel treats `uv run python` as a single literal program name and the shebang fails."
- After: "The `-S` flag enables env's split-string mode. Without it, the kernel treats `uv run python` as a single literal program name and the shebang fails."

**W2 — Personification as an operational near-redline.** `stated ×1`
Daniel: "I'm thinking almost always that's an immediate redline (The 'almost' is a hedge for my uncertainty about if there are use cases I might not actually hate)." When Claude pushed back with the dead-idiom vs. fresh-coinage distinction, he held the stance and revealed its basis: "my strong stance was operational move: I've just noticed a lot of what I consider *quite bad* and overly cute personification as a major Claude tendency." He accepted an established-usage carve-out. Landed: personification is its own tell in Well-constructed, with the carve-out ("the compiler complains" is fine).

- Before: `a personifying idiom ("a concern that isn't its own")`
- After: `A personifying idiom coined for color: "a concern that isn't its own". Established usage ("the compiler complains") is fine.`

### Construction and structure

**W3 — Umbrella hierarchy over flat headings.** `stated + edit ×1`
He rejected a flat all-`##` scheme — "the flat doesn't feel right. We should have an umbrella section for all writing related guidelines" — and executed the reorganization spike himself rather than specifying it. Landed: the doc's four top-level sections.

**W4 — Order sections by size and use-frequency.** `stated ×1`
He originated moving read-clean below the three short checks, citing its size: short high-frequency rules first, the long treatise after. Landed: the writing section's order.

**W5 — Subsection headings over bold-paragraph units.** `edit ×1`
His rewrite of the convoluted-construction lead introduced `###` headings for the failure classes. Adopted section-wide, including for Repair and Paragraph level (which would otherwise nest wrongly in the document outline). Landed: Well-constructed's five subsections.

- Before: `**Overload — the sentence parses fine but carries too much, or weights it wrong.** Tells:`
- After: `### Overload` followed by `The sentence parses fine but carries too much, or weights it wrong. Tells:`

**W6 — Structural repair over surface repair.** `stated ×2`
He rejected both the original asides bullet and a minimal period-split fix: "It feels like it is doing too much at once, and doesn't quite get there." The accepted repairs were structural — the delimiter rule relocated out of the tells list, and the overloaded repair paragraph rebuilt as three bulleted rules. Landed: both reworks.

- Before: "An aside holding more than a single clause. Paired dashes are the worst delimiter for asides; parens at least show their edges. A trailing dash introducing one explanatory clause is fine."
- After: "An aside holding more than a single clause — though a trailing dash introducing one explanatory clause is fine."
- Relocated: "When an aside earns its place, delimit it with parens rather than paired dashes — parens show where the aside begins and ends."

**W7 — Inferential distance between joined clauses as a sentence test.** `stated ×1`
His critique of "Grammatical correctness is not the bar — the reader pays extra parsing effort…": "The two parts to that sentence are too loosely / tangentially related. There is like 2 or 3 jumps at least between them." He originated counting the inferential jumps a connective papers over. The repair dropped the jump-generating metaphor. Landed: indirectly (the final #2 intro). The test itself is not codified in CLAUDE.md.

- Criticized: "Grammatically correct is not the bar — the reader pays extra parsing effort and gets nothing for it."
- Repaired: "The friction is pure downside: the reader spends more parsing effort and gains nothing."

**W8 — Implied tests get made explicit.** `stated ×1`
Defending "intentional and thoughtful," he articulated the test it gestured at — "If it can be worded more simply without losing anything, then the complexity was fluff and misguided cleverness" — and wanted it explicit rather than implied. Landed: the convoluted-construction test line's closing clause, "the complexity was fluff, not craft."

- Before: "If the plain construction says the same thing, use it."
- After: "If the plain construction says the same thing, use it — the complexity was fluff, not craft."

**W9 — The fragment floor binds his drafts too.** `edit ×1`
His blend contained a definitional fragment ("Specifically, cases where the friction is pure downside:"). Flagged, he accepted the de-fragmented version as "perfect." Landed: behavioral — no doc change beyond the final wording.

- Before: "Specifically, cases where the friction is pure downside: the reader spends more parsing effort and gains nothing."
- After: "The friction is pure downside: the reader spends more parsing effort and gains nothing."

**W10 — Doc-wide cohesion outranks local improvement.** `stated ×1`
On learning his rewrite duplicated a test that already had an authoritative home four lines down: "my edit failed a cohesion pass with the bigger picture. make sense. good call out." Landed: behavioral.

- His draft: "The constrution must be intentional and thoughtful — any extra complexity massively hampers legibility."
- Already in the doc, four lines down: "The test: topic as subject, action as verb, the point in the main clause. If the plain construction says the same thing, use it."

### Plainness and register

**W11 — Intensifiers and vague judgment words rejected in his own prose.** `edit ×1`
His draft's "massively hampers" and "sloppy writing" fell to the doc's own rules (banned intensifier class, judgment word carrying no check) via critique — never explicitly conceded, but his next blend dropped both. Landed: the final #2 intro.

- His draft: "Grammatical correctness is not an excuse for sloppy writing. The constrution must be intentional and thoughtful — any extra complexity massively hampers legibility."
- Landed: "The friction is pure downside: the reader spends more parsing effort and gains nothing."

**W12 — Worn technical metaphors pass, jump-generating abstractions don't.** `edit + framing ×1`
The accepted blend replaced "the bar" (which required cashing out) with "friction" (worn smooth in engineering prose). The trade was flagged and accepted knowingly. This mirrors the established-vs-coined line the doc draws for compound modifiers and idioms. Landed: the #2 intro wording.

- Before: "Grammatically correct is not the bar — the reader pays extra parsing effort and gets nothing for it."
- After: "The friction is pure downside: the reader spends more parsing effort and gains nothing."

**W13 — Logical quotation over American publishing style.** `stated ×1`
Asked whether period-outside-quotes is idiomatic, and told about the American-vs-logical split: "the 'american publishing style' sound a bit silly to me… it does feel more logical." Landed: ratified the doc's existing practice.

- Doc (logical): `Call a fix "the change", not "the hammer".`
- American equivalent (constructed for contrast, not a session artifact): `Call a fix "the change," not "the hammer."`

**W14 — Anti-compression applies inside examples.** `edit ×1`
His nit on the over-example replaced the compressed placeholder "the 'which? why?' it raises" with spelled-out questions: "the obvious questions of 'which bug fixes? why are they included?' are left hanging at the period." Landed: the over-example's annotation.

- Before: `— the "which? why?" it raises is stranded across a period.`
- After: `— the immediate questions ("which bug fixes? why are they included?") are left hanging at the period.`

**W15 — Guard clauses go when the main claim carries the load.** `edit ×1`
He dropped ", not a violation" from the passive-voice sentence, noting "I could be convinced to bring it back in some form." The claim "passive voice is the plain choice" already implied it. Landed: the base rule.

- Before: `… the passive is the plain choice, not a violation ("the join is validated after each dump completes").`
- After: `… passive voice is the plain choice. Example: "the join is validated after each dump completes".`

**W16 — Symmetry in parallel constructions.** `edit ×1`
He dropped "the" before active/passive voice so "both have 'voice'" — parallel form for a rule-pair. Landed: the base rule.

- Before: "Active voice when the actor is the topic; when the topic is the thing acted on, the passive is the plain choice …"
- After: "Use active voice when the actor is the topic. When the topic is the thing acted on, passive voice is the plain choice."

### Examples pedagogy

**W17 — Pair shown negatives with their positives, scoped.** `stated + framing ×1`
He recalled the principle — "always include a postive example with a negative exmaple… this may affect any 'tells' sections" — triggered by re-reading a just-landed bullet. He accepted the scoping that "always" overshoots: pair where the repair isn't derivable from the tell, and don't add examples to example-free tells. Landed: two pairs added (afterthought qualifier, pronoun tell).

- Before: `A pronoun with an ambiguous antecedent: "the loader checks the header before the body — it may be absent" (either noun fits "it").`
- After: `… (either noun fits "it"). The fix is naming the noun: "— the header may be absent".`

**W18 — Examples must force the taught fix.** `stated ×1, co-produced`
On learning that a lighter fix ("which") partially works on the pronoun specimen: "this means the example is not the right one then." The fix was flipped to name the far noun — the case where only the taught repair works. Landed: the flipped fix plus its anti-"which" parenthetical.

- Unflipped: `The fix is naming the noun: "— the body may be absent".`
- Flipped: `The fix is naming the noun: "— the header may be absent". (A "which" would have silently picked the body.)`

**W19 — Specimens self-label, gated by a coherence check.** `stated ×1`
He asked to fit the label "incomplete thought" into the over-example, and asked in the same breath "would that cohere with rest of the doc?" — the label ties the specimen to the rule it violates, and additions get checked against the whole before landing. Landed: the labeled over-example.

- Before: `"Bug fixes are included." — the "which? why?" it raises is stranded across a period.`
- After: `"Bug fixes are included." — an incomplete thought: the immediate questions ("which bug fixes? why are they included?") are left hanging at the period.`

**W20 — Display patterns over inline quoted examples.** `stated ×1`
On the repair bullets: "the most stand-out issue is that the multiple inline quoted examples, as written, make it hard to read." The prefer/over sublist pattern was the accepted answer, and he demanded a second finishing pass on a version he had already called "a strong step in the right direction." Landed: the repair bullets and the afterthought pair.

- Before (one sentence of the crammed paragraph):

  ```
  Demotion has a floor: don't fragment below a complete thought — "Bug fixes are included." strands the reader's immediate "which? why?" across a period; keep that connection attached ("Bug fixes are included, since an in-place fix that changes behavior is exactly the class that goes unrecorded").
  ```

- After:

  ```
  - **Demotion has a floor: a complete thought.**
    - **Prefer this:** "Bug fixes are included, since an in-place fix that changes behavior is exactly the class that goes unrecorded."
    - **Over this:** "Bug fixes are included." — an incomplete thought: the immediate questions ("which bug fixes? why are they included?") are left hanging at the period.
  ```

**W21 — Labeled examples over parenthetical ones.** `edit ×1`
He converted a parenthetical example into a labeled sentence: 'Example: "the join is validated after each dump completes".' Consistent with the doc's existing "Example:" usage. Landed: the base rule.

- Before: `… the plain choice, not a violation ("the join is validated after each dump completes").`
- After: `… the plain choice. Example: "the join is validated after each dump completes".`

### Guidance content and artifact architecture

**W22 — Operational defaults over option menus.** `edit ×1`
He rewrote the post-promotion memory disposition from an open choice ("removed … or kept … — also their call") to a default with an exception: "should be removed as redundant, unless there is a specific reason to keep it (e.g. it extends the general rule with project-specific aspects)." A style pass is allowed to upgrade content when the content is mushy. Landed: the memory-promotion section.

- Before: "After they say yes, write the global entry; the project-scoped memory can be removed (now redundant) or kept (a project-scoped echo doesn't hurt) — also their call."
- After: "After they say yes, write the global entry. The project-scoped memory should be removed as redundant, unless there is a specific reason to keep it (e.g. it extends the general rule with project-specific aspects)."

**W23 — Context-inclusive repair over symptom fix.** `stated ×1, validated`
"can you redo items #2 thru #5 including the immediate surrounding context and applying all of the guidelines to that snippet? that will be more useful I think." Validated by outcomes: one repair became a deletion (the clause restated its bullet's opening), another merged with an already-flagged structural fix. Landed: the method used for every remaining repair in the session.

- Before: `Established terms ("load test", "smoke test") are fine; the tell is fresh coinage.`
- After: `Established terms ("load test", "smoke test") are fine.`

**W24 — Two-tier artifact architecture.** `stated ×2`
"I want to de-couple this synthesis + cataloging work from the content that goes into the CLAUDE.md… this is source material," and: "They may repeat, which is totally fine. The difference in my mind is: selection and curation, then careful application and operationalization into this very improtant doc that affects every session." Landed: this document's charter.

**W25 — Supersede-and-delete over parallel maintenance.** `stated ×1`
"the tells-doc might be just deleted. the calibraiton exercise doc is the more recent and much more detailed / careful artifact." Preference for deleting a superseded artifact over reconciling it with its successor. Landed: pending — the deletion itself hasn't happened.

**W26 — The contagion hypothesis.** `stated ×1, hypothesis`
"I'm realizing a lot of the issues we've been having in recent weeks were likely amplified by a CLAUDE.md that was demonstrating the opposite of what we wanted." Stated as a likelihood, and recorded here at that strength: guidance whose own prose violates its rules trains against itself every session. This was the session's motivating insight, articulated near its end. Landed: as motivation throughout; recorded in the sober-writing memory.

### Genre

**W27 — Commit messages are a terser genre with the same construction floor.** `stated + edit ×2`
He deleted one message as "a massive, overly detailed, unorganized run-on paragrpah," requested "Less is more I think, focusing on the biggest details, and quite broad-stroke summary for the rest," and when the second attempt still packed three items into one sentence, rewrote it himself — one imperative changelog line per change, broad-stroke closer kept — and explained: "commit messages are a bit different, they can be a bit terser." Register loosens by genre. One-point-per-unit does not. Landed: the commit-message-content memory.

- Claude's version (the second attempt, which he rewrote):

  ```
  Make the doc's prose exemplify its own rules

  The big items: all nine prose semicolons scrubbed with repairs sized
  to their context, Well-constructed's failure classes promoted to
  subsections, and the in-flight/end-of-session timing model moved from
  read-clean to the finishing pass, which now defines both membership
  and operation. The rest is small polish throughout: repaired
  cross-references, negative examples paired with their fixes, naming
  collisions resolved, and a firmer default for post-promotion memory
  cleanup.
  ```

- His rewrite:

  ```
  Big pass updating CLAUDE.md to follow its own writing rules.

  Scrub prose semicolons, polishing the surrounding context in general.
  Promote well-constructed's failure classes to subsections.
  The in-flight/end-of-session timing model is now generalized to
  the overall finishing pass.

  Small polish throughout: repaired cross-references, negative examples
  paired with their fixes, naming collisions resolved, and a firmer default
  for post-promotion memory cleanup.
  ```

**W28 — House style binds every prose artifact.** `stated ×1`
"can you fix up the README, it doesn't honor house style" — about a personal-repo README. He named no specifics, expecting self-diagnosis from the rules (the violations found and accepted: a fragment opener, semicolon joins, a restated opener). Landed: the README fix, and behaviorally.

- Before: "Authored config, tracked in git and symlinked into place."
- After: "This repo tracks hand-authored config in git and symlinks it into the locations where tools read it."

## Process catalog — shaping-loop findings

Restricted to how the shaping loop itself worked best. Broader collaboration and interaction-dynamics findings from the session are reserved for a separate follow-up doc; their evidence is retained in this doc's facts file.

**P1 — Judgment work in batches of one or two.** `stated ×1`
"can you propose edits for 1-2 of each remaining semi-colon at a time."

**P2 — Sequence by impact per diff size.** `stated ×1`
"lets fix this small-easy edits (diff-wise small, impact large)" — small high-impact fixes first, judgment-sized items deferred and named.

**P3 — Independence engineering.** `stated, recurring`
Fresh reads requested before he shares his own thoughts ("I have some but I want to your latest thoughts"), the strict-isolation tightening for this catalog was his initiative ("the borrowing could bias things. ya?"), and he audited the author's contamination state before proceeding ("did you read it earlier?").

**P4 — Careful kickoff with explicit checkpoints.** `stated ×2`
"don't begin yet. lets just discuss the concept. I'd like kick it off carefully," then "pause after the frame plz." Concept discussion before work, checkpoints where review is cheap.

**P5 — The de-densify probe.** `stated ×1`
"can you de-densify the 'structure commit' section for me? what is it saying" — an unpack request that doubles as a test of whether prose is hiding content (the test reading is the author's interpretation). The unpack surfaced a justification the dense version had swallowed.

- Before: "evidence strength (stated-explicitly / revealed-by-edit / accepted-my-framing / n=1 vs recurring)"
- After: "how the preference surfaced (stated outright / revealed by Daniel's own edit / merely accepted from my proposal — the weakest, since it may be my framing unopposed)"

**P6 — Thoroughness under-filters without a sharp external criterion.** `stated ×1, self-referential`
This catalog's own process section over-expanded from a one-line frame charter to fifteen entries and survived the outline filter until Daniel's check: "is that plus a natural 'completionist' tendency? … I've noticed you can be quite thorough, which is often very helpful, but sometimes need to be checked." His diagnosis connected the tendency to the over-detail and overload failures generally. When the selection criterion is vague, "is this true and potentially useful?" silently replaces "does the frame need this?" — and almost everything passes the first test. The user check is part of the loop.

**P7 — Evidence means artifacts, not only speech.** `stated ×1`
The draft catalog quoted Daniel's meta-commentary verbatim while merely summarizing the text artifacts under discussion, and he called the gap: "I feel like the commentary and analysis is much less useful in the future without the examples to ground and demonstrate." The specimen pass that followed rebuilt the entries around before/after text. The frame-level lesson extends P6: a frame bar that isn't cashed into a checkable deliverable property (here, "every edit-tier entry shows its specimen") gets silently replaced by an executable default (quote the speech).

## Synthesis — cross-cutting patterns

Interpretations over the entries above, each citing its support. These are the author's generalizations, not Daniel's statements, except where an entry says otherwise.

**S1 — Rule strictness is an operational lever, not a claim about English.** (W1, W2. Rationale explicit once, pattern twice.) Both times a strict-vs-nuanced rule choice arose, Daniel chose strict, and the one rationale he gave was about counteracting a known model tendency rather than describing good English. A bright-line rule over-forbids a little and blocks the failure mode, while a nuanced rule leaves the biased reader room to apply it unevenly.

**S2 — Self-exemplification: guidance teaches through its own prose.** (W26, W28, W10. A stated hypothesis plus two behavioral confirmations.) The doc's prose is training signal every session, so it must demonstrate its rules, not merely state them. This was the session's motivating insight and its largest single implication for how guidance docs get written.

**S3 — Examples are engineered artifacts.** (W17–W21. The session's densest cluster.) Five separate decisions treat examples as designed objects: paired with their fixes, engineered so the taught repair is the only adequate one, self-labeled with the failure they illustrate, and displayed via patterns rather than inline.

**S4 — Structure before sentence repair, in prose as in code.** (W3, W5, W6, W20. Recurring, mixed stated/edit.) When a unit resists a clean sentence-level fix, the recurring resolution was structural: headings, bullets, relocation, display patterns. CLAUDE.md states this principle for code comments; the session confirmed he holds it for guidance prose.

**S5 — Genre modulates register, never the construction floor.** (W27, W28. n=2.) Commit messages earn terser idiom than docs, and a README earns full house style despite its size. What flexes by genre is register; one-point-per-unit and the fragment floor bind everywhere.

**S6 — Rules apply symmetrically.** (W9, W10, W11. Recurring.) His own drafts were subject to the same critique as Claude's, and he accepted rule-grounded pushback on them without friction — twice conceding outright. The calibration relationship runs on argument quality, not authorship.

**S7 — Independence and blinding are standing method.** (P3, plus this catalog's own isolation decision. Recurring.) Blinding is not confined to the content-review tooling — it shows up wherever a judgment could anchor on another judgment: fresh reads before disclosure, isolated derivation before comparison, contamination audits before proceeding.

**S8 — Rules must cash out into runnable tests.** (W7, W8. n=2, both stated.) The two rule-shaped things he originated this session were both tests: the simpler-wording test behind "intentional and thoughtful," and inferential-jump counting between joined clauses. A rule that can't be run doesn't reliably get followed.

**S9 — Two-tier artifact architecture: one operative home, repeating evidence homes.** (W24, W25, plus the duplicate-trigger drop in the timing-model move. Stated.) CLAUDE.md is curated and operational; catalogs like this one are upstream evidence; repetition across tiers is fine while duplication *within* the operative tier gets deleted.

**S10 — He models the target register.** (W2, W26. Observed, recurring.) His own speech carries the calibration he wants in the docs — hedges labeled as hedges, hypotheses stated as likelihoods. The register being asked for is the one being used.

## Closing note

This catalog makes no comparison with the 2026-07-06 exercise doc and no curation decisions — both are deliberately out of scope. Its intended consumers are the deferred integration task, which should weight convergences in light of the contamination declared in Methods, and the planned follow-up doc holding the collaboration findings filtered out of the process section.
