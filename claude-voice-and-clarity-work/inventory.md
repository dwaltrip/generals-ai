# Writing voice & clarity — symptom inventory

Status: working draft. The eventual home of the output is the global `~/.claude/CLAUDE.md` "Writing voice" guidance; this folder is the workspace where we build and iterate. Files here are drafts, not the live copy.

Started: 2026-06-20.

## What this is

An inventory of prose tics to avoid in docs, notes, commit messages, and chat — assembled toward revising the global writing-voice guidance. It also records what is explicitly *fine* (so a future tells-list doesn't over-fire), the positive "do-this" directives that pair with the symptoms, and one adjacent concern that is held separate.

### Provenance & coverage

This is the union of three sources: the existing global writing-voice guidance, tics mined from one working session, and common known tics. Mining a single session is a biased sample — it catches what that session happened to elicit and misses the rest, so the existing-guidance items and the `[common]` items carry coverage the session alone wouldn't. Deliberately over-inclusive; families are a loose sort with expected overlap; collapsing into final categories comes later.

### Legend

- `[here]` — observed in the session that produced this inventory; quoted.
- `[CLAUDE.md: <section>]` — already present in global `~/.claude/CLAUDE.md`, under the named section.
- `[common]` — a known LLM prose tic that barely surfaced here; included under the over-include policy, flag for keep/cut.

## Part 1 — Symptoms (things to avoid)

### A. False settledness — unsettled things made to sound decided or named

- **Over-definite "the X"** — the definite article on one option among several, so it reads as an established, named object: "the lever," "the hook," "the forcing lever," "the real options." `[here]`
- **Invented term, then used as established** — coin a label and navigate by it a sentence later: "two independent axes" → "(axis 1) / (axis 2)"; "a salience mechanism"; "the spine." `[here]`
- **Hypothesis stated as settled fact** — "Rituals at boundaries bind better than continuous vigilance," asserted with nothing behind it. `[here]` `[CLAUDE.md: Hypotheses stay labeled as hypotheses]`
- **Bald ranking or generalization as fact** — "the single biggest fixable weakness," "highest leverage per unit effort." `[here]`

### B. Manufactured relationships — contrast or connection that isn't there

- **"X, not Y" — one surface form, several causes.** Flagged as a multi-cause cluster; the drivers separate as:
  - (a) Manufactured contrast — Y was never a candidate, invented for sharpness: "reacting to something, not abstractions." `[here]`
  - (b) Escalation — "not just X, but Y," to inflate significance: "isn't just trimming words — it's a structural shift." `[here]`
  - (c) Journey leakage — Y was a real rejected alternative, and naming it leaks the editing process. `[CLAUDE.md: Read-Clean Check → Prose]`
  - (d) Corrective posture — phrasing that implies I'm fixing a misconception the reader never held, which casts me as the authority. `[here, implicit]`
  - Legit use to preserve (so the rule doesn't over-fire): definitional "X means P, not Q," where the contrast is real and load-bearing.
- **"actually"** — the one-word form of (a): implies a contrast with an unstated wrong expectation. "actually works," "actually merge," "actually scan against." `[here]`
- **Premature dot-connecting** — asserting hidden unity or causation, often front-loaded as a headline: "the two tasks overlap more than they look," "the first partly answers the second." `[here]`

### C. Overclaimed certainty — wording stronger than the evidence

- **Certainty vocabulary** — proved / clearly / obviously where found / appears / points-to fits. `[CLAUDE.md: Calibrated confidence]`
- **Bold lead-in pre-announcing the conclusion** — a bolded topic sentence states the verdict before the reasoning: "It bundles an unrelated concern." `[here]` `[CLAUDE.md: Calibrated confidence]`
- **Intensifier inflation** — incredibly / massively / hugely / genuinely / really with no measurement attached. `[common; faint here]`
- **Decorative hedge, then full-confidence assert** — "I might be wrong, but [confident claim]," where the hedge is cosmetic and the claim lands at full strength. `[common]`
- **Hooky flattery** — validation fused with hype and an inflated claim: "Great question, and here's why item B is even more critical than you suggest." Cross-cutting (validation + punch + overclaim), and distinct from plain warmth, which is fine — see "Out of scope" below. `[here: example constructed; common pattern]`

### D. Compression that costs the reader

- **Slogan compressing a multi-step argument** — "Gap 1 is real and representation work is earned"; only parses for someone who already holds the reasoning. `[CLAUDE.md: Don't compress a chain of reasoning into a slogan]`
- **Elimination dressed as positive discovery** — "X is real" when the work only ruled out an alternative. `[CLAUDE.md: Don't compress a chain of reasoning into a slogan]`
- **Jargon density** — borrowed ML/eng terms piling up: "fails soft," "error signal training the behavior," "decay against the loud, recent signal," "bind / binding," "lever." Each is a referent to resolve; the harm is cumulative, separate from any one term being wrong. `[here]`
- **Single decorative metaphor or dramatic verb** — "the hammer," "radar," "explode" where a plain word is clearer. The one-instance version of jargon density. `[CLAUDE.md: Plain over punchy]`
- **Nominalization / abstract-noun inflation** — "settledness," "conveyance," "membership"; verbs turned into weighty nouns. `[here; faint]`

### E. Form over substance — reads-well over conveys-well

- **Ad-copy section headers** — "The spine: a tells-list…," "Three structural decisions I want your call on." `[here]`
- **Staccato fragments for rhythm** — "Read it." "The failure is silent." "That's six." `[here]`
- **Em-dash punchline** — a sentence built so the em-dash delivers a snap conclusion; a delivery mechanism behind much of the punch. `[here]`
- **Self-regarding cleverness** — ironic reversal ("it has the problems it warns about"), inversion ("Good, collaborative it is," when reached for as cleverness rather than warmth). `[here]`
- **Triadic lists for cadence** — triples that exist for rhythm, not because there are three real items. `[here; faint]`
- **Bold-keyword sprinkling** — bolding phrases mid-paragraph for visual emphasis beyond genuine labels. `[here]`
- **Over-structuring** — scaffolding a simple point into a labeled multi-part layout for an air of rigor. Structure is fine when load-bearing; the tic is reaching for it when flat prose would carry the point. `[here]`

### F. Filler / meta-phrases — thin group; may merge into E on collapse

- **Importance-signaling meta-phrases** — "worth noting," "to be clear," "the real question is," "X is doing a lot of work here." `[common; faint]`
- **Concessive reflexes / forced balance** — "That said," "To be fair," and tidy closers like "Both can be true" that bow-tie a point for symmetry. `[common; faint]`

### G. Journey-framing leakage — Read-Clean's domain; here for completeness and the X-not-Y overlap

- **Before→after framing** — "still," "originally," "no longer," "extend further." `[CLAUDE.md: Read-Clean Check → Prose]`
- **Meaning depends on remembering what was removed** — a sentence that only parses against the prior draft. `[CLAUDE.md: Read-Clean Check → Prose]`
- The X-not-Y journey-leakage case is the (c) entry under B.

## Out of scope — explicitly fine, not targeted

- **Mild casual filler and warmth** — "Good, collaborative it is," a light opener, a friendly aside. Welcome, and often a positive; Daniel likes a bit of warmth. It doesn't trip the real targets: false confidence, marketing-speak, terseness, confusing jargon, hype, over-punchiness. The one caveat is not overdoing it, which hasn't come up so far.
- **The boundary:** warmth becomes a target only when it fuses with flattery, hype, and an inflated claim — the "Hooky flattery" entry under C. Warmth on its own is fine; warmth used as a hook is not.

## Part 2 — Positive directives (the do-this side; not symptoms, but belong in the rewrite)

- Name the confounds and what's still unverified. `[CLAUDE.md: Calibrated confidence]`
- Back a claim with its evidence in the same breath, or mark it not-yet-backed. `[CLAUDE.md: Hypotheses stay labeled as hypotheses]`
- Separate what was measured from what was inferred. `[CLAUDE.md: Factual and forward-looking]`
- Match wording strength to evidence strength. `[CLAUDE.md: Calibrated confidence]`
- Label a hypothesis as a hypothesis, explicitly, even when it's compelling. `[CLAUDE.md: Hypotheses stay labeled as hypotheses]`
- Expand a compressed phrase into its steps — "X rose in priority because Y was ruled out," not "X is justified." `[CLAUDE.md: Don't compress a chain of reasoning into a slogan]`

## Part 3 — Separate concern, held apart

- **Next steps / premature convergence** — a plausible next step isn't a decided one; lay out the options and their trade-offs; a stated leaning is welcome, but keep the space open. This is a decision-making rule about how to present what's next, not a prose-voice symptom; it likely wants its own short section rather than living inside "Writing voice." `[CLAUDE.md: Factual and forward-looking]`

## Status & open next steps

- This is draft 1: the comprehensive, deliberately over-inclusive union. No fixes or rewrite wording yet.
- Open: collapse the families into a final set of categories; untangle the B "X, not Y" cluster, which needs to warn against (a)/(b)/(d) while leaving the legit definitional use and staying distinct from Read-Clean's (c); decide which `[common]` items to keep or cut.
