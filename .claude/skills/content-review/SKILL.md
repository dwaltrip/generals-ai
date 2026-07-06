---
name: content-review
description: >
  Dispatch a blind cold read of prose artifacts — docs, commit messages, and
  the comments/docstrings in source files — to the content-reviewer subagent,
  then apply the findings and hand the user an attention-routing digest.
  Use when the user invokes /content-review with file paths, or propose it
  during a finishing pass on a substantial doc or a comment-heavy change.
  When self-initiated, always propose before running — never launch a review
  the user didn't ask for.
argument-hint: <path> [path ...] [prose notes — grouping, artifact description]
---

# content-review: caller procedure for blind prose review

This skill drives the content-reviewer subagent: a context-blinded cold reader that reviews prose for legibility. You (the main agent) scope the review, dispatch fresh reviewer instances, apply their findings, and report to the user. Design rationale and iteration notes: `docs/agent-tooling/content-review/notes.md`.

## Scoping

Explicit paths only. Bare invocation (no paths) is not supported — ask for paths rather than improvising a scope.

A **review unit** is the artifact set one reviewer instance sees.

- **Single path:** one unit — dispatch immediately, no confirmation.
- **Multiple paths:** state a one-line grouping plan — which files form one review unit, which get their own reviewer — and dispatch on the user's OK. Skip the confirmation when the invocation already states the grouping ("bundle these").

The rule behind the two cases: dispatch on scope the user settled, and confirm judgment you added. It holds across invocation channels — the slash command, a plain conversational request, a review you proposed and the user approved.

Grouping heuristics:

- Sibling code files usually bundle.
- A doc usually gets its own reviewer.
- Bundle code with its doc when comment-vs-doc redundancy is the question.

The principle: a unit contains what a realistic future reader would have in view — not what explains the prose. Over-bundling quietly un-blinds the read.

A unit may also carry a **context set**: companion files the reviewer reads but does not review — the reviewed prose may lean on them. The dispatch template has an optional slot for them. The same test governs what qualifies — would a realistic reader of the artifact have this file in view? A context set you assembled gets the same one-line confirmation as a grouping plan, with any stretch on the reader test noted. One the user named needs no ceremony.

Context files also catch drift. The reviewer checks the reviewed prose against everything it can see, so when the prose claims something about a companion file and the companion contradicts it, the mismatch gets reported. That is a benefit of files that belong in the set, and a temptation to add ones that don't. Every extra file makes the reviewer less of a stranger, and a reviewer who knows what the reader doesn't misses what would trip them. Wanting a claim checked doesn't qualify a file — the reader test does.

The artifact description — what it is and who reads it — is inferred from the files (`.md` → doc/session note; source files → code prose, comments and docstrings only) and can be overridden or extended in the invocation's prose.

## Dispatch

One Agent-tool call per review unit, in parallel across units. Always:

- `subagent_type: "content-reviewer"` — this literal string, never a substitute or fallback agent type.
- A fresh instance per dispatch. Never send a revision back to an instance that has already reviewed it — a re-consulted reviewer is no longer cold.
- The prompt is `dispatch-template.md` (next to this file) with its slots filled: artifact description, artifact set, optional context set, report path. Nothing else.

**Scope, not content.** The conversation may refine *scope* — which files, what grouping, the artifact description. It must never inject *content* into the dispatch: no background, no intent, no "the reader will know X," no steering toward or away from any part of the text. Anything that would pre-clear a finding or aim the reviewer's attention belongs after the cold read, in the digest, where the user resolves it holding full context. The template has no slot for such material; do not add one.

Reports go to `tmp/content-review/<yyyy-mm-dd>-<slug>/<unit-slug>[-<label>][-passN].md` — one report per unit.

- The directory names the review engagement on a target. `<slug>` usually derives from the target's basename, or is a short name for the effort when the engagement spans several units. Later passes on the same target join the existing directory while they're part of the same effort; a genuinely new engagement starts a new dated directory — a judgment call.
- `<unit-slug>` is the unit's file basename (shortened is fine), or a short group name for a bundle.
- A repeat pass appends `-pass2`, `-pass3`, …. A deliberately varied run (different model, grouping, effort) gets a short label instead: `-sonnet`, `-bundled`. A varied run repeated composes both: `-sonnet-pass2`.

## Post-dispatch checks

Before using any findings, verify for each unit:

1. The report file exists at the given path **and** ends with the reviewer's closing directive block (canonical text in `.claude/agents/content-reviewer.md`; it opens "These findings are for the human to judge"). The block is the authenticity marker: a report without it is treated as a failed dispatch — whether the dispatch silently resolved to a different agent type or the reviewer ignored its contract — and its findings must not be used.
2. No reviewed file was modified. The reviewer writes exactly one file (its report); any other diff is a contract violation to report to the user.

On failure: stop loudly and tell the user. No silent retry, no fallback agent. If the failure looks transient, at most one re-dispatch — note it and its outcome in the digest.

## Applying findings — fix-then-highlight

Before acting on any finding, check its quoted claim against the actual text — reviewers can overclaim, and severity is the reviewer's opinion, not ground truth.

Apply the findings. The digest routes the user's attention rather than asking permission:

- **Itemize** every judgment-heavy or multi-faceted edit: meaning-adjacent rewordings, trims of arguable substance, drops beyond obvious duplication, structural moves. The test for whether an edit is itemized: did it preserve the artifact's meaning exactly — every fact, claim, and calibration? If not, or if uncertain, itemize it.
- **Aggregate** mechanical fixes into one line with their finding IDs ("applied F2, F4–F6: wording clarity"). The diff is the audit trail.
- **Conditional findings** split by nature. If the condition is checkable, check it (read the definition, the doc) and note the resolution. If it's genuinely the user's call, leave the text unchanged and list it as awaiting the user.
- **Findings you decline to act on** are listed with the reason. Nothing is silently discarded — that is an invariant of the reviewer's directive, and it binds you.

A report may also carry **observations** — defects whose fix site is outside the reviewed set (the agent file defines them). Handle them like findings: verify the claim, then fix or route to the user.

## Digest

The digest is the user-facing summary, in this order:

1. Per-unit verdict(s) and report path(s).
2. Itemized judgment edits (ID, what changed, why it needed judgment).
3. Unresolved judgment calls awaiting the user (ID, the question).
4. Declined findings with reasons.
5. Observations — defects outside the reviewed set, with any action taken.
6. The aggregated mechanical-fixes line.

With multiple units, qualify IDs by unit (`train_loop.py F2`). Reviewers are mutually blind, so cross-unit synthesis is yours: the same pattern reported independently by several reviewers becomes one item ("same boilerplate comment in all three files — one fix, three sites").

If a verdict recommends a fresh read of the revision, propose a new invocation — a fresh instance on the revised text — and leave the decision to the user. One cold read per invocation. The user's diff review is the verification of your fixes.
