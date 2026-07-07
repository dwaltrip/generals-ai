---
name: content-review
description: >
  Dispatch a blind cold read of prose artifacts — docs, commit messages, and
  the comments/docstrings in source files — to the content-reviewer subagent,
  then apply the findings and hand the user an attention-routing digest.
  Use when the user invokes /content-review with file paths, or propose it
  during a finishing pass on a substantial doc or a comment-heavy change.
  Always propose before running — never launch a review silently.
argument-hint: <path> [path ...] [prose notes — grouping, artifact type]
---

# content-review: caller procedure for blind prose review

This skill drives the content-reviewer subagent: a context-blinded cold reader that reviews prose for legibility. You (the main agent) scope the review, dispatch fresh reviewer instances, apply their findings, and report to the user. Design rationale and iteration notes: `docs/agent-tooling/content-review/agent-notes.md`.

## Scoping

Explicit paths only. Bare invocation (no paths) is not supported — ask for paths rather than improvising a scope.

- **Single path:** dispatch immediately, no confirmation.
- **Multiple paths:** state a one-line grouping plan — which files form one review unit, which get their own reviewer — and dispatch on the user's OK. Skip the confirmation when the invocation already states the grouping ("bundle these").

A **review unit** is the artifact set one reviewer instance sees. Grouping heuristics: sibling code files usually bundle; a doc usually gets its own reviewer; bundle code with its doc when comment-vs-doc redundancy is the question. The principle: a unit contains what a realistic future reader would have in view — not what explains the prose. Over-bundling quietly un-blinds the read.

Artifact type is inferred from the files (`.md` → doc/session note; source files → code prose, comments and docstrings only) and can be overridden in the invocation's prose.

## Dispatch

One Agent-tool call per review unit, in parallel across units. Always:

- `subagent_type: "content-reviewer"` — this literal string, never a substitute or fallback agent type.
- A fresh instance per dispatch. Never send a revision back to an instance that has already reviewed it — a re-consulted reviewer is no longer cold.
- The prompt is `dispatch-template.md` (next to this file) with its slots filled: artifact set, artifact type, report path. Nothing else.

**Scope, not content.** The conversation may refine *scope* — which files, what grouping, what artifact type. It must never inject *content* into the dispatch: no background, no intent, no "the reader will know X," no steering toward or away from any part of the text. Anything that would pre-clear a finding or aim the reviewer's attention belongs after the cold read, in the digest, where the user resolves it holding full context. The template has no slot for such material; do not add one.

Reports go to `tmp/content-review/<yyyy-mm-dd>-<slug>/<unit-slug>.md` — one directory per invocation, one report per unit.

## Post-dispatch checks

Before using any findings, verify for each unit:

1. The report file exists at the given path **and** ends with the reviewer's closing directive block. The block is the authenticity marker: a report without it is treated as a failed dispatch — whether from name-resolution fallback or a reviewer that didn't follow its contract — and its findings must not be used.
2. No reviewed file was modified. The reviewer writes exactly one file (its report); any other diff is a contract violation to report to the user.

On failure: stop loudly and tell the user. No silent retry, no fallback agent. If the failure looks transient, at most one re-dispatch, stated in the digest either way.

## Applying findings — fix-then-highlight

Apply the findings; the digest routes the user's attention rather than asking permission:

- **Itemize** every judgment-heavy or multi-faceted edit: meaning-adjacent rewordings, trims of arguable substance, drops beyond obvious duplication, structural moves. The test for flag-worthiness: did the edit preserve the artifact's meaning exactly — every fact, claim, and calibration? If not, or if uncertain, itemize it.
- **Aggregate** mechanical fixes into one line with their finding IDs ("applied F2, F4–F6: wording clarity"). The diff is the audit trail.
- **Conditional findings** split by nature: if the condition is checkable, check it (read the definition, the doc) and note the resolution; if it's genuinely the user's call, leave the text unchanged and flag it.
- **Findings you decline to act on** are listed with the reason. Nothing is silently discarded — that is an invariant of the reviewer's directive, and it binds you.

## Digest

The digest is the user-facing summary, in this order:

1. Per-unit verdict(s) and report path(s).
2. Itemized judgment edits (ID, what changed, why it's flagged).
3. Unresolved judgment calls awaiting the user (ID, the question).
4. Declined findings with reasons.
5. The aggregated mechanical-fixes line.

With multiple units, qualify IDs by unit (`train_loop.py F2`). Reviewers are mutually blind, so cross-unit synthesis is yours: the same pattern reported independently by several reviewers becomes one item ("same boilerplate comment in all three files — one fix, three sites").

If a verdict recommends a fresh read of the revision, propose a new invocation — a fresh instance on the revised text — and leave the decision to the user. One cold read per invocation; the user's diff review is the verification of your fixes.
