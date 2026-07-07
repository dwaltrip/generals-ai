# content-reviewer — maintainer notes

Covers the content-reviewer agent (`.claude/agents/content-reviewer.md`) and, until the companion skill has its own notes, the caller-side design of the content-review skill (not yet built). The two are one system: the agent is the stable contract, the skill is the procedure we expect to iterate on.

Status (2026-07-06): agent v1 drafted, not yet smoke-tested. Skill not started — its open decisions are listed under Caller-side design.

## Origin

Adapted from an external draft agent definition (found copy at `zzz-scratch/content-review-agent.md`). Kept: the cold-read blinding mechanism, legibility lens, anti-pattern list, severity tiers, verdict-as-posture, two-voices rule. Cut: the source system's "dispatch covenant"/"governance layer" framing, the corporate exposure-lens machinery, stop-and-wait gating, and most of the persona theatrics.

## Load-bearing design decisions

**Custom agent over prompt-template-in-skill.** The template route (obra/superpowers) keeps the pair in one directory and has no name-resolution risk, but all its constraints are behavioral. The custom agent gives enforced capability limits via `tools:` and puts the contract in the system prompt, where adherence is strongest. Name-resolution mitigations: the skill pins the literal `subagent_type` string; bring-up includes a resolution smoke test; the closing directive block doubles as an authenticity marker — a report missing it is treated as a failed dispatch (a fallback to some other agent, or a reviewer that didn't follow its contract) and its findings must not be used.

**Blinding is conversational, not informational.** The reviewer is denied the conversation that produced the artifact, not knowledge of the repo. Two accepted leaks, both deliberate: (1) CLAUDE.md auto-loads into custom subagents — used as the policy-delivery channel (next item); (2) Read is behaviorally scoped to the handed paths — no per-agent path scoping exists in the harness (checked 2026-07: frontmatter `tools:` takes bare names, permission rules are session-wide, PreToolUse hook input doesn't identify the calling subagent). Available hardening if ever needed: subagents support their own frontmatter `hooks:`, so a PreToolUse hook in the agent file could validate paths.

**House policy arrives via CLAUDE.md auto-load.** The agent file carries one generic line (prose/comment policy in project context is governance context to apply); AGENTS.md stays the single authoritative home of the rewrite-trim-drop test. Portability follows: in another repo, that repo's CLAUDE.md supplies its own policy. This depends on the documented auto-load behavior — verify at smoke test.

**One framework plus code deltas.** Doc and code review share the legibility lens and anti-patterns. The "when the artifact is source code" section lists only the genuine differences (the deltas live in the agent file, not here). Chosen over parallel doc/code modes, which would duplicate most of the file.

**Single lens; conditionality is a property, not a tier.** Exposure demoted from a second lens to one anti-pattern entry. Any finding at any severity may be conditional and must state its concrete resolving condition. Findings get stable IDs (F1, F2, …) for reference across digest and discussion.

**Directive states invariants only; the dial lives in the skill.** Agent-side invariants: the complete report plus an itemized digest of judgment items reaches the human; no finding silently discarded; drops of content, trims of substance, and conditional resolutions are the human's call. How much the caller fixes directly (the "dial") is caller procedure with one authoritative home, the skill.

**Write access with a bright line.** `tools: Read, Write` — the reviewer writes its own report file. The alternative (caller transcribes the returned report) passes the full text through the caller's context twice and risks copy corruption, a weaker provenance story than reviewer-written files. The constraint is behavioral: exactly one file, at the caller-given path, never a reviewed file. A diff in a reviewed file after a review round is a loud violation signal.

**`model: inherit` for bring-up.** Validate the design at full model strength first, so reviewer weakness can't be mistaken for design weakness. Sonnet A/B afterwards — see Planned experiments.

## Caller-side design (agreed, not yet built)

- **Fix policy — fix-then-highlight (settled 2026-07-06):** the caller applies the findings and the digest routes attention rather than seeking permission: judgment-heavy or multi-faceted edits itemized, mechanical fixes aggregated, findings not acted on listed with reasons. Conditionals split by nature: resolvable-by-verification ones the caller resolves and notes; genuine judgment calls are left undone and flagged. Rationale: the historical failure modes were finding issues (thoroughness, contagion-blindness) — now the agent's job — not fixing clearly-pointed-out ones. An earlier gate-then-fix design (caller queues fixes for approval) was dropped as solving the wrong problem.
- **Rounds — single round (settled 2026-07-06):** one cold read per invocation; the human's diff review is the fix-verification step. Re-review is just another explicit invocation, proposed by the caller when the round-1 verdict recommends a fresh look. A re-consulted instance is never used — it's no longer cold and grades its own suggestions; any re-review is a fresh instance.
- **Reports:** `tmp/content-review/<date>-<slug>/round-N.md`, reviewer-written, never edited by the caller; caller annotations live in the digest. The tmp files decay — durable lessons land here or in docs, not in report files.
- **Review unit:** single file by default, dispatched in parallel per file. A multi-file bundle is the unit when cross-file duplication or comment-vs-doc redundancy is the question. Bundle principle: include what a realistic future reader would have in view, not what explains the prose — over-bundling quietly un-blinds the read. Instances are mutually blind, so cross-file pattern synthesis is the caller's job.
- **Invocation (settled 2026-07-06):** explicit paths only; single path dispatches immediately; multiple paths get a one-line grouping plan confirmed before dispatch (bundling is contextual — sibling code files usually bundle, a doc gets its own reviewer); grouping and artifact-type overrides in prose, no flags; never auto-run — the caller proposes. Bare no-args invocation (scope from uncommitted changes) deferred: it stacks file-selection and grouping judgment; revisit once grouping intuitions are established.
- **Remaining detail** (report/digest formatting, failure handling — authenticity-marker check, loud stop, no silent fallback) lands in SKILL.md directly.

## Parked alternatives and iteration ideas

- **Detect-then-validate** (Anthropic code-review plugin pattern): no fix-and-rerun loop; a second wave of validator agents re-checks each finding and unvalidated ones are dropped. The fallback if fix-and-rerun proves noisy.
- **Autonomous mode:** deferred until finding-quality data exists. Would relax the dial per-run; the agent-side invariants are written to hold unchanged.
- **Glossary injection:** skipped in v1 (CLAUDE.md covers some project vocabulary). Revisit if term-triage flags get noisy.
- **Shared-vocabulary skill** preloaded into both sides (agents support a `skills:` frontmatter field) if agent/skill drift actually bites.
- **Plugin-ization:** the documented route for versioning the pair as one shippable unit.
- **Frontmatter PreToolUse hook** to enforce the report-write path structurally.
- **Multi-round fix-and-rerun loop** (up to N rounds, early exit on a quiet round): designed, then cut to single-round for v1. Revive only with evidence that single-round + diff review misses things; the round-2 noise question (genuine misses vs sampling churn) is unmeasured.

## Bring-up checklist

- [ ] Smoke test: dispatch resolves to this agent, not a fallback; directive block present in the report.
- [ ] Verify the reviewer receives project CLAUDE.md at startup; check whether the global `~/.claude/CLAUDE.md` also loads (unverified either way).
- [ ] Verify the write bright-line: report lands at the given path, nothing else touched.
- [ ] First real trial: one doc (a recent session note) and one comment-heavy source file; compare findings against a normal finishing pass on the same material.

## Planned experiments

- **Model A/B** (after the design is validated at full strength): 2×2 — two Sonnet-pinned and two inherited instances on the same artifacts, one doc and one comment-heavy code file, the code file being the expected discriminator. Judge precision — the fraction of findings Daniel would act on — not finding count. Within-model overlap sets the sampling-variance bar that between-model differences must clear. Optional: blind adjudication — merged anonymized finding list, scored before unblinding.
- **Loop noise** (only if multi-round is revived): track what fraction of round-2 novel findings are genuine misses vs churn.

## External references

- obra/superpowers `requesting-code-review` and `subagent-driven-development` skills — the closest existing pattern: fresh reviewer per round, file-based handoffs, template-in-skill route.
- Anthropic `code-review` and `feature-dev` plugins (anthropics/claude-code repo, plugins dir) — staged detect-then-validate, human decision gates.
- claude-code issues #47191 (dispatch failures from skills that don't pin the subagent name) and #32910 (`skills:` field controls startup injection, not access restriction).
- Unverified claims encountered: ~20k-token fixed spawn overhead per subagent (community figure); user-vs-project shadowing precedence (the docs gave contradictory readings for skills vs subagents) — test before relying on either.
