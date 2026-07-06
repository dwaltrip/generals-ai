# content-review — maintainer notes

Covers the content-review system: the content-reviewer agent (`.claude/agents/content-reviewer.md`) and the content-review skill (`.claude/skills/content-review/`). The two are one system: the agent is the stable contract, the skill is the procedure we expect to iterate on.

Status (2026-07-06): agent + skill v1 built, smoke-tested, and through the doc half of the first real trial (see Trial notes). Remaining bring-up: the code-mode trial.

## Origin

Adapted from an external draft agent definition (scratch copy at `zzz-scratch/content-review-agent.md` — untracked and ephemeral; the cut list below is the durable record). Kept: the cold-read blinding mechanism, legibility lens, anti-pattern list, severity tiers, verdict-as-posture, two-voices rule. Cut: the source system's "dispatch covenant"/"governance layer" framing, the corporate exposure-lens machinery, stop-and-wait gating, and most of the persona theatrics.

## Load-bearing design decisions

**Custom agent over prompt-template-in-skill.** The template route (the obra/superpowers pattern — see External references) keeps the pair in one directory and has no name-resolution risk, but all its constraints are behavioral. The custom agent gives enforced capability limits via `tools:` and puts the contract in the system prompt, where adherence is strongest. Name-resolution mitigations: the skill pins the literal `subagent_type` string, and bring-up includes a resolution smoke test. The closing directive block doubles as an authenticity marker — a report missing it is treated as a failed dispatch (a fallback to some other agent, or a reviewer that didn't follow its contract) and its findings must not be used.

**Blinding is conversational, not informational.** The reviewer is denied the conversation that produced the artifact, not knowledge of the repo. Context channels beyond the dispatch prompt:

- **CLAUDE.md auto-load** — accepted, deliberate. Used as the policy-delivery channel (next item).
- **Read reaches any repo file** — accepted, deliberate. Scoping to the handed paths is behavioral only — the harness has no declarative per-agent path scoping (frontmatter `tools:` takes bare names, permission rules are session-wide). Hardening, if ever needed, is the parked PreToolUse-hook idea (see Parked alternatives).
- **Git-status snapshot in the subagent environment** — suspected, unverified (see Bring-up checklist). The reviewer of this notes file stated that `zzz-scratch/` is untracked, a fact Read alone can't reveal (the alternative explanation: a confident guess from the directory name). If real, this channel differs from the accepted two: git status carries session-circumstantial detail (branch name, freshly touched files), closer to what the blinding is meant to deny.

**House policy arrives via CLAUDE.md auto-load.** The agent file carries one generic line (prose/comment policy in project context is governance context to apply); AGENTS.md stays the single authoritative home of the rewrite-trim-drop test. Portability follows: in another repo, that repo's CLAUDE.md supplies its own policy. This depends on the documented auto-load behavior — verification status is in the Bring-up checklist (global confirmed; project not yet directly evidenced).

**One framework plus code deltas.** Doc and code review share the legibility lens and anti-patterns. The "when the artifact is source code" section lists only the genuine differences (the deltas live in the agent file, not here). Chosen over parallel doc/code modes, which would duplicate most of the file.

**Consistency checks are licensed, not a second mission (decided 2026-07-06, unvalidated in use).** The lens stays legibility. The handed set is ground truth for what the reviewed prose says about it, and the reviewer reports the mismatches its read surfaces — a license, not a search task. Defects clearly on a context file's side become **observations** (own O-series IDs, no severity tier, covered by the closing directive). Ambiguous contradictions stay conditional findings on the reviewed text. Context-set inclusion is still governed by the reader test alone — wanting a claim checked doesn't qualify a file. Rationale: the operative text already instructed reference-coverage checks and code-as-ground-truth, so the legibility-only self-description contradicted it, and the resulting drift catches were the tool's best output so far.

**Single lens; conditionality is a property, not a tier.** Exposure demoted from a second lens to one anti-pattern entry. Any finding at any severity may be conditional and must state its concrete resolving condition. Findings get stable IDs (F1, F2, …) for reference across digest and discussion.

**Directive states invariants only; the dial lives in the skill.** Agent-side invariants: the complete report — every finding and observation, annotated with any action the caller took — reaches the human. No finding or observation is silently discarded. Drops of content, trims of substance, and conditional resolutions are the human's call. How much the caller fixes directly (the "dial") is caller procedure with one authoritative home, the skill.

**Write access with a bright line.** `tools: Read, Write` — the reviewer writes its own report file. The alternative (caller transcribes the returned report) passes the full text through the caller's context twice and risks copy corruption, a weaker provenance story than reviewer-written files. The constraint is behavioral: exactly one file, at the caller-given path, never a reviewed file. A diff in a reviewed file after a review round is a loud violation signal.

**`model: inherit` for bring-up.** Validate the design at full model strength first, so reviewer weakness can't be mistaken for design weakness. Sonnet A/B afterwards — see Planned experiments.

## Caller-side design (built; see SKILL.md for the operative text)

- **Fix policy — fix-then-highlight (settled 2026-07-06):** the caller applies the findings and the digest routes attention rather than seeking permission: judgment-heavy or multi-faceted edits itemized, mechanical fixes aggregated, findings not acted on listed with reasons. Conditionals split by nature: the caller resolves and notes the ones checkable by verification, and leaves genuine judgment calls undone and flagged. Rationale: the historical failures were in *finding* issues — thoroughness, blindness to style contagion — and finding is now the agent's job. Fixing clearly-pointed-out issues was never the weak step. An earlier gate-then-fix design (caller queues fixes for approval) was dropped as solving the wrong problem.
- **Rounds — single round (settled 2026-07-06):** one cold read per invocation. The human's diff review is the fix-verification step. Re-review is just another explicit invocation, proposed by the caller when the round-1 verdict recommends a fresh look, and always a fresh instance — a re-consulted one is no longer cold and grades its own suggestions.
- **Reports:** reviewer-written, into per-engagement directories under `tmp/content-review/` (path scheme lives in SKILL.md). The caller never edits a report; caller annotations live in the digest. The tmp files decay — durable lessons land here or in other docs, not in report files.
- **Review unit:** single file by default, dispatched in parallel per file. A multi-file bundle is the unit when cross-file duplication or comment-vs-doc redundancy is the question. A unit may also carry a **context set** — files the reviewer reads and the prose may lean on, but does not review (added 2026-07-06). One principle governs both: include what a realistic future reader would have in view, not what explains the prose — over-bundling quietly un-blinds the read. Instances are mutually blind, so cross-file pattern synthesis is the caller's job.
- **Invocation (settled 2026-07-06):** explicit paths only. A single path dispatches immediately. Multiple paths get a one-line grouping plan confirmed before dispatch (bundling is contextual — sibling code files usually bundle, a doc gets its own reviewer). Grouping and artifact-description overrides travel in prose, no flags. Never auto-run — the caller proposes. Bare no-args invocation (scope from uncommitted changes) deferred: it stacks file-selection judgment on top of grouping judgment; revisit once grouping intuitions are established.
- **Further detail** (report/digest formatting, failure handling — authenticity-marker check, loud stop, no silent fallback) lives in SKILL.md.

## Parked alternatives and iteration ideas

- **Detect-then-validate** (Anthropic code-review plugin pattern): a second wave of validator agents re-checks each finding, and unvalidated ones are dropped. The alternative quality mechanism if reviewer findings prove too noisy to act on directly.
- **Autonomous mode:** deferred until finding-quality data exists. Would relax the dial per-run; the agent-side invariants are written to hold unchanged.
- **Glossary injection:** skipped in v1 (CLAUDE.md covers some project vocabulary, and the artifact-description line carries audience info). The agent file's dangling glossary allowance was cut 2026-07-06 — reviving means adding a template slot and re-adding the allowance. Revisit if term-triage flags get noisy.
- **Shared-vocabulary skill** preloaded into both sides (agents support a `skills:` frontmatter field) if agent/skill drift actually bites. Known pair-sync surface already live: SKILL.md's authenticity check quotes the directive block's opening words — if the agent file's directive is ever reworded, update the quote in lockstep (flagged by the first Sonnet review).
- **Plugin-ization:** Claude Code's documented route for versioning the pair as one shippable unit.
- **Verification-only context files:** deliberately including a file readers won't have, so the reviewer can check the prose's claims about it. In principle high-value — claims readers can't verify are where drift lives longest — but no real case has arisen, and it would split the context set's single reader-test semantic. Revisit when a real case arrives.
- **PreToolUse hook** to structurally enforce the report-write path and Read scoping. Per official docs (2026-07-06, not tested here): agent frontmatter supports its own `hooks:`, active only while that subagent runs, and hook input carries the tool input, so a hook in the agent file could validate paths. The docs also say hook input identifies the calling agent (`agent_type`/`agent_id`), making a session-level hook an alternative route. Caveat: plugin subagents ignore `hooks:`, so plugin-ization would forfeit the frontmatter option.
- **Multi-round fix-and-rerun loop** (up to N rounds, early exit on a quiet round): designed, then cut to single-round for v1. Revive only with evidence that single-round + diff review misses things; the round-2 noise question (genuine misses vs sampling churn) is unmeasured.

## Bring-up checklist

- [x] Smoke test: dispatch resolves to this agent, not a fallback; directive block present in the report. (2026-07-06, on SKILL.md itself)
- [x] Global `~/.claude/CLAUDE.md` confirmed loaded — the reviewer cited its semicolon rule unprompted. Project CLAUDE.md receipt not yet directly evidenced.
- [x] Write bright-line: report at the given path, nothing else touched.
- [x] First real trial, doc half (2026-07-06): 7.06-2 motivations doc, two Sonnet reviewers — single-file and context-set variants. Notes below.
- [ ] First real trial, code half: a comment-heavy source file; compare findings against a normal finishing pass. Code mode is entirely unexercised — the rewrite-trim-drop half of the design has never run.
- [ ] Check whether the subagent environment context includes a git-status snapshot (suspected — see the blinding item). If it does, decide whether it's a third accepted leak or wants hardening.

## Trial notes (2026-07-06, doc mode)

- **Context set proved itself on its first run.** The single-file reviewer flagged the doc's axis terminology conditionally ("plausibly defined in companions"); the context-set reviewer, able to read the companion, established the definition isn't there either — upgrading a hedged conditional into the doc's most actionable finding. It also verified another reference as fine, and produced zero findings against the context file (contract respected).
- **Term triage with a free-text audience sentence (inside the artifact-description line) worked acceptably.** "Intended readers are maintainers with working knowledge of the repo's vocabulary" muted repo-basic and long-standing project terms; effort-local coinage was flagged conditionally, which is the desired discrimination. Muting isn't deterministic — one run flagged `elim-head`, the other didn't — but the stray flags are hedged and cost seconds. Glossary still not warranted on this evidence.
- **Review axis (legibility vs. actual content) — line drawn 2026-07-06, not yet tested in use.** The bleed past the legibility line (reference-coverage checks, drift catches against the operative files) was sanctioned as a licensed check riding along with the cold read — see the load-bearing decision above. The license-not-mission shape and the observations channel are bets that haven't faced real usage; upcoming reviews show whether noise or diluted attention materializes.
- **Open question — the artifact-description line is the one free-text dispatch slot.** It's quasi-content: wording varies by session, so term-muting behavior varies with it, and it's the natural channel for framing to creep toward content injection. Canned per-class lines in the template (session note / evergreen doc / source file) would remove the variance; the counterargument is that a small amount of caller framing may be genuinely useful. Left free-text for now, deliberately.
- **Union-beats-either, in both paired comparisons so far** (Fable-vs-Sonnet on SKILL.md, and single-file-vs-context-set on the trial doc): each had genuine findings unique to each reviewer. Parked idea: an N-reviewers-per-unit mode (two parallel cold readers, merged digest) for high-stakes docs — the parallel answer to thoroughness, distinct from the cut serial loop.
- **Reviewer overclaim observed once:** a grammatical-but-heavy sentence reported as "doesn't parse." Caller-side mitigation added to the skill: verify the quoted claim before acting.
- **Report-only usage needs no mechanism.** The user asked to hold fixes; plain prose in the conversation handled it. Consistent with the no-flags stance: the executor is a model, invocation prose overrides defaults.
- **Cost per reviewer (Sonnet):** ~35–40k subagent tokens, ~2.5 minutes.

## Planned experiments

- **Model A/B** (after the design is validated at full strength): 2×2 — two Sonnet-pinned and two inherited instances on the same artifacts, one doc and one comment-heavy code file, the code file being the expected discriminator. Judge precision — the fraction of findings Daniel would act on — not finding count. Within-model overlap sets the sampling-variance bar that between-model differences must clear. Optional: blind adjudication — merged anonymized finding list, scored before unblinding. Reasoning effort is a second lever: agent frontmatter supports `effort:` (low–max, default inherits session). Hold it at inherit for the base comparison, and if Sonnet trails, try a Sonnet-at-higher-effort arm before conceding the cost difference. Preliminary uncontrolled sample (2026-07-06, one run each on differing SKILL.md versions): Sonnet found two genuine issues Fable had missed, with no shallow-pedantry signal — viable so far.
- **Loop noise** (only if multi-round is revived): track what fraction of round-2 novel findings are genuine misses vs churn.

## External references

- obra/superpowers `requesting-code-review` and `subagent-driven-development` skills — the closest existing pattern: fresh reviewer per round, file-based handoffs, template-in-skill route.
- Anthropic `code-review` and `feature-dev` plugins (anthropics/claude-code repo, plugins dir) — staged detect-then-validate, human decision gates.
- claude-code issues #47191 (dispatch failures from skills that don't pin the subagent name) and #32910 (`skills:` field controls startup injection, not access restriction).
- Unverified claims encountered: ~20k-token fixed spawn overhead per subagent (community figure); user-vs-project shadowing precedence (the docs gave contradictory readings for skills vs subagents) — test before relying on either.
