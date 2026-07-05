---
name: draft-doc
description: Four-step process for drafting substantial prose documents — session notes, design docs, summaries, handoffs, README sections. Not for commit messages, code comments, or chat replies. When a task calls for a document like this, propose using this skill rather than running it silently.
---

# draft-doc: High-quality write-ups with an intentional "ground-up" process

Drafting a document straight from rich source material fails in two ways. The prose inherits the sources' density. And the significance judgment gets skipped: with everything in view, every fact looks relevant, so completeness becomes the implicit goal. This process separates the work: facts get selected on a list, where each one can be judged on its own, rather than in finished prose, where every sentence reads as necessary.

The process runs in four steps: frame, gather, organize & filter, draft. Each step has one dominant focus. The steps are not gates: judgment is required throughout.

The frame carries more weight per word than any other step. It settles what the document should convey, for what purpose, and which angles beyond the headline are worth covering. Every later choice answers to those goals: filtering selects against them, and the draft is judged by them. It is also where alignment with the user matters most. A subtly wrong assumption, however reasonable, does more damage in the frame than anywhere later.

## Artifacts

Two scratch files per document, in `tmp/draft-doc/<yyyy-mm>/` (month folder mirroring `docs/`), with the slug named after the target document:

- `<slug>-facts.md`: the frame (step 1), then the gathered facts (step 2).
- `<slug>-outline.md`: the filtered outline (step 3).

Both stay in place after the draft ships. They can be useful for later reference, and the user can clean up as they desire.

## Review checkpoints

Two pause points for user review: after the frame (step 1) and after the outline (step 3). The default is to pause at both. The review is cheap at these points because the judgment is still open, and a problem found after drafting costs a rewrite. When proposing the skill, say which checkpoints you plan to pause at. The user can request a pause at any time.

Skip the pauses (share the artifact and keep going) when either holds:

- The session is executing a pre-agreed multi-step plan and the document is one of its tasks. The finishing passes and any planned review step carry the quality burden instead.
- The document is short and straightforward, the frame is already clear from the discussion, and alignment is strong.

If trimming to one checkpoint, keep the frame: no other review buys as much correction for as little reading.

## Step 1 — Frame

Write a few sentences at the top of the facts file: what the document is, who reads it, what the reader should know or be able to do afterward, and what it deliberately excludes. Write the frame before gathering anything, even when the purpose feels obvious. If the document's purpose isn't evident from the request, ask. That's requirement-gathering, not an optional pause.

## Step 2 — Gather

Collect facts into the facts file. Keep the sources open, record one fact per bullet, and check each claim against its source as you write it down. Record each fact in your own plain words. A bullet that copies the source's phrasing carries the source's register through the outline into the draft. Discovery order is fine. Don't impose structure yet. This is the one step where completeness is welcome.

## Step 3 — Organize & filter

Build the outline: the document's intended section structure, with content selected and ranked against the frame. Significance is the primary axis. Group by topic, dependency, or chronology where that helps. If the material is too messy to structure directly, do a loose clustering pass in the same file first. Facts that don't make the cut go in a pile at the bottom of the outline, each with a brief reason, so the outline review can rescue a wrong cut.

## Step 4 — Draft

Write the document from the outline with the sources closed. Re-read the outline immediately before drafting so it is the freshest thing in context, and don't re-open source material while writing. Rephrasing outline content is fine: the outline is not a verbatim contract. A fact that surfaces mid-draft goes into the facts file and the outline before it goes into the document. Its check against the sources waits for the after-draft verification.

## After the draft

Run the usual finishing passes, and verify the draft against the sources. Accuracy settles here, not at step 2: drafting synthesizes, and the connections, summaries, and emphasis it adds are claims no per-fact check covered.
