---
name: draft-doc
description: Four-step process for drafting substantial prose documents — session notes, design docs, summaries, handoffs, README sections. Not for commit messages, code comments, or chat replies. When a task calls for a document like this, propose using this skill rather than running it silently.
---

# draft-doc — gather, judge, then write

Drafting a document straight from rich source material fails in two ways. The prose inherits the sources' density. And the significance judgment gets skipped — with everything in view, every fact looks relevant, so completeness becomes the implicit goal. This process separates the work: facts get selected on a list, where each one can be judged on its own, rather than in finished prose, where every sentence reads as necessary.

Each step has one dominant focus. The steps are not gates — judgment is expected at every one.

## Review checkpoints

Two optional pause points for user review: after the frame (step 1) and after the outline (step 3). The default is to share the artifact and keep going. When the document is complex or high-stakes, or the session is actively collaborative, ask at kickoff which checkpoints the user wants. The user can request a pause at any time.

## Step 1 — Frame

Write a few sentences at the top of the facts file: what the document is, who reads it, what the reader should know or be able to do afterward, and what it deliberately excludes. This step is not skippable — the frame is the yardstick step 3 filters against. If the document's purpose isn't evident from the request, ask; that's requirement-gathering, not an optional pause.

## Step 2 — Gather

Collect facts into `tmp/<slug>-facts.md`, named after the target document. Keep the sources open, record one fact per bullet, and check each claim against its source as you write it down. Discovery order is fine — don't impose structure yet. This is the one step where completeness is welcome.

## Step 3 — Organize & filter

Build `tmp/<slug>-outline.md`: the document's intended section structure, with content selected and ranked against the frame. Significance is the primary axis; group by topic, dependency, or chronology where that helps. If the material is too messy to structure directly, do a loose clustering pass in the same file first. Facts that don't make the cut go in a pile at the bottom of the outline, each with a brief reason.

## Step 4 — Draft

Write the document from the outline with the sources closed. Re-read the outline immediately before drafting so it is the freshest thing in context, and don't re-open source material while writing. Rephrasing outline content is fine — the outline is not a verbatim contract. A fact that surfaces mid-draft goes into the facts file and the outline before it goes into the document.

## After the draft

Run the usual finishing passes, and verify the draft against the sources. Accuracy settles here, not at step 2 — drafting synthesizes, and the connections, summaries, and emphasis it adds are claims no per-fact check covered.

Leave both `tmp/` files in place afterward.
