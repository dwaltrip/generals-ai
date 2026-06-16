"""Curated families (bundled emit + columns) and their derived columns.

Families are hand-picked bundles, not a global registry of every column. The
derived columns here are plain numpy over a built table — including the
lowest-army *rule* as a computed column, so any head-vs-rule comparison is
matched to the eval frames by construction (the 6.14-1 hardcoded-baseline fix).
"""

from __future__ import annotations

from training.analysis.fq.frame_table import FrameSpec

# name -> FrameSpec, for the CLI's `--table` selection (a small registry of
# families, not of columns).
REGISTRY: dict[str, FrameSpec] = {
    # ----------------------------------------------------------------
    # TODO: this doesn't work... we need a dynamic registration system
    # ----------------------------------------------------------------
    # ELIM_HEAD_DEBUG.name: ELIM_HEAD_DEBUG
}
