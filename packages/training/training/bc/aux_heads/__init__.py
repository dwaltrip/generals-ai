# TODO: The concepts behind this module seem to fit into 3 slices.
#   - Slice 1: The generic concept of an auxiliary head (base.py is nearly this)
#   - Slice 2: The spec / contract / metadata / etc for a specific aux head
#   - Slice 3: The actual implementation details for an aux head (2 + 3 are linked)
# Right now, these slices are not cleanly represented in the code. We need to
# clean this up at some point.
# For future reference: slice 3 is spread across this module, as well as
# `bc/model/heads/` and targets code (`bc/targets/`, `bc/soft_target.py`).
# Original design notes: docs/2026-06/6.21-1-aux-head-spec-registry-refactor.md

# NOTE: Some of `bc.aux_heads` is imported by the numpy-only golden tests,
# so __init__.py can't re-export from parts that do use torch.
# Consumers of that code must import from the full path.

from __future__ import annotations

from training.bc.aux_heads.elim_head_meta import ElimHeadVariant


__all__ = ["ElimHeadVariant"]
