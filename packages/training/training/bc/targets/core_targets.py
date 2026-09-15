"""Core BC training targets: policy/pass move and value-head class label."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from training.bc import actions


def value_target(placement: int) -> int:
    """Value-head class label. Placement is 1..P (1st through Pth)"""
    # shift to 0-indexed class label for cross_entropy
    return placement - 1


# TODO: This function has no actual logic, it just extracts the values from `sim`
# and then delegates to `actions.encode`. That file needs a look.
# It wasn't touched in any of the July / Aug refactors.
def policy_pass_target(
    sim: Mapping[str, np.ndarray],
    perspective_slot: int,
    t: int,
    W: int,
    W_PADDED: int,
) -> tuple[bool, int]:
    """Policy/pass move target (is_pass, flat_idx) for (perspective_slot, t)"""
    src = int(sim["actions_source"][perspective_slot, t])
    dst = int(sim["actions_dest"][perspective_slot, t])
    is50 = int(sim["actions_is50"][perspective_slot, t])
    return actions.encode(src, dst, is50, W, W_PADDED)
