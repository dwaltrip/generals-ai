"""Core BC training targets: the value-head class label and the policy/pass move.

Thin by design — these are the two always-present supervision targets pulled out
of `encode_frame`'s tail so all target computation lives under `targets/` and the
encoder's tail is pure tensor-packing.
"""

from __future__ import annotations

import numpy as np

from training.bc import actions


def value_target(placement: int) -> int:
    """Value-head class label. Placement is 1..P (1st through Pth).
    Shift to 0-indexed class label for cross_entropy."""
    return placement - 1


def policy_pass_target(
    sim: dict[str, np.ndarray],
    perspective_slot: int,
    t: int,
    W: int,
    W_PADDED: int,
) -> tuple[bool, int]:
    """Policy/pass move target `(is_pass, flat_idx)` for `(perspective_slot, t)`.

    Reads the perspective's recorded action and encodes it via `actions.encode`.
    """
    src = int(sim["actions_source"][perspective_slot, t])
    dst = int(sim["actions_dest"][perspective_slot, t])
    is50 = int(sim["actions_is50"][perspective_slot, t])
    return actions.encode(src, dst, is50, W, W_PADDED)
