"""
Wire-action ↔ model-target conversions for the policy head.

`encode` maps a generals.io wire move `(source, dest, is50)` into the
policy head's flat target representation; `decode` is the inverse.
The dataloader calls `encode` to build training targets; inference
calls `decode` to convert model predictions back to wire moves.

The `(is_pass, flat_idx)` output shape is coupled to the **split-head**
policy design (a separate scalar pass head + per-cell directional
logits). If the policy head ever folds pass into the per-cell logits
(a joint H×W×9 design), this representation has to change.

Wire deltas are interpreted in **unpadded** coordinates (`±1, ±W_unpadded`);
the resulting cell is then re-indexed into the top-left-padded grid
before being flattened. Row-wrap moves (source at column `W_unpadded - 1`
with delta `+1`) are not validated here — the wire protocol never emits
them; the per-cell legality mask catches them downstream if they ever appear.
"""

# Direction enum — NESW clockwise. Must match the policy head's Conv→8
# channel order and the augmentation permutation tables.
N, E, S, W = 0, 1, 2, 3

# Layout pins (sub-channel + flat). Applied by encode/decode below.
#   sub  = dir * 2 + split             # direction-major
#   flat = cell_padded * 8 + sub       # cell-major

# TODO(constants): the policy channel count (8 = 4 dir × 2 split) is a magic
# literal here and in policy.py / mask.py / loss.py. Promote NUM_DIRECTIONS /
# NUM_SPLITS / POLICY_CHANNELS to named constants and have all sites import
# them, so the action-space shape has one source of truth. (Parallel: ValueHead's
# n_classes default of 8 in heads/value.py is really ELIGIBLE_PLAYER_COUNT.)

_PASS_FLAT_IDX = -1  # cross_entropy ignore_index sentinel


def _to_padded(source: int, W_unpadded: int, W_padded: int) -> int:
    """Unpadded source cell → top-left-padded grid cell."""
    row, col = divmod(source, W_unpadded)
    return row * W_padded + col


def _from_padded(cell_padded: int, W_unpadded: int, W_padded: int) -> int:
    """Inverse of `_to_padded` — top-left-padded grid cell → unpadded source."""
    row, col = divmod(cell_padded, W_padded)
    return row * W_unpadded + col


def encode(
    source: int,
    dest: int,
    is50: int,
    W_unpadded: int,
    W_padded: int,
) -> tuple[bool, int]:
    """
    Wire move → `(is_pass, flat_idx)`.

    Pass returns `(True, -1)`. Non-pass returns `(False, flat_idx)` with
    `flat_idx ∈ [0, H_padded * W_padded * 8)` under cell-major layout.
    """
    if source == -1:
        return True, _PASS_FLAT_IDX
    delta = dest - source
    if delta == -W_unpadded:
        direction = N
    elif delta == W_unpadded:
        direction = S
    elif delta == 1:
        direction = E
    elif delta == -1:
        direction = W
    else:
        raise ValueError(
            f"invalid move: source={source}, dest={dest}, W_unpadded={W_unpadded}; "
            f"delta={delta} is not ±1 or ±W_unpadded"
        )
    cell_padded = _to_padded(source, W_unpadded, W_padded)
    sub = direction * 2 + is50
    return False, cell_padded * 8 + sub


def decode(
    is_pass: bool,
    flat_idx: int,
    W_unpadded: int,
    W_padded: int,
) -> tuple[int, int, int]:
    """
    `(is_pass, flat_idx)` → wire move `(source, dest, is50)`.

    Pass returns `(-1, -1, -1)`.
    """
    if is_pass:
        return -1, -1, -1
    cell_padded, sub = divmod(flat_idx, 8)
    direction, is50 = divmod(sub, 2)
    source = _from_padded(cell_padded, W_unpadded, W_padded)
    if direction == N:
        dest = source - W_unpadded
    elif direction == E:
        dest = source + 1
    elif direction == S:
        dest = source + W_unpadded
    else:  # W
        dest = source - 1
    return source, dest, is50
