"""Round-trip + invalid-input tests for `bc.actions.encode/decode`."""

import pytest

from bc.actions import decode, encode


# Round-trip cases. Format: (source, dest, is50, W_unpadded, W_padded).
# Coverage: every (direction × split) combo across multiple boards, including
# the W_unpadded == W_padded identity-reindex path.
#
# Cell layout reference for an 18-wide board:
#   cell 0   = (row 0, col 0)
#   cell 17  = (row 0, col 17)   # rightmost col, top row
#   cell 18  = (row 1, col 0)
#   cell 35  = (row 1, col 17)
#   cell 323 = (row 17, col 17)  # bottom-right of an 18×18 board
ROUND_TRIP_CASES = [
    # 18-wide → 32-padded
    pytest.param(0,   1,   0, 18, 32, id="w18-topleft-E-full"),
    pytest.param(0,   18,  0, 18, 32, id="w18-topleft-S-full"),
    pytest.param(17,  16,  0, 18, 32, id="w18-toprow-rightedge-W-full"),
    pytest.param(35,  34,  1, 18, 32, id="w18-mid-W-half"),
    pytest.param(35,  53,  0, 18, 32, id="w18-mid-S-full"),
    pytest.param(35,  17,  1, 18, 32, id="w18-mid-N-half"),
    pytest.param(323, 305, 0, 18, 32, id="w18-bottomright-N-full"),
    pytest.param(35,  36,  1, 18, 32, id="w18-mid-E-half"),
    # 24-wide → 32-padded
    pytest.param(25,  26,  1, 24, 32, id="w24-r1c1-E-half"),
    pytest.param(25,  49,  0, 24, 32, id="w24-r1c1-S-full"),
    # 32-wide → 32-padded (identity re-index path)
    pytest.param(0,   1,   0, 32, 32, id="w32-identity-E-full"),
    pytest.param(33,  1,   0, 32, 32, id="w32-identity-N-full"),
    pytest.param(100, 132, 1, 32, 32, id="w32-identity-S-half"),
    # Pass — is50 should not affect round-trip (decode returns -1 regardless)
    pytest.param(-1, -1, 0, 18, 32, id="pass-is50-0"),
    pytest.param(-1, -1, 1, 18, 32, id="pass-is50-1"),
]


@pytest.mark.parametrize("source,dest,is50,W_unpadded,W_padded", ROUND_TRIP_CASES)
def test_round_trip(source, dest, is50, W_unpadded, W_padded):
    is_pass, flat_idx = encode(source, dest, is50, W_unpadded, W_padded)
    s2, d2, i2 = decode(is_pass, flat_idx, W_unpadded, W_padded)
    if source == -1:
        assert (s2, d2, i2) == (-1, -1, -1)
        assert is_pass is True
        assert flat_idx == -1
    else:
        assert (s2, d2, i2) == (source, dest, is50)
        assert is_pass is False
        assert flat_idx >= 0


# Invalid-input cases — encode should raise ValueError.
INVALID_CASES = [
    pytest.param(0,  19, 0, 18, 32, id="diagonal-delta-19"),    # SE diag
    pytest.param(35, 18, 0, 18, 32, id="diagonal-delta-neg17"), # NW-ish diag
    pytest.param(35, 35, 0, 18, 32, id="source-equals-dest"),   # delta == 0
]


@pytest.mark.parametrize("source,dest,is50,W_unpadded,W_padded", INVALID_CASES)
def test_invalid_input_raises(source, dest, is50, W_unpadded, W_padded):
    with pytest.raises(ValueError):
        encode(source, dest, is50, W_unpadded, W_padded)
