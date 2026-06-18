"""Load-bearing encoder test (6.18-1 §10, Risk #2).

The toy's headline is a comparison *across* encodings, so a silent encoding bug
doesn't crash — it produces a clean-looking wrong result. These assert each cfg
against a hand-built `(army, alive)` frame and prove the ladder behaves as §3 says:
explicit_mask is linear-solvable; zero_overload / neg1_sentinel require the notch.
"""

from __future__ import annotations

import torch

from training.analysis.argmin_toy.encode import NORM, EncoderCfg, encode


# One mixed frame: alive players p0,p1,p3; p1 is the true argmin (smallest live
# army). Dead players sit at army 0 (the post-filter binary population).
ARMY = torch.tensor([[500.0, 3.0, 0.0, 200.0, 0.0, 0.0, 0.0, 0.0]])
LAND = torch.tensor([[50.0, 1.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0]])
ALIVE = torch.tensor([[True, True, False, True, False, False, False, False]])
ARMY_NORM = ARMY / NORM  # [0.5, 0.003, 0, 0.2, 0, 0, 0, 0]
TRUE_LABEL = 1

BIG = 1.0  # the explicit_mask linear coefficient; must exceed max normalized army


def test_zero_overload_army_only() -> None:
    cfg = EncoderCfg(encoding="zero_overload", channels="army_only")
    x = encode(ARMY, LAND, ALIVE, cfg)
    assert x.shape == (1, 8, 1) == (1, 8, cfg.n_channels)
    torch.testing.assert_close(x[..., 0], ARMY_NORM)  # dead untouched, already 0


def test_zero_overload_gap_scale() -> None:
    cfg = EncoderCfg(encoding="zero_overload", channels="army_only", gap_scale=10.0)
    x = encode(ARMY, LAND, ALIVE, cfg)
    torch.testing.assert_close(x[..., 0], ARMY_NORM * 10.0)


def test_neg1_sentinel_overwrites_dead() -> None:
    cfg = EncoderCfg(encoding="neg1_sentinel", channels="army_only")
    x = encode(ARMY, LAND, ALIVE, cfg)
    assert x.shape == (1, 8, 1) == (1, 8, cfg.n_channels)
    expected = torch.where(ALIVE, ARMY_NORM, torch.full_like(ARMY_NORM, -1.0))
    torch.testing.assert_close(x[..., 0], expected)
    # Dead is linearly separable from all live (≥0): dead at −1, live at ≥0.
    assert (x[..., 0][~ALIVE] == -1.0).all()
    assert (x[..., 0][ALIVE] >= 0.0).all()


def test_explicit_mask_channels_and_linear_solvability() -> None:
    cfg = EncoderCfg(encoding="explicit_mask", channels="army_only")
    x = encode(ARMY, LAND, ALIVE, cfg)
    assert x.shape == (1, 8, 2) == (1, 8, cfg.n_channels)
    torch.testing.assert_close(x[..., 0], ARMY_NORM)            # army channel, dead at 0
    torch.testing.assert_close(x[..., 1], ALIVE.float())        # alive channel

    # The defining property: argmin is LINEAR in the augmented input. score =
    # −army_norm + BIG·alive picks the lowest-army alive player iff BIG exceeds
    # the max normalized army (so a live player always outscores any dead-at-0).
    assert BIG > float(ARMY_NORM.max())
    score = -x[..., 0] + BIG * x[..., 1]
    assert int(score.argmax(dim=1)) == TRUE_LABEL


def test_zero_overload_needs_the_notch() -> None:
    # Plain −army (the clean linear solution) FAILS under zero_overload: dead-at-0
    # is the largest value of −army, so argmax lands on a dead slot. Excluding it
    # needs a non-monotonic notch (Claim 1 / Appendix A worked example).
    cfg = EncoderCfg(encoding="zero_overload", channels="army_only")
    x = encode(ARMY, LAND, ALIVE, cfg)
    picked = int((-x[..., 0]).argmax(dim=1))
    assert not bool(ALIVE[0, picked]), "−army must pick a dead slot under zero_overload"


def test_army_land_channels() -> None:
    z = EncoderCfg(encoding="zero_overload", channels="army_land")
    x = encode(ARMY, LAND, ALIVE, z)
    assert x.shape == (1, 8, 2) == (1, 8, z.n_channels)
    torch.testing.assert_close(x[..., 0], ARMY_NORM)
    torch.testing.assert_close(x[..., 1], LAND / NORM)

    e = EncoderCfg(encoding="explicit_mask", channels="army_land")
    xe = encode(ARMY, LAND, ALIVE, e)
    assert xe.shape == (1, 8, 3) == (1, 8, e.n_channels)
    torch.testing.assert_close(xe[..., 1], LAND / NORM)
    torch.testing.assert_close(xe[..., 2], ALIVE.float())
