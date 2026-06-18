"""Load-bearing encoder test (6.18-1 §10 / 6.18-4 §8, Risk #2).

The toy's headline is a comparison *across* encodings, so a silent encoding bug
doesn't crash — it produces a clean-looking wrong result. These assert each cfg
against a hand-built frame that carries all three player statuses, and prove the
status-sufficiency ladder behaves as 6.18-4 §4 says: `boolean`/`categorical` are
linear-solvable; `none`/`capture_status`/`neg1_sentinel` require the notch.
"""

from __future__ import annotations

import torch

from training.analysis.argmin_toy.encode import NORM, EncoderCfg, encode


# One full-population frame, all three statuses present:
#   p0,p1,p3 alive (p1 the true argmin); p2 eliminated-by-capture (army 0, captured);
#   p4 surrendered-present (~alive, army 7, NOT captured); p5,p6,p7 neutralized
#   (~alive, army 0, not captured).
ARMY = torch.tensor([[500.0, 3.0, 0.0, 200.0, 7.0, 0.0, 0.0, 0.0]])
LAND = torch.tensor([[50.0, 1.0, 0.0, 20.0, 2.0, 0.0, 0.0, 0.0]])
ALIVE = torch.tensor([[True, True, False, True, False, False, False, False]])
CAPTURED = torch.tensor([[False, False, True, False, False, False, False, False]])
ARMY_NORM = ARMY / NORM
TRUE_LABEL = 1  # lowest-army alive player (the `alive` target)

BIG = 1.0  # the status-channel linear coefficient; must exceed max normalized army
SURRENDERED = torch.tensor([[False, False, False, False, True, False, False, False]])
ELIMINATED = torch.tensor([[False, False, True, False, False, True, True, True]])


def _enc(cfg: EncoderCfg) -> torch.Tensor:
    return encode(ARMY, LAND, ALIVE, CAPTURED, cfg)


def test_none_is_raw_army_and_needs_notch() -> None:
    cfg = EncoderCfg(encoding="none", channels="army_only")
    x = _enc(cfg)
    assert x.shape == (1, 8, 1) == (1, 8, cfg.n_channels)
    torch.testing.assert_close(x[..., 0], ARMY_NORM)  # no status channel, raw army
    # −army (the clean linear solution) FAILS: dead-at-0 is the largest −army, so
    # argmax lands on a dead slot. Excluding it needs a non-monotonic notch.
    picked = int((-x[..., 0]).argmax(dim=1))
    assert not bool(ALIVE[0, picked]), "−army must pick a dead slot under none"


def test_capture_status_flags_only_captured_and_needs_notch() -> None:
    cfg = EncoderCfg(encoding="capture_status", channels="army_only")
    x = _enc(cfg)
    assert x.shape == (1, 8, 2) == (1, 8, cfg.n_channels)
    torch.testing.assert_close(x[..., 0], ARMY_NORM)
    torch.testing.assert_close(x[..., 1], CAPTURED.float())  # captured flag only
    # The flag catches p2 but not the neutralized dead (p5–7, army 0, unflagged) nor
    # the surrendered p4 — so it adds nothing over the army-zero notch (none ≡
    # capture_status). A flag-only exclusion still leaves dead slots at army 0:
    unflagged_dead = (~ALIVE[0]) & (CAPTURED[0] == 0)
    assert bool(unflagged_dead.any()), "no unflagged dead — fixture wouldn't test the notch"
    assert (x[0, :, 0][unflagged_dead] == 0.0).any(), "a neutralized dead must sit at army 0"


def test_boolean_linear_solvability() -> None:
    cfg = EncoderCfg(encoding="boolean", channels="army_only")
    x = _enc(cfg)
    assert x.shape == (1, 8, 2) == (1, 8, cfg.n_channels)
    torch.testing.assert_close(x[..., 0], ARMY_NORM)
    torch.testing.assert_close(x[..., 1], ALIVE.float())
    # argmin is LINEAR in the augmented input: −army + BIG·alive picks the lowest-army
    # alive player iff BIG exceeds the max normalized army (a live player always
    # outscores any dead-at-0, and surrendered p4 is alive=0 → excluded).
    assert BIG > float(ARMY_NORM.max())
    score = -x[..., 0] + BIG * x[..., 1]
    assert int(score.argmax(dim=1)) == TRUE_LABEL


def test_categorical_channels_and_linear_solvability() -> None:
    cfg = EncoderCfg(encoding="categorical", channels="army_only")
    x = _enc(cfg)
    assert x.shape == (1, 8, 4) == (1, 8, cfg.n_channels)
    torch.testing.assert_close(x[..., 0], ARMY_NORM)
    torch.testing.assert_close(x[..., 1], ALIVE.float())          # alive
    torch.testing.assert_close(x[..., 2], SURRENDERED.float())    # ~alive & army>0
    torch.testing.assert_close(x[..., 3], ELIMINATED.float())     # ~alive & army==0
    # Linear-solvable via the alive channel alone (the surrendered/eliminated split
    # is the confirmatory extra): −army + BIG·alive picks the true label.
    score = -x[..., 0] + BIG * x[..., 1]
    assert int(score.argmax(dim=1)) == TRUE_LABEL


def test_neg1_sentinel_overwrites_dead_and_needs_notch() -> None:
    cfg = EncoderCfg(encoding="neg1_sentinel", channels="army_only")
    x = _enc(cfg)
    assert x.shape == (1, 8, 1) == (1, 8, cfg.n_channels)
    expected = torch.where(ALIVE, ARMY_NORM, torch.full_like(ARMY_NORM, -1.0))
    torch.testing.assert_close(x[..., 0], expected)
    # Dead (incl. surrendered p4, masked by alive) is linearly separable from live
    # (≥0), but −army still picks a dead slot (−(−1)=+1 is the max), so a notch at −1
    # is needed.
    assert (x[..., 0][~ALIVE] == -1.0).all()
    assert (x[..., 0][ALIVE] >= 0.0).all()
    assert not bool(ALIVE[0, int((-x[..., 0]).argmax(dim=1))])


def test_army_land_channels() -> None:
    n = EncoderCfg(encoding="none", channels="army_land")
    x = _enc(n)
    assert x.shape == (1, 8, 2) == (1, 8, n.n_channels)
    torch.testing.assert_close(x[..., 0], ARMY_NORM)
    torch.testing.assert_close(x[..., 1], LAND / NORM)

    b = EncoderCfg(encoding="boolean", channels="army_land")
    xb = _enc(b)
    assert xb.shape == (1, 8, 3) == (1, 8, b.n_channels)
    torch.testing.assert_close(xb[..., 1], LAND / NORM)            # land before status
    torch.testing.assert_close(xb[..., 2], ALIVE.float())

    c = EncoderCfg(encoding="categorical", channels="army_land")
    assert c.n_channels == 5  # army + land + 3 status
    assert _enc(c).shape == (1, 8, 5)
