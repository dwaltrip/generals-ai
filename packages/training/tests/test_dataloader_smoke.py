"""
End-to-end smoke test for the BC dataloader pipeline.

Wraps `bc.dataset.IterableDataset` in a torch DataLoader, pulls 100 batches
of size 64, and asserts the tensor contract holds on real-corpus data:
shapes, dtypes, no NaN/inf, target ranges, mask covers the recorded action
on non-pass frames, split-independence.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import islice
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from training.bc.constants import H_PADDED, W_PADDED
from training.bc.dataset import IterableDataset
from training.bc.model import BCModel
from training.bc.obs_config import OBS_CHANNELS, OBS_CONFIG_DEFAULTS


BATCH_SIZE = 64
NUM_BATCHES = 100

# Flat-index range for the policy head: [0, H_PADDED * W_PADDED * 8).
FLAT_ACTION_COUNT = H_PADDED * W_PADDED * 8


# Exercise both the raw per-perspective walk (0) and the buffered path (256).
# A small buffer is enough to confirm the wiring; helper correctness is
# unit-tested in test_dataset.py.
@pytest.mark.parametrize("shuffle_buffer_size", [0, 256])
def test_dataloader_pipeline_smoke(
    samples: list[tuple[Path, int]],
    shuffle_buffer_size: int,
) -> None:
    ds = IterableDataset(
        samples=samples,
        seed=0,
        obs_cfg=OBS_CONFIG_DEFAULTS,
        shuffle_buffer_size=shuffle_buffer_size,
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE)

    for batch in islice(loader, NUM_BATCHES):
        obs = batch["obs"]
        mask = batch["mask"]
        valid_mask = batch["valid_mask"]
        action_target = batch["action_target"]
        is_pass = batch["is_pass"]
        value_target = batch["value_target"]

        # --- Shapes ---
        assert obs.shape == (BATCH_SIZE, OBS_CHANNELS, H_PADDED, W_PADDED)
        assert mask.shape == (BATCH_SIZE, H_PADDED, W_PADDED, 8)
        assert valid_mask.shape == (BATCH_SIZE, 1, H_PADDED, W_PADDED)
        assert action_target.shape == (BATCH_SIZE,)
        assert is_pass.shape == (BATCH_SIZE,)
        assert value_target.shape == (BATCH_SIZE,)

        # --- Dtypes ---
        assert obs.dtype == torch.float16  # OBS_CONFIG_DEFAULTS is fp16
        assert mask.dtype == torch.bool
        assert valid_mask.dtype == torch.bool
        assert action_target.dtype == torch.int64
        assert is_pass.dtype == torch.bool
        assert value_target.dtype == torch.int64

        # --- No NaN/inf in obs (catches log-scaling or padding bugs) ---
        assert torch.isfinite(obs).all(), "obs contains NaN or inf"

        # --- Value target in [0, 8) — class index for placement 1..8 ---
        assert (value_target >= 0).all() and (value_target < 8).all()

        # --- Pass frames: action_target == -1, by construction ---
        assert (action_target[is_pass] == -1).all()

        # --- Non-pass frames ---
        nonpass = ~is_pass
        if nonpass.any():
            # Targets land in the cell-major flat index space.
            np_targets = action_target[nonpass]
            assert (np_targets >= 0).all()
            assert (np_targets < FLAT_ACTION_COUNT).all()

            # The recorded action's cell+direction must be flagged legal in the
            # mask — strongest single check that mask construction is correct.
            flat_mask = mask[nonpass].reshape(nonpass.sum().item(), FLAT_ACTION_COUNT)
            target_legality = flat_mask.gather(1, np_targets.unsqueeze(1)).squeeze(1)
            assert target_legality.all(), "recorded non-pass action not legal under mask"

            # Plan §3 step 7: mask must cover ≥1 legal cell on every non-pass frame.
            # Implied by the stronger check above but kept as an explicit assertion.
            assert (flat_mask.any(dim=1)).all(), "non-pass frame with empty legality mask"

        # --- Split independence: mask[..., 0::2] == mask[..., 1::2] ---
        # The two split sub-channels are constructed from the same per-direction
        # legality. A mismatch here would mean the np.repeat went wrong.
        assert torch.equal(mask[..., 0::2], mask[..., 1::2])

    # --- End-to-end: feed one real batch through the model ---
    # Catches any signature/contract drift between dataset and model heads
    # (e.g. valid_mask shape/dtype mismatch). Cheap; runs once after the loop.
    model = BCModel()
    model.eval()
    with torch.no_grad():
        # fp32 model, no autocast here → upcast the (fp16-default) obs, mirroring
        # the no-autocast branch of shared.device.obs_for_model.
        out = model(obs.float(), valid_mask)
    assert out.policy_logits.shape == (BATCH_SIZE, 8, H_PADDED, W_PADDED)
    assert out.pass_logit.shape == (BATCH_SIZE,)
    assert out.value_logits.shape == (BATCH_SIZE, 8)


def test_obs_fp16_matches_fp32_downcast(samples: list[tuple[Path, int]]) -> None:
    """fp16-built obs is bit-identical to the fp32 build cast to fp16 — the
    invariant behind "fp16 obs is free under autocast" (docs/2026-06/6.06-7).

    Under CUDA autocast the model's first conv casts the fp32 obs to fp16 with
    round-to-nearest; building obs as fp16 just applies that same rounding at
    assembly instead. Comparing the fp16-built obs against `o32.half()` proves
    both halves at once: the numpy assembly downcast matches torch's cast (so
    the conv sees identical bits), and no channel overflows fp16's range.

    `shuffle_buffer_size=0` is the deterministic per-perspective walk, so the
    two datasets yield frame-aligned batches and can be zipped.
    """
    ds32 = IterableDataset(
        samples=samples, seed=0, shuffle_buffer_size=0,
        obs_cfg=replace(OBS_CONFIG_DEFAULTS, obs_dtype="fp32"),
    )
    ds16 = IterableDataset(
        samples=samples, seed=0, shuffle_buffer_size=0,
        obs_cfg=replace(OBS_CONFIG_DEFAULTS, obs_dtype="fp16"),
    )
    loader32 = DataLoader(ds32, batch_size=BATCH_SIZE)
    loader16 = DataLoader(ds16, batch_size=BATCH_SIZE)

    checked = 0
    for b32, b16 in islice(zip(loader32, loader16, strict=False), 10):
        o32, o16 = b32["obs"], b16["obs"]
        assert o32.dtype == torch.float32 and o16.dtype == torch.float16
        assert torch.isfinite(o16).all(), "fp16 obs has inf: a channel exceeds fp16 range"
        assert torch.equal(o16, o32.half()), "fp16-built obs != fp32 build cast to fp16"
        checked += o32.shape[0]
    assert checked > 0, "no frames compared — corpus walk produced nothing"


def test_player_status_channels_are_append_only(samples: list[tuple[Path, int]]) -> None:
    """The player-status group appends 14 channels and leaves the base obs
    byte-identical — the guarantee that a pre-status checkpoint (status off)
    reconstructs the original obs and runs unchanged. Frame-aligned via the
    deterministic `shuffle_buffer_size=0` walk; fp32 so the compare is exact.
    """
    off_cfg = replace(OBS_CONFIG_DEFAULTS, obs_dtype="fp32", player_status_channels=False)
    on_cfg = replace(OBS_CONFIG_DEFAULTS, obs_dtype="fp32", player_status_channels=True)
    n_base = off_cfg.obs_channels  # 96 — no status channels
    assert on_cfg.obs_channels == n_base + 14

    loader_off = DataLoader(
        IterableDataset(samples=samples, seed=0, shuffle_buffer_size=0, obs_cfg=off_cfg),
        batch_size=BATCH_SIZE,
    )
    loader_on = DataLoader(
        IterableDataset(samples=samples, seed=0, shuffle_buffer_size=0, obs_cfg=on_cfg),
        batch_size=BATCH_SIZE,
    )

    checked = 0
    for b_off, b_on in islice(zip(loader_off, loader_on, strict=False), 10):
        o_off, o_on = b_off["obs"], b_on["obs"]
        assert o_off.shape[1] == n_base and o_on.shape[1] == n_base + 14
        assert torch.equal(o_on[:, :n_base], o_off), "base channels changed when status appended"
        # The appended status channels are broadcast 0/1 masks (padding stays 0).
        status = o_on[:, n_base:]
        assert torch.isin(status, torch.tensor([0.0, 1.0])).all()
        checked += o_off.shape[0]
    assert checked > 0, "no frames compared — corpus walk produced nothing"
