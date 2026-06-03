"""
End-to-end smoke test for the BC dataloader pipeline.

Wraps `bc.dataset.IterableDataset` in a torch DataLoader, pulls 100 batches
of size 64, and asserts the tensor contract holds on real-corpus data:
shapes, dtypes, no NaN/inf, target ranges, mask covers the recorded action
on non-pass frames, split-independence.
"""

from __future__ import annotations

from itertools import islice
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from bc.constants import H_PADDED, W_PADDED
from bc.dataset import IterableDataset
from bc.model import BCModel
from bc.obs_config import OBS_CHANNELS, OBS_CONFIG_DEFAULTS


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
        assert obs.dtype == torch.float32
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
        out = model(obs, valid_mask)
    assert out["policy_logits"].shape == (BATCH_SIZE, 8, H_PADDED, W_PADDED)
    assert out["pass_logit"].shape == (BATCH_SIZE,)
    assert out["value_logits"].shape == (BATCH_SIZE, 8)
