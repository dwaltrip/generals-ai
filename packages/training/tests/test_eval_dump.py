"""Smoke test for the per-frame stratified capture (`eval.dump`).

Synthetic batches stand in for the val pass — the end-to-end producers
(in-training val, offline harness) are covered by their integration paths;
this checks the record contract: column shapes/dtypes, the per-frame math
against a hand reduction, the pass-frame NaN convention, the
`persp_val_index` remap, and the npz + meta round-trip.
"""

import json

import numpy as np
import torch
import torch.nn.functional as F

from training.bc.eval import FrameRecordCapture, dump_path, save_dump
from training.bc.model import ModelOut


def _fake_batch_and_out(B: int = 4, H: int = 2, W: int = 2):
    g = torch.Generator().manual_seed(0)
    batch = {
        "frame_t": torch.arange(B, dtype=torch.int64),
        "players_alive": torch.full((B,), 4, dtype=torch.int64),
        "p_start": torch.full((B,), 4, dtype=torch.int64),
        "sample_idx": torch.tensor([0, 0, 1, 1]),
        "value_target": torch.tensor([0, 1, 2, 3]),
        "is_pass": torch.tensor([False, True, False, False]),
        "action_target": torch.tensor([3, -1, 7, 12]),
        "mask": torch.ones(B, 8, H, W, dtype=torch.bool),
    }
    out = ModelOut(
        policy_logits=torch.randn(B, 8, H, W, generator=g),
        pass_logit=torch.randn(B, generator=g),
        value_logits=torch.randn(B, 8, generator=g),
    )
    return batch, out


def test_capture_records_and_save(tmp_path):
    capture = FrameRecordCapture()
    batch, out = _fake_batch_and_out()
    capture.add_batch(batch, batch, out)  # host == moved on CPU
    capture.add_batch(batch, batch, out)
    assert capture.n_frames == 8

    records = capture.finalize()
    assert all(col.shape[0] == 8 for col in records.values())
    # sample_idx is replaced by the stable join id (identity without a map).
    assert "sample_idx" not in records
    np.testing.assert_array_equal(records["persp_val_index"][:4], [0, 0, 1, 1])

    # Value head math vs a hand reduction; probs stored fp16, rows sum to 1.
    logp = F.log_softmax(out.value_logits, dim=1)
    expected_ce = -logp[torch.arange(4), batch["value_target"]].numpy()
    np.testing.assert_allclose(records["value_ce"][:4], expected_ce, rtol=1e-6)
    assert records["value_probs"].dtype == np.float16
    np.testing.assert_allclose(records["value_probs"].sum(axis=1), 1.0, atol=2e-3)

    # Pass frames: policy CE is NaN'd in place, top-k excluded.
    is_pass = records["is_pass"]
    assert np.isnan(records["policy_ce"][is_pass]).all()
    assert not np.isnan(records["policy_ce"][~is_pass]).any()
    assert not records["top1"][is_pass].any()

    path = dump_path(tmp_path, epoch=3)
    save_dump(records, path, {"producer": "test"})
    assert path == tmp_path / "analysis" / "stratified_val_epoch_003.npz"
    loaded = dict(np.load(path))
    assert set(loaded) == set(records)
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    assert meta == {"producer": "test", "n_frames": 8}


def test_finalize_persp_index_remap():
    capture = FrameRecordCapture()
    batch, out = _fake_batch_and_out()
    capture.add_batch(batch, batch, out)
    # A subsampled producer maps dataset-local idx -> full-val-list position.
    records = capture.finalize(persp_index_map=np.array([10, 42]))
    np.testing.assert_array_equal(records["persp_val_index"], [10, 10, 42, 42])
