"""Per-frame stratified val-dump records, shared by two producers.

`FrameRecordCapture` accumulates one record per val frame across the batches
of a forward-only pass: the value head's full predicted distribution + CE,
policy CE / entropy / top-k, pass prob, and the frame's provenance scalars
(`frame_t`, `players_alive`, `p_start`, perspective id). When the elim head is
active it also captures per-player columns (bin distribution + hard CE, bin
target, alive mask) carrying a player axis. `dump_path` + `save_dump` write the
columns to `<run>/analysis/stratified_val_epoch_NNN.npz` plus a sibling meta
json.

The producers: `train.train_loop` captures during the per-epoch val pass when
`TrainConfig.dump_val_frames` is set, and the offline
`scripts/stratified_val_loss_analysis.py dump` subcommand runs the same
capture against saved checkpoints. Its `report` subcommand consumes either
artifact unchanged. Frames join across dumps of the same run on
`(persp_val_index, frame_t)` — row order is not significant (multi-worker
batch interleave varies between passes).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from training.bc.model import flatten_policy_logits


class FrameRecordCapture:
    """Accumulates per-frame record columns over a forward-only val pass.

    Call `add_batch` once per batch, then `finalize` to get the concatenated
    column dict for `save_dump`. Columns live on CPU as plain numpy arrays
    (~100 B/frame), so a full cloud-scale val pass stays in the tens of MB.

    Requires the dataset to be built with `include_frame_info=True` — the
    provenance scalars come from the batch, not the model.
    """

    def __init__(self) -> None:
        self._cols: dict[str, list[np.ndarray]] = {}
        self.n_frames = 0

    def _push(self, name: str, t: torch.Tensor) -> None:
        # Copy host-resident tensors: DataLoader batches may live in pinned
        # memory (a limited system-wide resource), and `.numpy()` on a CPU
        # tensor shares its buffer — without the copy, the capture would keep
        # every batch's pinned allocation alive for the whole pass. Device
        # tensors already materialize a fresh host buffer via `.cpu()`.
        arr = t.detach().cpu().numpy()
        if t.device.type == "cpu":
            arr = arr.copy()
        self._cols.setdefault(name, []).append(arr)

    def add_batch(
        self,
        host_batch: dict[str, torch.Tensor],
        moved_batch: dict[str, torch.Tensor],
        out: dict[str, torch.Tensor],
    ) -> None:
        """Record one batch's frames.

        `host_batch` is the pre-`move_batch` dict — the provenance/target
        scalars are consumed host-side, and reading them from the host copy
        skips a device round-trip. `moved_batch` is its on-device counterpart
        the forward consumed; `out` the model outputs. Call outside any
        autocast block so the `.float()` math below runs fp32.
        """
        for key in ("frame_t", "players_alive", "p_start", "sample_idx"):
            self._push(key, host_batch[key])
        self._push("placement", host_batch["value_target"])
        self._push("is_pass", host_batch["is_pass"])

        # Value head: full predicted distribution + per-frame CE. The probs
        # are stored fp16 — they feed only display-grade aggregates
        # (prediction entropy, argmax mode share; measured impact ≤3e-4 on
        # both), and halving them cuts the compressed artifact ~35%. The
        # load-bearing CE column stays fp32.
        value_logp = F.log_softmax(out["value_logits"].float(), dim=1)
        value_ce = -value_logp.gather(
            1, moved_batch["value_target"].unsqueeze(1)
        ).squeeze(1)
        self._push("value_probs", value_logp.exp().to(torch.float16))
        self._push("value_ce", value_ce)

        # Policy head, on the flat masked layout (same as bc_loss). Pass
        # frames carry no action target, so their policy CE is NaN'd in
        # place — downstream report code uses nan-aware reductions.
        masked = flatten_policy_logits(
            out["policy_logits"].float(), moved_batch["mask"]
        )
        pol_logp = F.log_softmax(masked, dim=1)
        target = moved_batch["action_target"]
        pol_ce = -pol_logp.gather(1, target.clamp(min=0).unsqueeze(1)).squeeze(1)
        is_pass = moved_batch["is_pass"]
        pol_ce = torch.where(
            is_pass, torch.full_like(pol_ce, float("nan")), pol_ce
        )
        self._push("policy_ce", pol_ce)

        # Entropy over the masked softmax. MASK_NEG keeps illegal logits
        # finite, so p·log p underflows to 0 there — no NaN guard needed.
        pol_p = pol_logp.exp()
        self._push("policy_entropy", -(pol_p * pol_logp).sum(dim=1))
        self._push(
            "n_legal", moved_batch["mask"].reshape(masked.shape[0], -1).sum(dim=1)
        )

        topk = torch.topk(masked, k=3, dim=1).indices
        non_pass = ~is_pass
        self._push("top1", (topk[:, 0] == target) & non_pass)
        self._push("top3", (target.unsqueeze(1) == topk).any(dim=1) & non_pass)
        self._push("pass_prob", torch.sigmoid(out["pass_logit"].float()))

        # Elim head: per-(player, frame) bin distribution + hard CE. The columns
        # carry a player axis (probs [B, 8, n_bins]; ce/target/mask [B, 8]) and
        # are masked to alive players at report time. Present only when the head
        # is active — non-elim runs add no elim columns. CE is unweighted (the
        # report wants per-bin unweighted CE; matches the loss's `elim` only when
        # `elim_bin_weights` is None, the current default).
        if "elim_logits" in out:
            elim_logp = F.log_softmax(out["elim_logits"].float(), dim=2)
            elim_ce = -elim_logp.gather(
                2, moved_batch["elim_bin_target"].unsqueeze(2)
            ).squeeze(2)
            self._push("elim_probs", elim_logp.exp().to(torch.float16))
            self._push("elim_ce", elim_ce)
            self._push("elim_bin_target", host_batch["elim_bin_target"])
            self._push("elim_alive_mask", host_batch["elim_alive_mask"])

        self.n_frames += int(host_batch["is_pass"].shape[0])

    def finalize(
        self, persp_index_map: np.ndarray | None = None
    ) -> dict[str, np.ndarray]:
        """Concatenate the accumulated columns into the dump's record dict.

        `sample_idx` (the perspective's position in the walked sample list) is
        replaced by `persp_val_index` — its position in the run's full val
        split, the stable join id across dumps. Producers that walk the full
        val list pass `None` (identity); the offline harness passes its
        `--sample-frac` subsample's index map.
        """
        records = {name: np.concatenate(parts) for name, parts in self._cols.items()}
        sample_idx = records.pop("sample_idx")
        records["persp_val_index"] = (
            persp_index_map[sample_idx] if persp_index_map is not None else sample_idx
        )
        return records


def dump_path(run_dir: Path, epoch: int) -> Path:
    return run_dir / "analysis" / f"stratified_val_epoch_{epoch:03d}.npz"


def save_dump(records: dict[str, np.ndarray], path: Path, meta: dict) -> None:
    """Write the record columns as a compressed npz + sibling `.meta.json`.

    `n_frames` is derived from the records here rather than taken from the
    caller's meta, so the two can't desync.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, allow_pickle=False, **records)
    meta = {**meta, "n_frames": int(records["value_ce"].shape[0])}
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
