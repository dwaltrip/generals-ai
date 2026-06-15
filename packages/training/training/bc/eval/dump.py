"""Per-frame stratified val-dump records + the val forward pass behind them.

`iter_val_forward` drives one forward-only pass over a val sample list, yielding
`(host_batch, moved_batch, out)` per batch — the shared loader+loop skeleton
both producers of val metrics run on. `FrameRecordCapture` accumulates one
record per val frame from those batches: the value head's full predicted
distribution + CE, policy CE / entropy / top-k, pass prob, and the frame's
provenance scalars (`frame_t`, `players_alive`, `p_start`, perspective id). When
the elim head is active it also captures per-player columns (bin distribution +
hard CE, bin target, alive mask) carrying a player axis. `dump_path` +
`save_dump` write the columns to `<run>/analysis/stratified_val_epoch_NNN.npz`
plus a sibling meta json.

The producers: `train.train_loop` captures during the per-epoch val pass when
`TrainConfig.dump_val_frames` is set (via `run_val`, which fills a capture
alongside its summary reductions), and `capture_val_frames` runs the same pass
standalone against a saved checkpoint for the offline
`scripts/stratified_val_loss_analysis.py dump` subcommand. Both share
`iter_val_forward`. The `report` subcommand consumes either artifact unchanged.
Frames join across dumps of the same run on `(persp_val_index, frame_t)` — row
order is not significant (multi-worker batch interleave varies between passes).
"""

from __future__ import annotations

from collections.abc import Iterator
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from training.bc.dataset import IterableDataset, assert_safe_loader
from training.bc.model import BCModel, flatten_policy_logits
from training.bc.obs_config import ObsConfig
from training.shared.device import dataloader_kwargs, move_batch, obs_for_model


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
            self._push("alive_mask", host_batch["alive_mask"])

        # who-dies-next head: the cross-player softmax over the alive field, its
        # per-frame CE, the next-victim target, the alive mask, and the
        # ticks-to-next-death horizon. The horizon (`next_elim_dt`) is dumped so
        # the confidence-ramp / horizon-stratified reads are computable offline
        # without a re-run. Masked to the alive players (dead/phantom → -inf →
        # prob 0); winner-tail frames (target -1) get NaN CE (nan-aware
        # reductions downstream), mirroring the pass-frame policy CE.
        if "next_elim_logits" in out:
            alive_mask = moved_batch["alive_mask"]
            nd_logits = out["next_elim_logits"].float().masked_fill(
                ~alive_mask, float("-inf")
            )
            nd_logp = F.log_softmax(nd_logits, dim=1)               # [B, 8]
            nd_target = moved_batch["next_elim_target"]            # [B]; -1 = none
            nd_ce = -nd_logp.gather(
                1, nd_target.clamp(min=0).unsqueeze(1)
            ).squeeze(1)
            nd_ce = torch.where(
                nd_target < 0, torch.full_like(nd_ce, float("nan")), nd_ce
            )
            self._push("next_elim_probs", nd_logp.exp().to(torch.float16))
            self._push("next_elim_ce", nd_ce)
            self._push("next_elim_target", host_batch["next_elim_target"])
            self._push("alive_mask", host_batch["alive_mask"])
            self._push("next_elim_dt", host_batch["next_elim_dt"])

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


def iter_val_forward(
    model: torch.nn.Module,
    val_samples: list[tuple[Path, int]],
    device: torch.device,
    *,
    batch_size: int,
    num_workers: int,
    obs_cfg: ObsConfig,
    include_frame_info: bool,
    elim_bin_edges: tuple[int, ...] | None,
    elim_head_variant: str | None = None,
    seed: int = 0,
    amp_dtype: torch.dtype | None = None,
    pin_memory: bool | None = None,
    prefetch_factor: int = 2,
) -> Iterator[tuple[dict, dict, dict]]:
    """Drive one forward-only val pass, yielding `(host_batch, moved_batch, out)`
    per batch — the loader+loop skeleton shared by both val-metric producers
    (`run_val`'s summary reductions and `capture_val_frames`'s per-frame dump).

    Sets `model.eval()` and runs under `no_grad`; the caller restores `train()`.
    The forward runs under autocast (a no-op when `amp_dtype` is None), but each
    batch is yielded *outside* the autocast block, so the consumer's post-forward
    math runs fp32 — the capture's `.float()` reductions, and `run_val`'s metric
    accumulation. A consumer that wants the loss under autocast (run_val)
    re-enters it itself; that re-entry is numerically a no-op since the loss
    reuses no model weights (autocast's weight-cast cache is irrelevant to it).
    """
    ds = IterableDataset(
        samples=val_samples,
        seed=seed,
        obs_cfg=obs_cfg,
        include_frame_info=include_frame_info,
        elim_bin_edges=elim_bin_edges,
        elim_head_variant=elim_head_variant,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        **dataloader_kwargs(
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            device=device,
        ),
    )
    assert_safe_loader(loader)

    model.eval()
    with torch.no_grad():
        for host_batch in loader:
            moved = move_batch(host_batch, device)
            with torch.amp.autocast(
                device.type,
                dtype=amp_dtype or torch.float32,
                enabled=amp_dtype is not None,
            ):
                out = model(obs_for_model(moved, amp_dtype), moved["valid_mask"])
            yield host_batch, moved, out


def capture_val_frames(
    model: BCModel,
    val_samples: list[tuple[Path, int]],
    device: torch.device,
    *,
    batch_size: int,
    num_workers: int,
    persp_index_map: np.ndarray,
    progress_every_sec: float | None = 30.0,
) -> dict[str, np.ndarray]:
    """Run one offline forward-only val pass against a loaded checkpoint and
    return the finalized per-frame record columns (ready for `save_dump`).

    The offline twin of the in-training capture path: where `run_val` fills a
    `FrameRecordCapture` alongside its summary reductions during training, this
    fills one standalone, with a progress heartbeat for the long interactive
    pass. Always fp32 (`amp_dtype=None`); the elim columns ride along when the
    model carries the head. `persp_index_map` maps the (possibly subsampled)
    walked sample list back to full-val-split positions for `finalize`.
    """
    capture = FrameRecordCapture()
    elim_on = model.cfg.elim_head_variant is not None
    elim_bin_edges = model.cfg.elim_bin_edges if elim_on else None
    start = time.perf_counter()
    last_report = start
    for host_batch, moved, out in iter_val_forward(
        model,
        val_samples,
        device,
        batch_size=batch_size,
        num_workers=num_workers,
        obs_cfg=model.cfg.obs,
        include_frame_info=True,
        elim_bin_edges=elim_bin_edges,
        elim_head_variant=model.cfg.elim_head_variant,
    ):
        capture.add_batch(host_batch, moved, out)
        if progress_every_sec is not None:
            now = time.perf_counter()
            if now - last_report >= progress_every_sec:
                rate = capture.n_frames / (now - start)
                print(
                    f"  ... {capture.n_frames} frames, {rate:.0f} samples/sec",
                    flush=True,
                )
                last_report = now

    dur = time.perf_counter() - start
    rate = capture.n_frames / dur if dur > 0 else 0.0
    print(
        f"  {capture.n_frames} frames in {dur:.1f}s ({rate:.0f} samples/sec)",
        flush=True,
    )
    return capture.finalize(persp_index_map=persp_index_map)


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
