"""Frozen-representation probe framework — reusable across probe targets.

A probe answers one question: *is signal X decodable from a frozen
representation Y, at this data scale — and does decoding it need capacity?*
We freeze a representation, cache it over a held-out val sub-split, then train
small swap-in heads on it and read their generalization.

Two axes vary between probes; everything else is shared and lives here:

  - `FeatureSource` — *which representation* we read.
      `TrunkSource` wraps a frozen trained trunk; point it at different
      checkpoints (elim-ON vs elim-OFF) to ask whether a given training
      objective preserved the signal. `RawObsSource` is the identity control
      — it reads the obs directly, bounding what *any* readout of the inputs
      can recover (the sanity anchor: a signal that lives in an obs channel
      must be recoverable here, or the probe itself is broken).

  - `ProbeTask` — *which signal* we decode: the per-frame target, the head
      variants to try (linear vs. fat = "is it readable, and does reading it
      need capacity"), the loss, the metrics, and the model-free baselines to
      draw as reference lines.

The orchestrator (`run_probe`) is target-agnostic: it caches once per source,
then loops the task's head variants on the cached features. Adding a probe is
one new `ProbeTask`; reusing it across representations is a swapped
`FeatureSource`.

Why the cache lives on CPU: MPS chokes ("tensor dims larger than INT_MAX") on
op graphs compiled against gigabyte-scale parent tensors even when we minibatch
into them, so features are parked on CPU and each minibatch is moved to the
device at train time. On M1 unified memory that copy is bookkeeping; on CUDA it
overlaps the step. (Same rationale as the original value-head probe.)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import random
from typing import Protocol

import torch
import torch.nn as nn

from training.bc.aux_heads.elim_head_meta import ElimHeadVariant
from training.bc.config.targets_config import TargetsConfig
from training.bc.dataset import IterableDataset
from training.bc.emit_spec import PartialEmitSpec
from training.bc.model.bc_model import BCModel
from training.bc.obs_config import ObsConfig
from training.bc.storage.checkpoint import load_checkpoint


# ---------------------------------------------------------------------------
# Frame needs — what a task declares it needs the dataset to emit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameNeeds:
    """What a probe task needs each frame to carry beyond the always-on obs +
    valid_mask. Translated 1:1 into an emit spec by `emit_partial_for`.

    - `elim_variant`: an elimination head's per-frame targets (`next_death` /
      `time_bin`) — for tasks that predict elimination.
    - `alive_mask`: the per-player alive field — for tasks that need the
      softmax/regression domain (e.g. lowest-army, army-regression).
    """

    elim_variant: str | None = None
    alive_mask: bool = False


def emit_partial_for(needs: FrameNeeds) -> PartialEmitSpec:
    return PartialEmitSpec(
        targets=TargetsConfig(
            elim_variant=ElimHeadVariant.coerce(needs.elim_variant), elim_bin_edges=None
        ),
        emit_alive_mask=needs.alive_mask,
        attach_sim_frame=False,
    )


# ---------------------------------------------------------------------------
# Feature sources — what representation the probe head reads
# ---------------------------------------------------------------------------


class FeatureSource(Protocol):
    """Maps a batch of obs `[B, C_obs, H, W]` to the feature map the head reads
    `[B, C, H, W]`. The head is sized from the cached features' channel count,
    so the source only has to produce them."""

    def __call__(self, obs: torch.Tensor) -> torch.Tensor: ...


class TrunkSource:
    """A frozen trained trunk. Swap the checkpoint to compare representations
    shaped by different training objectives (e.g. elim-ON vs elim-OFF)."""

    def __init__(self, trunk: nn.Module):
        self.trunk = trunk

    @torch.no_grad()
    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        return self.trunk(obs)


class RawObsSource:
    """Identity — the head reads the obs directly. The control that bounds what
    any readout of the inputs can recover (no trunk in the path)."""

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        return obs


# ---------------------------------------------------------------------------
# Probe task — what signal we decode
# ---------------------------------------------------------------------------


class ProbeTask(Protocol):
    """One decodable-signal question. Implementations own the target, the head
    zoo, the loss, the metrics, and the reference baselines.

    `extract_target(frame)` returns a per-frame dict with key `"target"` plus
    any aux tensors the heads/loss/metrics need (e.g. `valid_mask`,
    `alive_mask`). All entries are stacked into the cache.

    Heads take `(feats[B,C,H,W], aux)` where `aux` holds the batched aux tensors
    on-device, and return predictions `[B, ...]` (logits or regressed values).
    `loss`/`metrics` receive the full prediction/target/aux tensors.
    """

    name: str
    baselines: dict[str, float]
    frame_needs: FrameNeeds  # what the dataset must emit per frame (see FrameNeeds)

    def extract_target(self, frame: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]: ...
    def build_heads(self, in_ch: int, H: int, W: int) -> dict[str, nn.Module]: ...
    def loss(self, pred: torch.Tensor, target: torch.Tensor, aux: dict) -> torch.Tensor: ...
    def metrics(self, pred: torch.Tensor, target: torch.Tensor, aux: dict) -> dict[str, float]: ...


# ---------------------------------------------------------------------------
# Frozen model loading
# ---------------------------------------------------------------------------


def load_frozen_model(ckpt_path: Path, device: torch.device) -> BCModel:
    """Load a checkpoint into a full `BCModel`, frozen and in eval mode, cast to
    fp32 for stable CPU/MPS forward passes.

    Architecture (width, value/elim variant, obs config) comes from the
    checkpoint's recorded arch — `load_checkpoint` makes the `state_dict` load
    strict, so any arch↔weights drift raises loudly here rather than producing
    silently-wrong features.
    """
    model = load_checkpoint(ckpt_path, device).model
    model.float()
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def split_perspectives(
    val_samples: list[tuple[Path, int]],
    val_frac: float,
    seed: int,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    """Sub-split `(sim_path, k)` perspectives into (train_probe, val_probe).

    Perspective-level (not frame-level) so the two never share frames from the
    same game: a frame-level split lets a head memorize "what this perspective
    looks like" — adjacent timesteps in one game have near-identical features —
    rather than the signal we're probing for.
    """
    rng = random.Random(seed)
    shuffled = list(val_samples)
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_frac))
    return shuffled[n_val:], shuffled[:n_val]


# ---------------------------------------------------------------------------
# Feature cache
# ---------------------------------------------------------------------------


@dataclass
class ProbeCache:
    """Cached features + targets + aux for one sub-split, all on CPU."""

    feats: torch.Tensor                       # [N, C, H, W]
    target: torch.Tensor                      # [N, ...]
    aux: dict[str, torch.Tensor] = field(default_factory=dict)  # each [N, ...]

    @property
    def n(self) -> int:
        return self.feats.shape[0]


@torch.no_grad()
def cache_probe_features(
    source: FeatureSource,
    task: ProbeTask,
    samples: list[tuple[Path, int]],
    obs_cfg: ObsConfig,
    device: torch.device,
    seed: int,
    chunk: int = 64,
    max_frames: int | None = None,
) -> ProbeCache:
    """Walk `samples`, run `source` over obs in chunks, and stack each frame's
    `task.extract_target` output. Features land on CPU; transient device memory
    is bounded by `chunk`. `max_frames` caps total frames kept (the walk stops
    once the cap is hit) — useful for a quick smoke run.
    """
    ds = IterableDataset(
        samples=samples,
        seed=seed,
        spec=emit_partial_for(task.frame_needs).to_spec(obs_cfg, emit_frame_info=False),
    )

    obs_buf: list[torch.Tensor] = []
    extracted_buf: list[dict[str, torch.Tensor]] = []
    feats_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    aux_chunks: dict[str, list[torch.Tensor]] = defaultdict(list)
    n_kept = 0

    def flush() -> None:
        nonlocal n_kept
        if not obs_buf:
            return
        obs_batch = torch.stack(obs_buf).to(device).float()
        # Store features fp16 on CPU — the [N, C, H, W] cache is the run's memory
        # ceiling (≈0.75 MB/frame at C=192), so fp16 halves it. Minibatches cast
        # back to fp32 at train/eval time.
        feats_chunks.append(source(obs_batch).cpu().half())
        target_chunks.append(torch.stack([e["target"] for e in extracted_buf]))
        for key in extracted_buf[0]:
            if key == "target":
                continue
            aux_chunks[key].append(torch.stack([e[key] for e in extracted_buf]))
        n_kept += len(obs_buf)
        obs_buf.clear()
        extracted_buf.clear()

    for frame in ds:
        obs_buf.append(frame["obs"])
        extracted_buf.append(task.extract_target(frame))
        # Cap check precedes the chunk-flush: trim the still-buffered overflow to
        # land exactly on max_frames, then flush + stop. (If the chunk-flush ran
        # first it would commit past the cap, leaving nothing to trim.)
        if max_frames is not None and n_kept + len(obs_buf) >= max_frames:
            overflow = (n_kept + len(obs_buf)) - max_frames
            if overflow > 0:
                del obs_buf[-overflow:]
                del extracted_buf[-overflow:]
            flush()
            break
        if len(obs_buf) >= chunk:
            flush()
    flush()

    return ProbeCache(
        feats=torch.cat(feats_chunks, dim=0),
        target=torch.cat(target_chunks, dim=0),
        aux={k: torch.cat(v, dim=0) for k, v in aux_chunks.items()},
    )


# ---------------------------------------------------------------------------
# Head training / eval
# ---------------------------------------------------------------------------


def _evaluate(
    head: nn.Module,
    task: ProbeTask,
    cache: ProbeCache,
    batch_size: int,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    """Run `head` over the full cache in minibatches, collect predictions, then
    compute the task's loss + metrics once on the whole split."""
    head.eval()
    preds: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, cache.n, batch_size):
            sl = slice(start, start + batch_size)
            feats = cache.feats[sl].to(device).float()
            aux = {k: v[sl].to(device) for k, v in cache.aux.items()}
            preds.append(head(feats, aux).cpu().float())
    pred = torch.cat(preds, dim=0)
    loss = task.loss(pred, cache.target, cache.aux).item()
    metrics = task.metrics(pred, cache.target, cache.aux)
    return loss, metrics


def train_probe_head(
    head: nn.Module,
    task: ProbeTask,
    train: ProbeCache,
    val: ProbeCache,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> dict[str, list[float]]:
    """Train `head` on the cached train split, eval on val each epoch. Returns
    per-epoch curves (entry 0 = pre-training baseline) for `train_loss`,
    `val_loss`, and each task metric (prefixed `val_`)."""
    optim = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    curves: dict[str, list[float]] = defaultdict(list)

    def record(label: str) -> None:
        train_loss, _ = _evaluate(head, task, train, batch_size, device)
        val_loss, val_metrics = _evaluate(head, task, val, batch_size, device)
        curves["train_loss"].append(train_loss)
        curves["val_loss"].append(val_loss)
        metric_str = "  ".join(f"{k} {v:.3f}" for k, v in val_metrics.items())
        for k, v in val_metrics.items():
            curves[f"val_{k}"].append(v)
        print(f"  {label:>8}  train {train_loss:.4f}  val {val_loss:.4f}  {metric_str}")

    print(f"  {'epoch':>8}  {'train':>10}  {'val':>10}  metrics")
    record("init")
    for epoch in range(1, epochs + 1):
        head.train()
        perm = torch.randperm(train.n, generator=rng)
        for start in range(0, train.n, batch_size):
            idx = perm[start : start + batch_size]
            feats = train.feats[idx].to(device).float()
            target = train.target[idx].to(device)
            aux = {k: v[idx].to(device) for k, v in train.aux.items()}
            optim.zero_grad()
            loss = task.loss(head(feats, aux), target, aux)
            loss.backward()
            optim.step()
        record(f"ep {epoch}")

    return dict(curves)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_probe(
    task: ProbeTask,
    source: FeatureSource,
    val_samples: list[tuple[Path, int]],
    obs_cfg: ObsConfig,
    *,
    device: torch.device,
    seed: int,
    val_frac: float,
    epochs: int,
    batch_size: int,
    lr: float,
    cache_chunk: int,
    max_train_frames: int | None,
    max_val_frames: int | None,
) -> dict[str, dict]:
    """Cache the source's features over a val sub-split, then train each of the
    task's head variants on them. Returns `{variant: {curves, n_params}}`."""
    train_p, val_p = split_perspectives(val_samples, val_frac, seed)
    print(f"sub-split: {len(train_p)} train_probe / {len(val_p)} val_probe perspectives")

    print("caching train_probe features ...")
    train_cache = cache_probe_features(
        source, task, train_p, obs_cfg, device, seed, cache_chunk, max_train_frames
    )
    print(f"  train feats {tuple(train_cache.feats.shape)}  ({train_cache.n} frames)")
    print("caching val_probe features ...")
    val_cache = cache_probe_features(
        source, task, val_p, obs_cfg, device, seed, cache_chunk, max_val_frames
    )
    print(f"  val feats   {tuple(val_cache.feats.shape)}  ({val_cache.n} frames)")

    _, C, H, W = train_cache.feats.shape
    results: dict[str, dict] = {}
    for variant, head in task.build_heads(C, H, W).items():
        torch.manual_seed(seed)  # reproducible head init per variant
        head = head.to(device)
        n_params = sum(p.numel() for p in head.parameters())
        print(f"\n--- variant: {variant}  ({n_params:,} params) ---")
        curves = train_probe_head(
            head, task, train_cache, val_cache, epochs, batch_size, lr, seed, device
        )
        results[variant] = {"curves": curves, "n_params": n_params}
    return results
