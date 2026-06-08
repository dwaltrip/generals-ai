"""Value-head probe: frozen trunk + swap-in head variants.

Diagnoses whether the dead-ReLU collapse from `5.21-1` is (a) a structural
property of the head's shape, and/or (b) a sign that the trunk's representation
doesn't encode placement information at all.

Mechanism:
  1. Load `epoch_005.pt`; build BCModel; freeze the trunk in eval mode.
  2. Pull a stratified batch of val frames (~per_class per placement class).
  3. Run the trunk once over the batch to cache `[B, C, H, W]` features.
  4. For each head variant: instantiate fresh, train on the cached features
     for N steps with value-CE only, log the loss curve.

The trunk forward is the bulk of compute; caching it once and looping on
the head only makes the per-step cost tiny — typically minutes per variant
for Mode A's single-batch overfit.

Modes:
  - `--mode overfit` (Mode A): single stratified batch, ~500 steps per variant.
    Pass criterion = loss drops below the overfit threshold (default 0.1).
    Filters out structurally-broken heads. Status-quo head should *fail*
    this — confirms the dead-ReLU diagnosis under controlled conditions.
  - `--mode probe` (Mode B): multi-batch eval on a larger val slice, frozen
    trunk. For variants that pass Mode A. Tells us whether the trunk encodes
    board-conditional placement signal at all.

Run from training/:
    uv run python scripts/value_head_probe.py \\
        --checkpoint data/runs/20260521-005930/checkpoints/epoch_005.pt \\
        --manifest data/splits/poc_2kish_perspecs.json \\
        --mode overfit
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import random
import time

import matplotlib


matplotlib.use("Agg")  # headless; must precede pyplot import
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.bc.checkpoint import load_bc_model
from training.bc.constants import H_PADDED, W_PADDED
from training.bc.dataset import IterableDataset
from training.bc.obs_config import OBS_CONFIG_DEFAULTS
from training.bc.splits import load_manifest, samples_for_split
from training.shared.device import disable_mps_fallback, pick_device
from utils.docstring import doc_summary


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INTERMEDIATE = REPO_ROOT / "replay-parser" / "data" / "intermediate"
DEFAULT_OUT_ROOT = REPO_ROOT / "training" / "data" / "probes"

# Frame-weighted marginal placement entropy on the PoC val split. Source:
# 5.21-1; reproducible via `scripts/marginal_entropy.py`. A constant
# predictor (or one that's learned only the marginal) sits at exactly
# this loss. The point of Mode B is "does any variant beat this".
VAL_MARGINAL_ENTROPY = 1.5195


# ---------------------------------------------------------------------------
# Head variants
# ---------------------------------------------------------------------------
#
# All variants take a frozen-trunk feature tensor `[B, C, H, W]` and return
# `[B, 8]` placement logits. The differences are deliberately small — each
# variant probes one hypothesis about the dead-ReLU pathology in `bc/model.py`
# `ValueHead`:
#
#   - status_quo : reproduces the current head. Should *fail* Mode A — the
#                  reproduction is the sanity baseline.
#   - no_relu    : removes the activation. Tests whether ReLU itself is the
#                  trap.
#   - gelu       : swaps ReLU for GELU (smooth, gradient everywhere). Tests
#                  whether the activation can be the same shape but better.
#   - wide_conv  : widens the conv from K=1 to K=8. Tests whether channel
#                  redundancy alone fixes it (one dead channel no longer
#                  catastrophic).
#   - gap_linear : strips to GlobalAvgPool → Linear(C, 8). Simplest possible
#                  classifier; gut-check baseline.


class StatusQuoHead(nn.Module):
    """Reproduces `bc/model.py`'s `ValueHead`. Pathology baseline."""

    def __init__(self, in_ch: int, H: int, W: int, n_classes: int = 8):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 1, kernel_size=3, padding=1)
        self.linear = nn.Linear(H * W, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = F.relu(x)
        x = x.flatten(1)
        return self.linear(x)


class NoReluHead(nn.Module):
    """Status quo minus the ReLU. One-line minimal-change candidate fix."""

    def __init__(self, in_ch: int, H: int, W: int, n_classes: int = 8):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 1, kernel_size=3, padding=1)
        self.linear = nn.Linear(H * W, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.flatten(1)
        return self.linear(x)


class GeluHead(nn.Module):
    """ReLU → GELU. Smooth gradient even in the negative region."""

    def __init__(self, in_ch: int, H: int, W: int, n_classes: int = 8):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 1, kernel_size=3, padding=1)
        self.linear = nn.Linear(H * W, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = F.gelu(x)
        x = x.flatten(1)
        return self.linear(x)


class WideConvHead(nn.Module):
    """Conv(C→K=8) instead of K=1. Channel redundancy against dead-channel
    collapse — one bad channel no longer freezes the whole input vector."""

    def __init__(self, in_ch: int, H: int, W: int, n_classes: int = 8, K: int = 8):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, K, kernel_size=3, padding=1)
        self.linear = nn.Linear(K * H * W, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = F.relu(x)
        x = x.flatten(1)
        return self.linear(x)


class GapLinearHead(nn.Module):
    """Global avg pool → Linear(C, 8). Simplest possible head — no spatial
    bottleneck, no activation between conv and linear. Useful as the
    'how much do we even need' baseline."""

    def __init__(self, in_ch: int, H: int, W: int, n_classes: int = 8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(in_ch, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x).flatten(1)
        return self.linear(x)


VARIANTS: dict[str, type[nn.Module]] = {
    "status_quo": StatusQuoHead,
    "no_relu": NoReluHead,
    "gelu": GeluHead,
    "wide_conv": WideConvHead,
    "gap_linear": GapLinearHead,
}


def build_head(variant: str, in_ch: int, H: int, W: int) -> nn.Module:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant!r}; choices: {list(VARIANTS)}")
    return VARIANTS[variant](in_ch, H, W)


# ---------------------------------------------------------------------------
# Frozen trunk
# ---------------------------------------------------------------------------


def load_frozen_trunk(
    ckpt_path: Path,
    device: torch.device,
    value_head_variant: str = "direct",
) -> nn.Module:
    """Load BCModel state_dict, return the trunk in frozen eval mode.

    Constructs a full BCModel so the state_dict load is strict — catches
    checkpoint/architecture drift loudly. We then extract just the trunk
    and discard the heads; the probe instantiates fresh heads anyway.
    `value_head_variant` must match the variant the checkpoint was trained
    with so the strict load succeeds.
    """
    print(f"loading checkpoint: {ckpt_path}")
    model = load_bc_model(ckpt_path, device, value_head_variant)
    trunk = model.trunk
    for p in trunk.parameters():
        p.requires_grad = False
    trunk.eval()
    return trunk


# ---------------------------------------------------------------------------
# Stratified batch gathering
# ---------------------------------------------------------------------------


def gather_stratified_frames(
    val_samples: list[tuple[Path, int]],
    seed: int,
    per_class: int,
    n_classes: int = 8,
    max_per_perspective: int = 2,
) -> list[dict[str, torch.Tensor]]:
    """Walk val set, accumulate up to `per_class` frames per placement class.

    Caps per-(game, k) contribution at `max_per_perspective` so that the
    rarest classes (8th = 0.7% of frames) don't all come from a single
    perspective. The IterableDataset walks one perspective fully before
    moving on, so without this cap, 8 frames of an 8th-place player would
    all be consecutive timesteps from the same game.
    """
    ds = IterableDataset(samples=val_samples, seed=seed, obs_cfg=OBS_CONFIG_DEFAULTS)

    by_class: dict[int, list[dict]] = defaultdict(list)
    # Track contribution per (sim_path-id-stand-in, perspective) — we don't
    # actually have sim_path on the frame dict, so use placement-class +
    # contiguous-run heuristic: when we see a frame of class c, count how
    # many in a row we've taken. The dataset emits per-perspective contiguously.
    last_class = None
    run_count = 0
    target_total = per_class * n_classes

    for frame in ds:
        cls = int(frame["value_target"].item())
        if cls == last_class:
            run_count += 1
        else:
            run_count = 1
            last_class = cls

        if len(by_class[cls]) >= per_class:
            continue
        if run_count > max_per_perspective:
            continue
        by_class[cls].append(frame)

        if sum(len(v) for v in by_class.values()) >= target_total:
            break

    return [f for cls in range(n_classes) for f in by_class[cls]]


def collate_frames(frames: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack a list of per-frame dicts into a batch dict (DataLoader's default
    collate, sans the DataLoader.)"""
    return {k: torch.stack([f[k] for f in frames]) for k in frames[0]}


# ---------------------------------------------------------------------------
# Trunk feature caching
# ---------------------------------------------------------------------------


@torch.no_grad()
def cache_trunk_features(
    trunk: nn.Module,
    obs: torch.Tensor,
    device: torch.device,
    chunk: int = 64,
) -> torch.Tensor:
    """Run trunk over `obs` in chunks; return `[N, C, H, W]` features on device.

    Chunking matters for Mode B (thousands of frames) where one giant
    forward pass would OOM on MPS. Mode A's stratified batch fits in one
    chunk but uses the same path.
    """
    feats_chunks: list[torch.Tensor] = []
    for start in range(0, obs.shape[0], chunk):
        x = obs[start : start + chunk].to(device)
        feats_chunks.append(trunk(x))
    return torch.cat(feats_chunks, dim=0)


# ---------------------------------------------------------------------------
# Head training loop
# ---------------------------------------------------------------------------


def train_head_on_cache(
    head: nn.Module,
    feats: torch.Tensor,
    targets: torch.Tensor,
    steps: int,
    lr: float,
    log_every: int,
) -> list[float]:
    """Train `head` on cached `(feats, targets)` for `steps`. Returns the
    per-step loss list (length steps+1 because we log step 0 pre-update).

    Full-batch update each step — Mode A's batch fits comfortably and we
    want the cleanest "can it memorize" signal. No mini-batching here.
    """
    head.train()
    optim = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)

    losses: list[float] = []
    for step in range(steps + 1):
        optim.zero_grad()
        logits = head(feats)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optim.step()
        losses.append(float(loss.item()))
        if step % log_every == 0 or step == steps:
            print(f"  step {step:>4}  loss {loss.item():.4f}")
    return losses


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_overfit_curves(
    losses_per_variant: dict[str, list[float]],
    out: Path,
    marginal_entropy: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, losses in losses_per_variant.items():
        ax.plot(losses, label=name, linewidth=1.5)
    if marginal_entropy is not None:
        ax.axhline(
            marginal_entropy,
            color="grey",
            linestyle=":",
            linewidth=0.8,
            label=f"marginal H = {marginal_entropy:.4f}",
        )
    ax.set_xlabel("step")
    ax.set_ylabel("value CE loss")
    ax.set_yscale("log")
    ax.set_title("Value-head probe: per-step loss (frozen trunk)")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Mode A orchestrator
# ---------------------------------------------------------------------------


def run_overfit_mode(
    trunk: nn.Module,
    val_samples: list[tuple[Path, int]],
    variants: list[str],
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
) -> None:
    print(f"\n--- gather stratified batch (target {args.per_class}/class) ---")
    t0 = time.perf_counter()
    frames = gather_stratified_frames(
        val_samples, seed=args.seed, per_class=args.per_class,
    )
    batch = collate_frames(frames)
    cls_counts = torch.bincount(batch["value_target"], minlength=8).tolist()
    print(
        f"  collected {len(frames)} frames in {time.perf_counter() - t0:.1f}s; "
        f"per-class: {cls_counts}"
    )

    print("\n--- cache trunk features ---")
    t0 = time.perf_counter()
    feats = cache_trunk_features(trunk, batch["obs"], device)
    targets = batch["value_target"].to(device)
    print(
        f"  feats: {tuple(feats.shape)} on {feats.device} "
        f"in {time.perf_counter() - t0:.1f}s"
    )

    losses_per_variant: dict[str, list[float]] = {}
    summary: dict[str, dict] = {}
    for variant in variants:
        print(f"\n--- variant: {variant} ---")
        torch.manual_seed(args.seed)  # head init reproducible per variant
        head = build_head(
            variant, in_ch=feats.shape[1], H=feats.shape[2], W=feats.shape[3]
        ).to(device)
        n_params = sum(p.numel() for p in head.parameters())
        print(f"  head params: {n_params:,}")

        t0 = time.perf_counter()
        losses = train_head_on_cache(
            head, feats, targets,
            steps=args.steps, lr=args.lr, log_every=args.log_every,
        )
        elapsed = time.perf_counter() - t0
        final = losses[-1]
        verdict = "OVERFIT" if final < args.overfit_threshold else "STUCK"
        print(f"  final {final:.4f}  |  min {min(losses):.4f}  |  {elapsed:.1f}s  [{verdict}]")

        losses_per_variant[variant] = losses
        summary[variant] = {
            "final_loss": final,
            "min_loss": min(losses),
            "verdict": verdict,
            "n_params": n_params,
            "duration_sec": round(elapsed, 2),
        }

    plot_path = out_dir / "overfit_loss.png"
    plot_overfit_curves(losses_per_variant, plot_path, marginal_entropy=VAL_MARGINAL_ENTROPY)
    print(f"\nwrote plot: {plot_path}")

    with (out_dir / "overfit_summary.json").open("w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"wrote summary: {out_dir / 'overfit_summary.json'}")

    # Console table — easy at-a-glance view alongside the plot.
    print("\n=== Mode A summary ===")
    print(f"  {'variant':<12} {'final':>8} {'min':>8} {'verdict':>10}")
    for variant, row in summary.items():
        print(
            f"  {variant:<12} {row['final_loss']:>8.4f} "
            f"{row['min_loss']:>8.4f} {row['verdict']:>10}"
        )


# ---------------------------------------------------------------------------
# Mode B: perspective-level sub-split
# ---------------------------------------------------------------------------


def split_perspectives(
    val_samples: list[tuple[Path, int]],
    val_frac: float,
    seed: int,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    """Sub-split a list of (sim_path, k) at the perspective level.

    Perspective-level (not frame-level) so train_probe and val_probe never
    share frames from the same perspective. A frame-level split would let
    the head memorize "what perspective X looks like" rather than "what
    placement looks like" — and the trunk's spatial features for adjacent
    timesteps in the same game are nearly identical.
    """
    rng = random.Random(seed)
    shuffled = list(val_samples)
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_frac))
    return shuffled[n_val:], shuffled[:n_val]


# ---------------------------------------------------------------------------
# Mode B: streaming trunk feature cache
# ---------------------------------------------------------------------------


@torch.no_grad()
def stream_trunk_features(
    trunk: nn.Module,
    samples: list[tuple[Path, int]],
    device: torch.device,
    seed: int,
    chunk: int = 64,
    max_frames: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Walk `samples` via IterableDataset, run trunk in chunks, return
    `(feats[N, C, H, W], targets[N])` **on CPU**.

    Streams obs frames into a chunk buffer, flushes through the trunk
    (on `device`), drops obs, parks the chunk's features on CPU. Peak
    transient memory on `device` is bounded by `chunk × obs_size`.

    Why CPU-side cache: MPS hits "tensor dims larger than INT_MAX" on
    op graphs compiled against gigabyte-scale parent tensors, even when
    we minibatch into them. On M1 unified memory the CPU↔MPS copy is
    bookkeeping, not real transfer — minibatches `.to(device)` cheaply
    at training time. On CUDA boxes the per-minibatch copy is real but
    overlaps the SGD step.

    `max_frames` caps the total samples kept; the walk stops once the
    cap is reached. Useful for memory-tight runs.
    """
    ds = IterableDataset(samples=samples, seed=seed, obs_cfg=OBS_CONFIG_DEFAULTS)

    obs_buffer: list[torch.Tensor] = []
    target_buffer: list[torch.Tensor] = []
    feats_chunks: list[torch.Tensor] = []
    targets_chunks: list[torch.Tensor] = []
    n_kept = 0

    def flush() -> None:
        nonlocal n_kept
        if not obs_buffer:
            return
        obs_batch = torch.stack(obs_buffer).to(device)
        feats_chunks.append(trunk(obs_batch).cpu())
        targets_chunks.append(torch.stack(target_buffer))
        n_kept += len(obs_buffer)
        obs_buffer.clear()
        target_buffer.clear()

    for frame in ds:
        obs_buffer.append(frame["obs"])
        target_buffer.append(frame["value_target"])
        if len(obs_buffer) >= chunk:
            flush()
        if max_frames is not None and n_kept + len(obs_buffer) >= max_frames:
            # Trim the buffer to land exactly on the cap, then exit.
            overflow = (n_kept + len(obs_buffer)) - max_frames
            if overflow > 0:
                del obs_buffer[-overflow:]
                del target_buffer[-overflow:]
            flush()
            break
    flush()

    return torch.cat(feats_chunks, dim=0), torch.cat(targets_chunks, dim=0)


# ---------------------------------------------------------------------------
# Mode B: mini-batched head training
# ---------------------------------------------------------------------------


def evaluate_head(
    head: nn.Module,
    feats: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    """Run head over cached features, return `(mean_ce_loss, top1_accuracy)`.

    `feats` / `targets` live on CPU (see `stream_trunk_features` rationale);
    each minibatch is moved to `device` here. Mini-batching keeps peak
    `device`-side memory predictable on large val caches.
    """
    head.eval()
    total_loss = 0.0
    n_correct = 0
    n_seen = 0
    with torch.no_grad():
        for start in range(0, feats.shape[0], batch_size):
            f = feats[start : start + batch_size].to(device)
            t = targets[start : start + batch_size].to(device)
            logits = head(f)
            loss = F.cross_entropy(logits, t, reduction="sum")
            total_loss += float(loss.item())
            n_correct += int((logits.argmax(dim=1) == t).sum().item())
            n_seen += t.shape[0]
    return total_loss / n_seen, n_correct / n_seen


def train_head_minibatched(
    head: nn.Module,
    train_feats: torch.Tensor,
    train_targets: torch.Tensor,
    val_feats: torch.Tensor,
    val_targets: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> dict[str, list[float]]:
    """Train `head` on the cached train_probe features; eval on val_probe each epoch.

    `train_feats` / `val_feats` live on CPU; each minibatch is moved to
    `device` here. Returns per-epoch curves (`train_loss`, `val_loss`,
    `val_top1`), each with length `epochs + 1` (entry 0 = pre-training
    baseline).
    """
    optim = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    n_train = train_feats.shape[0]
    rng = torch.Generator(device="cpu").manual_seed(seed)

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_top1s: list[float] = []

    def record(epoch_label: str) -> None:
        train_loss, _ = evaluate_head(head, train_feats, train_targets, batch_size, device)
        val_loss, val_top1 = evaluate_head(head, val_feats, val_targets, batch_size, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_top1s.append(val_top1)
        print(
            f"  {epoch_label:>8}  train {train_loss:.4f}  "
            f"val {val_loss:.4f}  val_top1 {val_top1:.3f}"
        )

    print(f"  {'epoch':>8}  {'train':>10}  {'val':>10}  {'val_top1':>10}")
    record("init")

    for epoch in range(1, epochs + 1):
        head.train()
        # Reshuffle the per-epoch order so SGD sees fresh batch compositions.
        perm = torch.randperm(n_train, generator=rng)
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            f = train_feats[idx].to(device)
            t = train_targets[idx].to(device)
            optim.zero_grad()
            logits = head(f)
            loss = F.cross_entropy(logits, t)
            loss.backward()
            optim.step()
        record(f"ep {epoch}")

    return {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_top1": val_top1s,
    }


# ---------------------------------------------------------------------------
# Mode B: plotting
# ---------------------------------------------------------------------------


def plot_probe_curves(
    curves_per_variant: dict[str, dict[str, list[float]]],
    out: Path,
    marginal_entropy: float,
) -> None:
    """Two-panel plot: per-variant val loss (left), per-variant val top-1 (right).

    Train loss is omitted from the plot — it's saved in the JSON summary and
    visible in stdout. The interesting comparison is val behavior across
    variants, with the marginal-entropy / always-1st baselines as horizontal
    references.
    """
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 5))

    for name, curves in curves_per_variant.items():
        epochs = list(range(len(curves["val_loss"])))
        ax_loss.plot(epochs, curves["val_loss"], marker="o", label=name)
        ax_acc.plot(epochs, curves["val_top1"], marker="o", label=name)

    ax_loss.axhline(
        marginal_entropy, color="grey", linestyle=":", linewidth=0.8,
        label=f"marginal H = {marginal_entropy:.4f}",
    )
    ax_loss.set_xlabel("epoch (0 = pre-training)")
    ax_loss.set_ylabel("val CE loss")
    ax_loss.set_title("Probe val loss vs. epoch (frozen trunk)")
    ax_loss.legend(fontsize=8)
    ax_loss.grid(True, alpha=0.3)

    # Always-1st baseline is the frame-weighted marginal's top class share.
    # Source: 5.21-1 §Supporting evidence — 47.6% on the PoC val split.
    ax_acc.axhline(
        0.476, color="grey", linestyle=":", linewidth=0.8,
        label="always-1st (val frame-weighted = 0.476)",
    )
    ax_acc.set_xlabel("epoch (0 = pre-training)")
    ax_acc.set_ylabel("val top-1 accuracy")
    ax_acc.set_title("Probe val top-1 vs. epoch")
    ax_acc.legend(fontsize=8)
    ax_acc.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Mode B orchestrator
# ---------------------------------------------------------------------------


def run_probe_mode(
    trunk: nn.Module,
    val_samples: list[tuple[Path, int]],
    variants: list[str],
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
) -> None:
    val_frac = args.probe_val_frac
    print(f"\n--- sub-split val perspectives ({1 - val_frac:.0%}/{val_frac:.0%}) ---")
    train_persp, val_persp = split_perspectives(val_samples, args.probe_val_frac, args.seed)
    print(f"  train_probe: {len(train_persp)} perspectives")
    print(f"  val_probe:   {len(val_persp)} perspectives")

    print("\n--- cache trunk features (train_probe) ---")
    t0 = time.perf_counter()
    train_feats, train_targets = stream_trunk_features(
        trunk, train_persp, device, seed=args.seed,
        chunk=args.cache_chunk, max_frames=args.probe_max_train_frames,
    )
    print(
        f"  train_feats: {tuple(train_feats.shape)} on {train_feats.device} "
        f"in {time.perf_counter() - t0:.1f}s"
    )
    train_cls = torch.bincount(train_targets.cpu(), minlength=8).tolist()
    print(f"  train per-class: {train_cls}")

    print("\n--- cache trunk features (val_probe) ---")
    t0 = time.perf_counter()
    val_feats, val_targets = stream_trunk_features(
        trunk, val_persp, device, seed=args.seed,
        chunk=args.cache_chunk, max_frames=args.probe_max_val_frames,
    )
    print(
        f"  val_feats:   {tuple(val_feats.shape)} on {val_feats.device} "
        f"in {time.perf_counter() - t0:.1f}s"
    )
    val_cls = torch.bincount(val_targets.cpu(), minlength=8).tolist()
    print(f"  val per-class:   {val_cls}")

    curves_per_variant: dict[str, dict[str, list[float]]] = {}
    summary: dict[str, dict] = {}
    for variant in variants:
        print(f"\n--- variant: {variant} ---")
        torch.manual_seed(args.seed)
        head = build_head(
            variant, in_ch=train_feats.shape[1],
            H=train_feats.shape[2], W=train_feats.shape[3],
        ).to(device)
        n_params = sum(p.numel() for p in head.parameters())
        print(f"  head params: {n_params:,}")

        t0 = time.perf_counter()
        curves = train_head_minibatched(
            head,
            train_feats, train_targets,
            val_feats, val_targets,
            epochs=args.probe_epochs,
            batch_size=args.probe_batch_size,
            lr=args.lr,
            seed=args.seed,
            device=device,
        )
        elapsed = time.perf_counter() - t0
        final_val_loss = curves["val_loss"][-1]
        final_val_top1 = curves["val_top1"][-1]
        best_val_loss = min(curves["val_loss"])
        beats_marginal = final_val_loss < VAL_MARGINAL_ENTROPY
        verdict = "BEATS MARGINAL" if beats_marginal else "AT/ABOVE MARGINAL"
        print(
            f"  final: val {final_val_loss:.4f}  "
            f"top1 {final_val_top1:.3f}  |  {elapsed:.1f}s  [{verdict}]"
        )

        curves_per_variant[variant] = curves
        summary[variant] = {
            "n_params": n_params,
            "final_val_loss": final_val_loss,
            "final_val_top1": final_val_top1,
            "best_val_loss": best_val_loss,
            "delta_vs_marginal": final_val_loss - VAL_MARGINAL_ENTROPY,
            "verdict": verdict,
            "duration_sec": round(elapsed, 2),
            "curves": curves,
        }

    plot_path = out_dir / "probe_curves.png"
    plot_probe_curves(curves_per_variant, plot_path, marginal_entropy=VAL_MARGINAL_ENTROPY)
    print(f"\nwrote plot: {plot_path}")

    with (out_dir / "probe_summary.json").open("w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"wrote summary: {out_dir / 'probe_summary.json'}")

    # Console table — at-a-glance view alongside the plot.
    print("\n=== Mode B summary ===")
    print(
        f"  {'variant':<12} {'val_loss':>10} {'Δmarg':>10} "
        f"{'val_top1':>10} {'verdict':>20}"
    )
    for variant, row in summary.items():
        print(
            f"  {variant:<12} {row['final_val_loss']:>10.4f} "
            f"{row['delta_vs_marginal']:>+10.4f} "
            f"{row['final_val_top1']:>10.3f} {row['verdict']:>20}"
        )
    print(f"  (marginal H = {VAL_MARGINAL_ENTROPY:.4f}; always-1st top1 = 0.476)")


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=doc_summary(__doc__))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--intermediate", type=Path, default=DEFAULT_INTERMEDIATE)
    parser.add_argument(
        "--mode", choices=("overfit", "probe"), default="overfit",
        help="overfit = Mode A (single stratified batch); probe = Mode B (deferred).",
    )
    parser.add_argument(
        "--variants", nargs="+", default=None,
        choices=list(VARIANTS.keys()),
        help="Subset of variants to run (default: all 5).",
    )
    parser.add_argument(
        "--per-class", type=int, default=8,
        help="Stratified samples per placement class for Mode A.",
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--overfit-threshold", type=float, default=0.1,
        help="Final-loss threshold below which Mode A counts as OVERFIT.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--value-head",
        choices=("direct", "pyramid"),
        default="direct",
        help="Variant the checkpoint was trained with (for strict state_dict load).",
    )

    # --- Mode B (probe) knobs ---
    parser.add_argument(
        "--probe-val-frac", type=float, default=0.2,
        help="Mode B: fraction of val perspectives held out for val_probe eval.",
    )
    parser.add_argument(
        "--probe-epochs", type=int, default=5,
        help="Mode B: epochs over train_probe features.",
    )
    parser.add_argument(
        "--probe-batch-size", type=int, default=128,
        help="Mode B: mini-batch size during head training.",
    )
    parser.add_argument(
        "--cache-chunk", type=int, default=64,
        help="Mode B: chunk size for streaming trunk forward during caching.",
    )
    parser.add_argument(
        "--probe-max-train-frames", type=int, default=None,
        help="Mode B: cap train_probe cache size (default: no cap; use all frames).",
    )
    parser.add_argument(
        "--probe-max-val-frames", type=int, default=None,
        help="Mode B: cap val_probe cache size (default: no cap).",
    )
    return parser


def _make_run_dir(out_root: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = out_root / f"{ts}-value-probe"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main() -> None:
    args = _build_arg_parser().parse_args()
    disable_mps_fallback()

    if not args.checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    if not args.manifest.exists():
        raise SystemExit(f"manifest not found: {args.manifest}")
    if not args.intermediate.exists():
        raise SystemExit(f"intermediate corpus not found: {args.intermediate}")

    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    print(f"device: {device}")

    out_dir = _make_run_dir(args.out_root)
    print(f"out dir: {out_dir}")
    with (out_dir / "args.json").open("w") as fp:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  fp, indent=2)

    manifest = load_manifest(args.manifest)
    val_samples = samples_for_split(manifest, "val", args.intermediate)
    print(f"manifest: {args.manifest}  ({len(val_samples):,} val perspectives)")

    trunk = load_frozen_trunk(args.checkpoint, device, value_head_variant=args.value_head)

    variants = args.variants if args.variants else list(VARIANTS.keys())
    print(f"variants: {variants}")
    print(f"trunk output dim: {H_PADDED}×{W_PADDED}")

    if args.mode == "overfit":
        run_overfit_mode(trunk, val_samples, variants, args, device, out_dir)
    elif args.mode == "probe":
        run_probe_mode(trunk, val_samples, variants, args, device, out_dir)


if __name__ == "__main__":
    main()
