"""Plot training curves for a BC run.

Reads `epochs.jsonl` + `batches.jsonl` from a run dir and writes PNGs to
`<run_dir>/plots/`. No interactive display; safe to run headless.

Example usage:
    plot_run.py data/training/runs/20260521-005930
    plot_run.py <run_dir> --out <other_dir>

Plots produced:
  - per_batch_loss.png       4-panel (policy/value/pass/total), EMA + raw,
                             with epoch-boundary lines.
  - per_batch_throughput.png Per-batch wall_time_sec; reveals stalls and
                             the MPS warmup transient in epoch 1.
  - per_epoch_train_val.png  4-panel (policy/value/pass/total), train +
                             val series on the same axes.
  - per_epoch_topk.png       top-1 / top-3 / pass_acc trajectories.
  - per_epoch_action_dist.png Grouped bars: model vs target distribution
                             over the 8 (direction, split) buckets, per epoch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib


matplotlib.use("Agg")  # headless; must precede pyplot import
import matplotlib.pyplot as plt

from utils.docstring import doc_summary


LOSS_COMPONENTS = ("policy", "value", "pass", "total")
ACTION_BUCKETS = (
    "n_move",
    "n_split",
    "e_move",
    "e_split",
    "s_move",
    "s_split",
    "w_move",
    "w_split",
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def ema(xs: list[float], alpha: float) -> list[float]:
    # EMA recovers to steady state over ~1/alpha samples; α=0.01 → ~100-sample window.
    out: list[float] = []
    s = xs[0]
    for x in xs:
        s = alpha * x + (1 - alpha) * s
        out.append(s)
    return out


def epoch_boundaries(batches: list[dict]) -> list[int]:
    # Global batch index where each epoch *ends*. Vertical lines drawn at these x's.
    bounds: list[int] = []
    prev_epoch = batches[0]["epoch"]
    for i, b in enumerate(batches):
        if b["epoch"] != prev_epoch:
            bounds.append(i - 1)
            prev_epoch = b["epoch"]
    bounds.append(len(batches) - 1)
    return bounds


def plot_per_batch_loss(batches: list[dict], out: Path, alpha: float = 0.01) -> None:
    x = list(range(len(batches)))
    bounds = epoch_boundaries(batches)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, comp in zip(axes.flat, LOSS_COMPONENTS, strict=True):
        ys = [b[comp] for b in batches]
        ema_ys = ema(ys, alpha)
        ax.scatter(x, ys, s=1, alpha=0.08, color="C0", linewidths=0)
        ax.plot(x, ema_ys, color="C0", linewidth=1.5)
        for xb in bounds[:-1]:
            ax.axvline(xb, color="grey", linestyle=":", linewidth=0.7)
        # Clip y to the EMA range + headroom — raw outliers (per-batch spikes far
        # above the trend) would otherwise compress the readable signal. Faint
        # scatter above the limit gets clipped; the EMA always stays in frame.
        ax.set_ylim(0, max(ema_ys) * 1.25)
        ax.set_title(f"{comp} loss")
        ax.set_xlabel("global batch index")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Per-batch loss (EMA α={alpha}, raw scatter behind)")
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_per_batch_throughput(batches: list[dict], out: Path) -> None:
    # wall_time_sec is a run-wide cumulative timestamp; per-batch duration is its diff.
    # The first batch of each epoch (after the first) straddles val + checkpoint time,
    # so we drop those points to keep the curve about training step time alone.
    xs: list[int] = []
    ys: list[float] = []
    bounds = epoch_boundaries(batches)
    for i in range(1, len(batches)):
        if batches[i]["epoch"] != batches[i - 1]["epoch"]:
            continue
        xs.append(i)
        ys.append(batches[i]["wall_time_sec"] - batches[i - 1]["wall_time_sec"])

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.scatter(xs, ys, s=1, alpha=0.15, color="C1", linewidths=0)
    ax.plot(xs, ema(ys, 0.01), color="C1", linewidth=1.2)
    for xb in bounds[:-1]:
        ax.axvline(xb, color="grey", linestyle=":", linewidth=0.7)
    ax.set_title("Per-batch training step time (EMA α=0.01; epoch-boundary batches dropped)")
    ax.set_xlabel("global batch index")
    ax.set_ylabel("seconds per batch")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_per_epoch_train_val(epochs: list[dict], out: Path) -> None:
    xs = [e["epoch"] for e in epochs]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, comp in zip(axes.flat, LOSS_COMPONENTS, strict=True):
        ax.plot(xs, [e[comp] for e in epochs], marker="o", label="train", color="C0")
        ax.plot(xs, [e["val"][comp] for e in epochs], marker="o", label="val", color="C3")
        ax.set_title(f"{comp} loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_xticks(xs)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Per-epoch loss: train vs val")
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_per_epoch_topk(epochs: list[dict], out: Path) -> None:
    xs = [e["epoch"] for e in epochs]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, [e["val"]["top1"] for e in epochs], marker="o", label="top-1")
    ax.plot(xs, [e["val"]["top3"] for e in epochs], marker="o", label="top-3")
    ax.plot(xs, [e["val"]["pass_acc"] for e in epochs], marker="o", label="pass_acc")
    pass_frac = epochs[-1]["val"]["pass_frac"]
    # baseline for pass_acc if you predicted "never pass":
    ax.axhline(1 - pass_frac, color="grey", linestyle=":", linewidth=0.8,
               label=f"never-pass baseline ({1 - pass_frac:.3f})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_xticks(xs)
    ax.set_title("Val accuracy: top-k + pass head")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_per_epoch_action_dist(epochs: list[dict], out: Path) -> None:
    n_epochs = len(epochs)
    n_buckets = len(ACTION_BUCKETS)
    fig, axes = plt.subplots(n_epochs, 1, figsize=(11, 1.8 * n_epochs), sharex=True)
    if n_epochs == 1:
        axes = [axes]

    bar_w = 0.4
    positions = list(range(n_buckets))
    for ax, e in zip(axes, epochs, strict=True):
        model = [e["val"]["action_dist"][k] for k in ACTION_BUCKETS]
        target = [e["val"]["action_target_dist"][k] for k in ACTION_BUCKETS]
        ax.bar([p - bar_w / 2 for p in positions], model, bar_w, label="model", color="C0")
        ax.bar([p + bar_w / 2 for p in positions], target, bar_w, label="target", color="C3")
        ax.set_ylabel(f"epoch {e['epoch']}")
        ax.set_xticks(positions)
        ax.set_xticklabels(ACTION_BUCKETS, rotation=30, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
        if e is epochs[0]:
            ax.legend(loc="upper right")
    fig.suptitle("Val action distribution: model vs target")
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=doc_summary(__doc__))
    p.add_argument("run_dir", type=Path, help="Path to training run dir")
    p.add_argument("--out", type=Path, default=None,
                   help="Output dir for PNGs (default: <run_dir>/plots/)")
    p.add_argument("--ema-alpha", type=float, default=0.01,
                   help="EMA smoothing factor for per-batch plots")
    args = p.parse_args()

    run_dir: Path = args.run_dir
    out_dir: Path = args.out or run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = load_jsonl(run_dir / "epochs.jsonl")
    batches = load_jsonl(run_dir / "batches.jsonl")
    print(f"loaded {len(epochs)} epochs, {len(batches)} batches from {run_dir}")

    plot_per_batch_loss(batches, out_dir / "per_batch_loss.png", alpha=args.ema_alpha)
    plot_per_batch_throughput(batches, out_dir / "per_batch_throughput.png")
    plot_per_epoch_train_val(epochs, out_dir / "per_epoch_train_val.png")
    plot_per_epoch_topk(epochs, out_dir / "per_epoch_topk.png")
    plot_per_epoch_action_dist(epochs, out_dir / "per_epoch_action_dist.png")
    print(f"wrote 5 plots to {out_dir}")


if __name__ == "__main__":
    main()
