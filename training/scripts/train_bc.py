"""BC training loop scaffold.

Reads a split manifest produced by `bc.splits build`, walks the train split
via `IterableDataset`, runs N epochs of AdamW SGD with `bc_loss`. Prints
per-batch component losses + rolling samples/sec every `--log-every` batches;
end-of-epoch summary collects the sample-weighted means via `LossAccumulator`.

This is the loss-curve scaffold — no val pass (chunk 5), no JSONL logging
(chunk 3), no checkpoints (chunk 4). Pass `--max-batches N` to cap a run
for smoke testing.

Run from `training/`:
    uv run python scripts/train_bc.py \\
        --manifest data/splits/smoke.json \\
        --epochs 1 --batch-size 16 --max-batches 5
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bc.dataset import IterableDataset
from bc.loss import LossAccumulator, bc_loss
from bc.model import BCModel
from bc.splits import load_manifest, samples_for_split
from shared.device import disable_mps_fallback, move_batch, pick_device


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INTERMEDIATE = REPO_ROOT / "replay-parser" / "data" / "intermediate"


def main() -> None:
    # TODO: as the knob count grows past ~15, or when we start doing
    # cross-run sweeps, revisit moving to a config file (YAML/TOML). The
    # args.json dump (chunk 3) captures per-run config provenance for now.
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--intermediate", type=Path, default=DEFAULT_INTERMEDIATE)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help=(
            "Stop each epoch early after N batches — for smoke testing the "
            "loop end-to-end without committing to a full epoch's runtime."
        ),
    )
    args = parser.parse_args()

    disable_mps_fallback()

    if not args.manifest.exists():
        raise SystemExit(f"manifest not found: {args.manifest}")
    if not args.intermediate.exists():
        raise SystemExit(f"intermediate corpus not found: {args.intermediate}")

    device = pick_device(args.device)
    torch.manual_seed(args.seed)

    # --- Manifest + dataset ---
    print(f"loading manifest: {args.manifest}")
    manifest = load_manifest(args.manifest)
    train_samples = samples_for_split(manifest, "train", args.intermediate)
    print(
        f"  filter_version={manifest['filter_version']}  "
        f"git_sha={manifest['git_sha']}  "
        f"kept_pairs={manifest['kept_pairs']:,}  "
        f"train_pairs={len(train_samples):,}"
    )

    ds = IterableDataset(samples=train_samples, seed=args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size)

    # --- Model + optimizer ---
    print(f"building model on {device}")
    model = BCModel().to(device)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}")
    print()

    # --- Train loop ---
    model.train()
    for epoch in range(1, args.epochs + 1):
        acc = LossAccumulator()
        epoch_start = time.perf_counter()
        n_batches_seen = 0

        # Rolling samples/sec — instantaneous rate across the last log-every
        # window. Reset every print so the number tracks current throughput
        # rather than smoothing over the whole epoch.
        window_start = epoch_start
        window_samples = 0

        for batch_idx, batch in enumerate(loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

            batch = move_batch(batch, device)
            optim.zero_grad()
            out = model(batch["obs"])
            losses = bc_loss(out, batch)
            losses["total"].backward()
            optim.step()

            B = batch["obs"].shape[0]
            acc.update(losses, batch_size=B)
            window_samples += B
            n_batches_seen += 1

            if (batch_idx + 1) % args.log_every == 0:
                rate = window_samples / (time.perf_counter() - window_start)
                print(
                    f"[epoch {epoch}] batch {batch_idx + 1} | "
                    f"policy {losses['policy'].item():6.4f} "
                    f"value {losses['value'].item():6.4f} "
                    f"pass {losses['pass'].item():6.4f} "
                    f"total {losses['total'].item():6.4f} | "
                    f"{rate:.0f} samples/sec"
                )
                window_start = time.perf_counter()
                window_samples = 0

        epoch_dur = time.perf_counter() - epoch_start
        s = acc.summary()
        rate = s["n_samples"] / epoch_dur if epoch_dur > 0 else 0.0
        print()
        print(
            f"[epoch {epoch}] complete | "
            f"{s['n_samples']:,} frames ({s['n_non_pass']:,} non-pass) "
            f"in {epoch_dur:.1f}s | "
            f"{rate:.0f} samples/sec ({n_batches_seen} batches)"
        )
        print(
            f"[epoch {epoch}] mean: "
            f"policy {s['policy']:.4f}  "
            f"value {s['value']:.4f}  "
            f"pass {s['pass']:.4f}  |  "
            f"total {s['total']:.4f}"
        )
        print()


if __name__ == "__main__":
    main()
