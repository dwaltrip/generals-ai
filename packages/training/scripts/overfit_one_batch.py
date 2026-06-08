"""Single-batch overfit harness — Phase 2 exit criterion.

Pulls one fixed batch off the real corpus, then trains on it repeatedly
for N steps. If the loss drops to near-zero, gradients are flowing
correctly end-to-end (obs → trunk → heads → loss → backward). If it
doesn't, the bug is local enough to debug here, not in a corpus run.

Run from repo root:
    uv run python training/scripts/overfit_one_batch.py
    uv run python training/scripts/overfit_one_batch.py --steps 1000 --batch-size 8
    uv run python training/scripts/overfit_one_batch.py --device cpu

The script unsets `PYTORCH_ENABLE_MPS_FALLBACK` so unsupported MPS ops
error loudly rather than silently degrading to CPU. If you hit a fallback
error, the first debug step is `--device cpu` — that distinguishes
algorithmic from MPS-coverage bugs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from training.bc.dataset import IterableDataset
from training.bc.filters import eligible_perspectives
from training.bc.loss import bc_loss
from training.bc.model import BCModel
from training.bc.model_config import build_model_cfg
from training.bc.obs_config import OBS_CONFIG_DEFAULTS
from training.bc.splits import load_curated_names
from training.bc.utils import list_sim_paths, meta_path_for
from training.shared.device import disable_mps_fallback, move_batch, pick_device


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INTERMEDIATE = REPO_ROOT / "replay-parser" / "data" / "intermediate"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--intermediate", type=Path, default=DEFAULT_INTERMEDIATE)
    parser.add_argument(
        "--min-non-pass-frac",
        type=float,
        default=0.25,
        help=(
            "Search the loader until we find a batch where at least this "
            "fraction of frames are non-pass. The dataloader's first batches "
            "are from early-game turns (mostly pass while armies build); we "
            "skip past them so the policy head gets a real gradient signal."
        ),
    )
    parser.add_argument(
        "--max-batch-search",
        type=int,
        default=100,
        help="Give up if no mixed-enough batch is found within this many tries.",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=1000,
        help=(
            "Number of sim files to scan for eligible (game, perspective) pairs. "
            "The overfit harness only needs one batch; we cap scan cost rather "
            "than building a full-corpus manifest just to pull a single batch."
        ),
    )
    parser.add_argument(
        "--value-head",
        choices=("direct", "pyramid"),
        default="direct",
        help="Value-head variant to construct.",
    )
    args = parser.parse_args()

    # Unset MPS fallback so unsupported ops fail loudly (see module docstring).
    disable_mps_fallback()

    if not args.intermediate.exists():
        raise SystemExit(f"intermediate corpus not found at {args.intermediate}")

    device = pick_device(args.device)
    torch.manual_seed(args.seed)

    # --- Pull one fixed batch ---
    print(f"building dataset (intermediate={args.intermediate})...")
    curated_names = load_curated_names()
    sim_paths = list_sim_paths(args.intermediate)[: args.subset_size]
    samples: list[tuple[Path, int]] = []
    for sim_path in sim_paths:
        for k in eligible_perspectives(sim_path, meta_path_for(sim_path), curated_names):
            samples.append((sim_path, k))
    print(f"scanned {len(sim_paths):,} games -> {len(samples):,} eligible (game, k) pairs")
    ds = IterableDataset(samples=samples, seed=args.seed, obs_cfg=OBS_CONFIG_DEFAULTS)
    loader = DataLoader(ds, batch_size=args.batch_size)
    min_non_pass = int(args.batch_size * args.min_non_pass_frac)
    print(
        f"searching for a batch of {args.batch_size} with "
        f">= {min_non_pass} non-pass frames..."
    )
    batch = None
    batch_idx = -1
    for i, candidate in enumerate(loader):
        n_pass_i = int(candidate["is_pass"].sum().item())
        n_non_pass_i = args.batch_size - n_pass_i
        if n_non_pass_i >= min_non_pass:
            batch = candidate
            batch_idx = i
            break
        if i >= args.max_batch_search:
            raise SystemExit(
                f"no mixed batch found in {args.max_batch_search} tries; "
                "sanity check the dataloader (pass rate seems pathologically high)"
            )
    assert batch is not None
    batch = move_batch(batch, device)
    n_pass = int(batch["is_pass"].sum().item())
    print(
        f"using batch {batch_idx}: obs={tuple(batch['obs'].shape)} "
        f"({n_pass}/{args.batch_size} pass frames)"
    )

    # --- Model + optimizer ---
    print(f"building model on {device}...")
    model = BCModel(build_model_cfg(value_head_variant=args.value_head)).to(device)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}")
    print()

    # --- Overfit loop ---
    model.train()
    print(
        f"{'step':>6} {'total':>10} {'policy':>10} "
        f"{'value':>10} {'pass':>10} {'sec':>8}"
    )
    print("-" * 60)
    t_start = time.perf_counter()
    last_loss = None
    for step in range(args.steps + 1):
        optim.zero_grad()
        # fp32 model, no autocast → upcast the (fp16-default) obs to match.
        out = model(batch["obs"].float(), batch["valid_mask"])
        losses = bc_loss(out, batch)
        losses["total"].backward()
        optim.step()
        last_loss = losses

        if step % args.log_every == 0 or step == args.steps:
            t_elapsed = time.perf_counter() - t_start
            print(
                f"{step:>6} "
                f"{losses['total'].item():>10.4f} "
                f"{losses['policy'].item():>10.4f} "
                f"{losses['value'].item():>10.4f} "
                f"{losses['pass'].item():>10.4f} "
                f"{t_elapsed:>8.1f}"
            )
    print("-" * 60)

    assert last_loss is not None
    total = last_loss["total"].item()
    print()
    if total < 0.1:
        print(f"OVERFIT SUCCESS: final total loss = {total:.4f}")
    else:
        print(f"OVERFIT INCOMPLETE: final total loss = {total:.4f}")
        print("(threshold is informal; check the curve above — if losses are")
        print(" still trending down, try --steps with a larger value)")


if __name__ == "__main__":
    main()
