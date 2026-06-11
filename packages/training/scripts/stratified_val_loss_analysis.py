#!/usr/bin/env -S uv run python
"""Stratified val-loss analysis: per-frame head metrics from an offline val pass.

Two subcommands sharing one artifact format:

  dump    Load checkpoint(s) from a run dir, run one forward-only val pass
          each, and write per-frame records (value probs + CE, policy CE /
          entropy / top-k, pass prob, players_alive, frame provenance) to
          `<run>/analysis/stratified_val_epoch_NNN.npz` + a meta json.

  report  Read dumps and print the stratified read: the (p_start,
          players_alive) frame histogram, then a per-bucket table — frames,
          bucket-conditional floor, mean value CE, Δ vs floor, value
          prediction entropy (collapse check), policy CE / top-1. Bucket
          boundaries are report-time choices; nothing is baked into the dump.

The conditional floor is the entropy of the placement distribution *within*
the bucket — a stronger baseline than the global marginal, since it credits
the model only for signal beyond what bucket membership itself implies
(e.g. 2-alive ⇒ placement ∈ {1st, 2nd}).

`--sample-frac` subsamples at the perspective level (whole trajectories,
seeded) so dumps stay comparable across checkpoints of the same run. With
frac = 1.0, `dump` cross-checks its aggregates against the run's
`epochs.jsonl` row — the end-to-end correctness check for this harness.

Usage:
    stratified_val_loss_analysis.py dump RUN_DIR [--epochs all|best|2,4]
        [--sample-frac 0.25] [--device auto] [--num-workers 4]
    stratified_val_loss_analysis.py report RUN_DIR [--epochs all|best|2,4]
        [--by players_alive|p_start]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from settings import INTERMEDIATE_DIR
from training.analysis.marginal_entropy import N_CLASSES, entropy_nats
from training.analysis.run_metrics import resolve_manifest
from training.bc.checkpoint import load_bc_model
from training.bc.dataset import IterableDataset, assert_safe_loader
from training.bc.loss import flatten_policy_logits
from training.bc.model import BCModel
from training.bc.splits import load_manifest, samples_for_split
from training.shared.device import (
    dataloader_kwargs,
    disable_mps_fallback,
    move_batch,
    obs_for_model,
    pick_device,
)


def dump_path(run_dir: Path, epoch: int) -> Path:
    return run_dir / "analysis" / f"stratified_val_epoch_{epoch:03d}.npz"


def load_epoch_rows(run_dir: Path) -> list[dict]:
    path = run_dir / "epochs.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def resolve_epochs(arg: str, run_dir: Path) -> list[int]:
    """`all` → every checkpoint on disk; `best` → argmin val.value in
    epochs.jsonl; otherwise a comma list of epoch numbers."""
    if arg == "all":
        ckpts = sorted((run_dir / "checkpoints").glob("epoch_*.pt"))
        return [int(p.stem.split("_")[1]) for p in ckpts]
    if arg == "best":
        rows = [r for r in load_epoch_rows(run_dir) if (r.get("val") or {}).get("value")]
        if not rows:
            raise SystemExit("--epochs best needs val rows in epochs.jsonl")
        return [min(rows, key=lambda r: r["val"]["value"])["epoch"]]
    return [int(tok) for tok in arg.split(",")]


def resolve_val_samples(run_dir: Path) -> list[tuple[Path, int]]:
    """The run's val split as (sim_path, k) pairs, via the same manifest
    resolution the floor display uses (recorded path, basename fallback)."""
    args = json.loads((run_dir / "args.json").read_text())
    manifest_path = resolve_manifest(args["manifest"])
    if manifest_path is None:
        raise SystemExit(f"cannot resolve manifest {args['manifest']!r} locally")
    recorded = args.get("intermediate")
    intermediate = (
        Path(recorded) if recorded and Path(recorded).is_dir() else INTERMEDIATE_DIR
    )
    return samples_for_split(load_manifest(manifest_path), "val", intermediate)


# ---------------------------------------------------------------- dump


def run_dump(args: argparse.Namespace) -> None:
    run_dir: Path = args.run_dir
    epochs = resolve_epochs(args.epochs, run_dir)
    device = pick_device(args.device)
    disable_mps_fallback()

    all_samples = resolve_val_samples(run_dir)
    if args.sample_frac < 1.0:
        rng = random.Random(args.seed)
        n_keep = max(1, round(args.sample_frac * len(all_samples)))
        chosen = sorted(rng.sample(range(len(all_samples)), n_keep))
    else:
        chosen = list(range(len(all_samples)))
    samples = [all_samples[i] for i in chosen]
    chosen_arr = np.asarray(chosen, dtype=np.int64)
    print(
        f"val perspectives: {len(samples)}/{len(all_samples)} "
        f"(frac={args.sample_frac}, seed={args.seed}), device={device.type}",
        flush=True,
    )

    out_dir = run_dir / "analysis"
    out_dir.mkdir(exist_ok=True)

    for epoch in epochs:
        ckpt = run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt"
        print(f"epoch {epoch}: {ckpt.name}", flush=True)
        model = load_bc_model(ckpt, device, value_head_variant=args.value_head_variant)
        records = _val_pass(model, samples, device, args)
        # Map dataset-local sample_idx back to the position in the full val
        # list, so dumps with different fracs join on a stable id.
        records["persp_val_index"] = chosen_arr[records.pop("sample_idx")]

        out = dump_path(run_dir, epoch)
        np.savez_compressed(out, allow_pickle=False, **records)
        meta = {
            "run_dir": str(run_dir),
            "checkpoint": ckpt.name,
            "epoch": epoch,
            "sample_frac": args.sample_frac,
            "seed": args.seed,
            "n_perspectives": len(samples),
            "n_perspectives_total": len(all_samples),
            "n_frames": int(records["value_ce"].shape[0]),
            "device": device.type,
        }
        out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"wrote {out} ({meta['n_frames']} frames)")

        _check_against_epoch_row(records, run_dir, epoch, exact=args.sample_frac >= 1.0)


def _val_pass(
    model: BCModel,
    samples: list[tuple[Path, int]],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    """One forward-only pass; returns per-frame columns, concatenated."""
    ds = IterableDataset(
        samples=samples,
        seed=0,
        obs_cfg=model.cfg.obs,
        include_frame_info=True,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        **dataloader_kwargs(
            num_workers=args.num_workers,
            pin_memory=None,
            prefetch_factor=2,
            device=device,
        ),
    )
    assert_safe_loader(loader)

    cols: dict[str, list[np.ndarray]] = {}

    def push(name: str, arr: torch.Tensor) -> None:
        cols.setdefault(name, []).append(arr.cpu().numpy())

    start = time.perf_counter()
    n_frames = 0
    last_report = start
    model.eval()
    with torch.no_grad():
        for batch in loader:
            now = time.perf_counter()
            if now - last_report >= 30.0:
                rate = n_frames / (now - start)
                print(
                    f"  ... {n_frames} frames, {rate:.0f} samples/sec",
                    flush=True,
                )
                last_report = now
            # Frame-info / target scalars are consumed host-side; grab them
            # before the device move.
            for key in ("frame_t", "players_alive", "p_start", "sample_idx"):
                push(key, batch[key])
            push("placement", batch["value_target"])
            push("is_pass", batch["is_pass"])

            moved = move_batch(batch, device)
            out = model(obs_for_model(moved, None), moved["valid_mask"])

            # Value head: full predicted distribution + per-frame CE.
            value_logp = F.log_softmax(out["value_logits"].float(), dim=1)
            value_ce = -value_logp.gather(
                1, moved["value_target"].unsqueeze(1)
            ).squeeze(1)
            push("value_probs", value_logp.exp())
            push("value_ce", value_ce)

            # Policy head, on the flat masked layout (same as bc_loss).
            # NOTE: no bool-mask gathers on the device tensors here — that op
            # returns garbage indices on MPS (see TODO(mps-val-crash) in
            # bc/eval/run.py). Equality/topk/gather are fine; pass-frame columns
            # are NaN'd out host-side instead.
            masked = flatten_policy_logits(out["policy_logits"].float(), moved["mask"])
            pol_logp = F.log_softmax(masked, dim=1)
            target = moved["action_target"]
            pol_ce = -pol_logp.gather(1, target.clamp(min=0).unsqueeze(1)).squeeze(1)
            is_pass_dev = moved["is_pass"]
            pol_ce = torch.where(
                is_pass_dev, torch.full_like(pol_ce, float("nan")), pol_ce
            )
            push("policy_ce", pol_ce)

            # Entropy over the masked softmax. MASK_NEG keeps illegal logits
            # finite, so p·log p underflows to 0 there — no NaN guard needed.
            pol_p = pol_logp.exp()
            push("policy_entropy", -(pol_p * pol_logp).sum(dim=1))
            push("n_legal", moved["mask"].reshape(masked.shape[0], -1).sum(dim=1))

            topk = torch.topk(masked, k=3, dim=1).indices
            non_pass = ~is_pass_dev
            push("top1", (topk[:, 0] == target) & non_pass)
            push("top3", (target.unsqueeze(1) == topk).any(dim=1) & non_pass)
            push("pass_prob", torch.sigmoid(out["pass_logit"].float()))

            n_frames += batch["is_pass"].shape[0]

    dur = time.perf_counter() - start
    rate = n_frames / dur if dur > 0 else 0.0
    print(f"  {n_frames} frames in {dur:.1f}s ({rate:.0f} samples/sec)", flush=True)
    return {name: np.concatenate(parts) for name, parts in cols.items()}


def _check_against_epoch_row(
    records: dict[str, np.ndarray], run_dir: Path, epoch: int, exact: bool
) -> None:
    """Compare dump aggregates to the run's recorded val metrics. With a full
    pass (`exact`) these should agree to float-accumulation noise; subsampled
    passes get a labeled approximate comparison."""
    row = next(
        (r for r in load_epoch_rows(run_dir) if r.get("epoch") == epoch), None
    )
    val = (row or {}).get("val")
    if not val:
        print("  (no val row in epochs.jsonl to check against)")
        return
    non_pass = ~records["is_pass"]
    ours = {
        "value": float(records["value_ce"].mean()),
        "policy": float(np.nanmean(records["policy_ce"])),
        "top1": float(records["top1"].sum() / max(non_pass.sum(), 1)),
        "n_samples": int(records["is_pass"].shape[0]),
    }
    label = "exact check" if exact else "approx check (subsampled)"
    print(f"  {label} vs epochs.jsonl:")
    for key in ("value", "policy", "top1", "n_samples"):
        theirs = val.get(key)
        if theirs is None:
            continue
        diff = ours[key] - theirs
        print(f"    {key:9s} dump={ours[key]:.6g} recorded={theirs:.6g} diff={diff:+.2e}")


# ---------------------------------------------------------------- report


def run_report(args: argparse.Namespace) -> None:
    run_dir: Path = args.run_dir
    if args.epochs == "all":
        paths = sorted((run_dir / "analysis").glob("stratified_val_epoch_*.npz"))
        if not paths:
            raise SystemExit(f"no dumps under {run_dir / 'analysis'} — run dump first")
        epochs = [int(p.stem.split("_")[-1]) for p in paths]
    else:
        epochs = resolve_epochs(args.epochs, run_dir)

    for i, epoch in enumerate(epochs):
        path = dump_path(run_dir, epoch)
        if not path.is_file():
            raise SystemExit(f"missing dump: {path}")
        d = dict(np.load(path))
        meta = json.loads(path.with_suffix(".meta.json").read_text())
        print(f"\n## {run_dir.name} — epoch {epoch}")
        print(
            f"{meta['n_frames']} frames, {meta['n_perspectives']}/"
            f"{meta['n_perspectives_total']} val perspectives "
            f"(frac={meta['sample_frac']})\n"
        )
        if i == 0:
            _print_histogram(d)
        _print_bucket_table(d, by=args.by)


def _print_histogram(d: dict[str, np.ndarray]) -> None:
    """Frame counts by (p_start, players_alive) — the raw material for
    choosing bucket boundaries."""
    p_starts = np.unique(d["p_start"])
    alive_vals = np.unique(d["players_alive"])
    print("Frames by (p_start ↓, players_alive →):\n")
    header = "| p_start | " + " | ".join(str(a) for a in alive_vals) + " |"
    print(header)
    print("|" + "---|" * (len(alive_vals) + 1))
    for ps in p_starts:
        sel = d["p_start"] == ps
        counts = [int(((d["players_alive"] == a) & sel).sum()) for a in alive_vals]
        print(f"| {ps} | " + " | ".join(f"{c:,}" for c in counts) + " |")
    print()


def _bucket_row(d: dict[str, np.ndarray], sel: np.ndarray) -> dict[str, float] | None:
    n = int(sel.sum())
    if n == 0:
        return None
    placement = d["placement"][sel]
    cond_floor = max(0.0, entropy_nats(np.bincount(placement, minlength=N_CLASSES)))
    value_ce = float(d["value_ce"][sel].mean())
    probs = d["value_probs"][sel]
    pred_entropy = float(-(probs * np.log(probs + 1e-12)).sum(axis=1).mean())
    argmax = probs.argmax(axis=1)
    mode_share = float((argmax == np.bincount(argmax).argmax()).mean())
    non_pass = sel & ~d["is_pass"]
    n_np = int(non_pass.sum())
    return {
        "n_frames": n,
        "n_persp": int(np.unique(d["persp_val_index"][sel]).size),
        "cond_floor": cond_floor,
        "value_ce": value_ce,
        "delta": value_ce - cond_floor,
        "pred_entropy": pred_entropy,
        "argmax_mode_share": mode_share,
        "policy_ce": float(np.nanmean(d["policy_ce"][non_pass])) if n_np else float("nan"),
        "top1": float(d["top1"][non_pass].sum() / n_np) if n_np else float("nan"),
    }


def _print_bucket_table(d: dict[str, np.ndarray], by: str) -> None:
    keys = d[by]
    print(f"Buckets by `{by}` (cond. floor = placement entropy within bucket):\n")
    print(
        "| bucket | frames | persp | cond floor | value CE | Δ | pred H | "
        "argmax mode | policy CE | top1 |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    rows = [(f"{by}={val}", keys == val) for val in np.unique(keys)]
    rows.append(("all", np.ones(keys.shape[0], dtype=bool)))
    for label, sel in rows:
        r = _bucket_row(d, sel)
        if r is None:
            continue
        print(
            f"| {label} | {r['n_frames']:,} | {r['n_persp']:,} "
            f"| {r['cond_floor']:.4f} | {r['value_ce']:.4f} | {r['delta']:+.4f} "
            f"| {r['pred_entropy']:.4f} | {r['argmax_mode_share']:.0%} "
            f"| {r['policy_ce']:.4f} | {r['top1']:.1%} |"
        )
    print()


# ---------------------------------------------------------------- CLI


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dump = sub.add_parser("dump", help="run val pass(es), write per-frame npz")
    p_dump.add_argument("run_dir", type=Path)
    p_dump.add_argument("--epochs", default="all", help="all | best | comma list")
    p_dump.add_argument("--sample-frac", type=float, default=1.0)
    p_dump.add_argument("--seed", type=int, default=0)
    p_dump.add_argument("--batch-size", type=int, default=256)
    p_dump.add_argument("--num-workers", type=int, default=4)
    p_dump.add_argument("--device", default="auto")
    p_dump.add_argument(
        "--value-head-variant", default="direct",
        help="legacy (pre-arch) checkpoints only; ignored for arch-bearing ones",
    )
    p_dump.set_defaults(fn=run_dump)

    p_rep = sub.add_parser("report", help="print stratified tables from dumps")
    p_rep.add_argument("run_dir", type=Path)
    p_rep.add_argument("--epochs", default="all", help="all | best | comma list")
    p_rep.add_argument(
        "--by", default="players_alive", choices=("players_alive", "p_start"),
    )
    p_rep.set_defaults(fn=run_report)

    args = parser.parse_args()
    if not args.run_dir.is_dir():
        parser.error(f"not a directory: {args.run_dir}")
    args.fn(args)


if __name__ == "__main__":
    main()
