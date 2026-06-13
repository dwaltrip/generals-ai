#!/usr/bin/env -S uv run python
"""Stratified val-loss analysis: per-frame head metrics from an offline val pass.

Two subcommands sharing one artifact format (record-building + IO live in
`training.bc.eval.dump`, shared with the in-training per-epoch dumps that
`TrainConfig.dump_val_frames` produces — `report` reads either producer's
artifact):

  dump    Load checkpoint(s) from a run dir, run one forward-only val pass
          each, and write per-frame records (value probs + CE, policy CE /
          entropy / top-k, pass prob, players_alive, frame provenance) to
          `<run>/analysis/stratified_val_epoch_NNN.npz` + a meta json.

  report  Read dumps and print the stratified read: the (p_start,
          players_alive) frame histogram, then a per-bucket table — frames,
          bucket-conditional floor, mean value CE, Δ vs floor, value
          prediction entropy (collapse check), policy CE / top-1. Bucket
          boundaries are report-time choices; nothing is baked into the dump.
          When the dump carries elim columns, also prints the elim head's
          per-true-bin, by-bucket, and self-vs-opponent tables (top-1 and
          prediction entropy are the tax-free reads; the CE-vs-floor margin is
          tax-confounded at τ>0).

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
from torch.utils.data import DataLoader

from settings import INTERMEDIATE_DIR
from training.analysis.marginal_entropy import N_CLASSES, entropy_nats
from training.analysis.run_metrics import resolve_manifest
from training.bc.checkpoint import load_bc_model
from training.bc.dataset import IterableDataset, assert_safe_loader
from training.bc.eval import FrameRecordCapture, dump_path, save_dump
from training.bc.model import BCModel
from training.bc.splits import load_manifest, samples_for_split
from training.shared.device import (
    dataloader_kwargs,
    disable_mps_fallback,
    move_batch,
    obs_for_model,
    pick_device,
)


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

    for epoch in epochs:
        ckpt = run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt"
        print(f"epoch {epoch}: {ckpt.name}", flush=True)
        model = load_bc_model(ckpt, device, value_head_variant=args.value_head_variant)
        # `persp_index_map` maps dataset-local sample_idx back to the position
        # in the full val list, so dumps with different fracs join on a stable id.
        records = _val_pass(model, samples, device, args).finalize(
            persp_index_map=chosen_arr
        )

        out = dump_path(run_dir, epoch)
        meta = {
            "run_dir": str(run_dir),
            "checkpoint": ckpt.name,
            "epoch": epoch,
            "sample_frac": args.sample_frac,
            "seed": args.seed,
            "n_perspectives": len(samples),
            "n_perspectives_total": len(all_samples),
            "device": device.type,
            "producer": "offline",
            "forward_dtype": "fp32",
        }
        if model.cfg.elim_head_enabled:
            # Edges label the per-bin report ranges; the report reads them from
            # meta (the npz carries only columns).
            meta["elim_bin_edges"] = list(model.cfg.elim_bin_edges)
        save_dump(records, out, meta)
        print(f"wrote {out} ({records['value_ce'].shape[0]} frames)")

        _check_against_epoch_row(records, run_dir, epoch, exact=args.sample_frac >= 1.0)


def _val_pass(
    model: BCModel,
    samples: list[tuple[Path, int]],
    device: torch.device,
    args: argparse.Namespace,
) -> FrameRecordCapture:
    """One forward-only pass; returns the filled per-frame capture."""
    ds = IterableDataset(
        samples=samples,
        seed=0,
        obs_cfg=model.cfg.obs,
        include_frame_info=True,
        # Switch on the elim targets iff the checkpoint's model has the head,
        # so the capture's elim columns get their `host_batch` targets. The
        # in-training producer threads edges from its train dataset; this is the
        # offline path's equivalent.
        elim_bin_edges=(
            model.cfg.elim_bin_edges if model.cfg.elim_head_enabled else None
        ),
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

    capture = FrameRecordCapture()
    start = time.perf_counter()
    last_report = start
    model.eval()
    with torch.no_grad():
        for batch in loader:
            now = time.perf_counter()
            if now - last_report >= 30.0:
                rate = capture.n_frames / (now - start)
                print(
                    f"  ... {capture.n_frames} frames, {rate:.0f} samples/sec",
                    flush=True,
                )
                last_report = now
            moved = move_batch(batch, device)
            out = model(obs_for_model(moved, None), moved["valid_mask"])
            capture.add_batch(batch, moved, out)

    dur = time.perf_counter() - start
    rate = capture.n_frames / dur if dur > 0 else 0.0
    print(
        f"  {capture.n_frames} frames in {dur:.1f}s ({rate:.0f} samples/sec)",
        flush=True,
    )
    return capture


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
    keys = ["value", "policy", "top1", "n_samples"]
    if "elim_ce" in records:
        alive = records["elim_alive_mask"]
        ours["elim"] = float(records["elim_ce"][alive].mean())
        keys.insert(3, "elim")
    label = "exact check" if exact else "approx check (subsampled)"
    print(f"  {label} vs epochs.jsonl:")
    for key in keys:
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
        if "elim_ce" in d:
            _print_elim_report(d, meta, by=args.by)


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
    # Dumps store the probs fp16; upcast so the entropy reduction below
    # accumulates fp32 (numpy keeps the input dtype through sum/mean).
    probs = d["value_probs"][sel].astype(np.float32)
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


# ---------------------------------------------------------------- elim report


def _elim_flat(
    d: dict[str, np.ndarray], sel: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten the elim columns to alive (player, frame) pairs within the
    frame selection `sel`. Returns (hard CE, true bin, probs [M, n_bins],
    channel index 0..7) — channel 0 is the perspective (self), 1..7 opponents."""
    mask = d["elim_alive_mask"][sel]                       # [n, 8] bool
    ce = d["elim_ce"][sel][mask]                           # [M]
    bins = d["elim_bin_target"][sel][mask]                 # [M]
    probs = d["elim_probs"][sel][mask].astype(np.float32)  # [M, n_bins]
    ch = np.broadcast_to(np.arange(8), mask.shape)[mask]   # [M]
    return ce, bins, probs, ch


def _elim_stats(
    ce: np.ndarray, bins: np.ndarray, probs: np.ndarray
) -> dict[str, float] | None:
    """Per-group elim metrics over flattened alive pairs. `top1` and `pred_h`
    are tax-free (argmax / softmax spread); `ce` is the tax-confounded hard CE."""
    n = int(ce.shape[0])
    if n == 0:
        return None
    return {
        "n": n,
        "ce": float(ce.mean()),
        "top1": float((probs.argmax(axis=1) == bins).mean()),
        "pred_h": float(-(probs * np.log(probs + 1e-12)).sum(axis=1).mean()),
    }


def _elim_bin_labels(meta: dict, n_bins: int) -> list[str]:
    """Human range labels for the bins, from the dump's `elim_bin_edges` meta;
    falls back to plain indices when edges are absent."""
    edges = meta.get("elim_bin_edges")
    if not edges:
        return [str(b) for b in range(n_bins)]
    labels, lo = [], 0
    for e in edges:
        labels.append(f"[{lo},{e})")
        lo = e
    labels.append(f"[>={lo}/never]")
    return labels


def _print_elim_per_bin(d: dict[str, np.ndarray], labels: list[str]) -> None:
    """Per-true-bin table — the which-bins-does-the-head-learn read. `top1`
    here is per-bin recall (argmax lands on the true bin), so a near-zero top1
    on the imminent bins (0–1) with healthy top1 on `never` is the lazy-head /
    class-imbalance signature the Stage-0 read flagged."""
    ce, bins, probs, _ = _elim_flat(d, np.ones(d["elim_ce"].shape[0], dtype=bool))
    n_bins = probs.shape[1]
    total = max(ce.shape[0], 1)
    print("Elim head — by true bin (alive player·frame pairs):\n")
    print("| bin | range | pairs | share | mean CE | top1 (recall) | pred H |")
    print("|---|---|---|---|---|---|---|")
    for b in range(n_bins):
        m = bins == b
        s = _elim_stats(ce[m], bins[m], probs[m])
        if s is None:
            print(f"| {b} | {labels[b]} | 0 | — | — | — | — |")
            continue
        print(
            f"| {b} | {labels[b]} | {s['n']:,} | {s['n'] / total:.1%} "
            f"| {s['ce']:.4f} | {s['top1']:.1%} | {s['pred_h']:.3f} |"
        )
    print()


def _print_elim_buckets(d: dict[str, np.ndarray], by: str) -> None:
    """Elim metrics stratified by `by` (players_alive / p_start). The cond floor
    is the bin-marginal entropy within the bucket — note the CE-vs-floor margin
    is tax-confounded at τ>0 (the tax-adjusted floor is a deferred tool); read
    top1 / pred H for the tax-free signal."""
    keys = d[by]
    print(
        f"Elim head — by `{by}` (cond floor = bin entropy within bucket; "
        "CE-vs-floor Δ tax-confounded at τ>0):\n"
    )
    print("| bucket | pairs | persp | cond floor | mean CE | Δ | top1 | pred H |")
    print("|---|---|---|---|---|---|---|---|")
    rows = [(f"{by}={v}", keys == v) for v in np.unique(keys)]
    rows.append(("all", np.ones(keys.shape[0], dtype=bool)))
    for label, sel in rows:
        ce, bins, probs, _ = _elim_flat(d, sel)
        s = _elim_stats(ce, bins, probs)
        if s is None:
            continue
        cond = max(0.0, entropy_nats(np.bincount(bins, minlength=probs.shape[1])))
        n_persp = int(np.unique(d["persp_val_index"][sel]).size)
        print(
            f"| {label} | {s['n']:,} | {n_persp:,} | {cond:.4f} | {s['ce']:.4f} "
            f"| {s['ce'] - cond:+.4f} | {s['top1']:.1%} | {s['pred_h']:.3f} |"
        )
    print()


def _print_elim_channel(d: dict[str, np.ndarray]) -> None:
    """Self (channel 0) vs opponents (1–7) — a distinct estimand split (Stage 0:
    self ~48% `never` vs opp ~19%). `% never` doubles as a channel-mapping
    sanity check against that corpus read."""
    ce, bins, probs, ch = _elim_flat(d, np.ones(d["elim_ce"].shape[0], dtype=bool))
    n_bins = probs.shape[1]
    print("Elim head — self (ch0) vs opponents (ch1–7):\n")
    print("| channel | pairs | mean CE | top1 | pred H | % never |")
    print("|---|---|---|---|---|---|")
    for label, m in (("self (ch0)", ch == 0), ("opp (ch1–7)", ch > 0)):
        s = _elim_stats(ce[m], bins[m], probs[m])
        if s is None:
            continue
        pct_never = float((bins[m] == n_bins - 1).mean())
        print(
            f"| {label} | {s['n']:,} | {s['ce']:.4f} | {s['top1']:.1%} "
            f"| {s['pred_h']:.3f} | {pct_never:.1%} |"
        )
    print()


def _print_elim_report(d: dict[str, np.ndarray], meta: dict, by: str) -> None:
    n_bins = d["elim_probs"].shape[-1]
    _print_elim_per_bin(d, _elim_bin_labels(meta, n_bins))
    _print_elim_buckets(d, by)
    _print_elim_channel(d)


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
