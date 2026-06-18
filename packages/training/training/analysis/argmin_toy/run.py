#!/usr/bin/env -S uv run python
"""The argmin-toy driver: build the fq table once, then loop the grid.

Build → surrender-filter → for each cell: pick the population, `to_tensors` +
`split_by_game`, train the scorer, eval, and `.inspect()` the weights. Two slices:

  - `anchor` — just A1 / all-alive, the sanity cell that must recover `W ≈ −I` at
    ~100% strict-margin before the grid means anything (6.18-1 §11 step 7).
  - `full` — the 16-cell grid (6.18-1 §7): all-alive (1 encoding × 4 models) +
    mixed (3 encodings × 4 models). The overload encodings × notch-capable models
    (B1, MLP-φ E1) run ≥3 seeds so a depth-1 failure isn't a single unlucky seed
    (Risk #3); one seed elsewhere.

Outputs a structured JSON (per-cell curves + final metrics + `.inspect()` dumps +
error-vs-margin) and a streamed human-readable log (tee'd to console + file).

    ./run.py --slice anchor                         # the sanity cell, ~seconds
    ./run.py --slice full                           # the grid, on probe_500
    ./run.py --slice full --manifest <cloud_24k_sub_5k.json>   # the final read
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import sys

import torch

from settings import TRAINING_DATA_DIR
from training.analysis.argmin_toy.data import (
    all_alive,
    mixed,
    split_by_game,
    surrender_filter,
    to_tensors,
)
from training.analysis.argmin_toy.encode import EncoderCfg
from training.analysis.argmin_toy.metrics import (
    error_vs_margin,
    make_metrics_fn,
    tie_stats,
    weight_report,
)
from training.analysis.argmin_toy.models import A1, B1, D1, E1, ArgminModel
from training.analysis.argmin_toy.train import HParams, Roles, train_argmin_model
from training.analysis.families import REGISTRY  # imports families first → registers ARGMIN_TOY
from training.analysis.fq.frame_table import (
    GROUND_TRUTH_OBS_CFG,
    FrameTable,
    build_frame_table,
)
from training.bc.splits import load_manifest, samples_for_split


OVERLOAD = {"zero_overload", "neg1_sentinel"}  # encodings whose dead-handling is a notch
NOTCH_CAPABLE = {"B1", "E1_mlp"}               # depth-1 models that can draw the notch
SEED_FUND = (0, 1, 2)                          # ≥3 seeds on overload × notch (Risk #3)
PHI_HIDDEN = 32                                # MLP-φ width for E1 on the notch encodings


@dataclass(frozen=True)
class Cell:
    population: str       # "all_alive" | "mixed"
    encoding: str         # "army_only" | "explicit_mask" | "zero_overload" | "neg1_sentinel"
    model: str            # "A1" | "B1" | "D1" | "E1_linear" | "E1_mlp"
    encoder_cfg: EncoderCfg
    seeds: tuple[int, ...]


def _e1_flavor(encoding: str) -> str:
    """E1 uses a linear φ where the argmin is linear-solvable (explicit_mask,
    all-alive army-only) and an MLP φ where the encoding needs the notch."""
    return "E1_mlp" if encoding in OVERLOAD else "E1_linear"


def _seeds(encoding: str, model: str) -> tuple[int, ...]:
    return SEED_FUND if (encoding in OVERLOAD and model in NOTCH_CAPABLE) else (0,)


def grid_cells(slice_name: str) -> list[Cell]:
    """Enumerate the cells for a slice. The all-alive encodings collapse to one
    (army-only, identity), so it carries 4 models; mixed runs 3 encodings × 4."""
    if slice_name == "anchor":
        return [Cell("all_alive", "army_only", "A1",
                     EncoderCfg("zero_overload", "army_only"), (0,))]

    cells: list[Cell] = []
    # all-alive: identity army-only encoding (no dead to encode), the W≈−I oracle.
    aa_cfg = EncoderCfg("zero_overload", "army_only")
    for model in ("A1", "B1", "D1", "E1_linear"):
        cells.append(Cell("all_alive", "army_only", model, aa_cfg, (0,)))

    # mixed: where the encoding bites.
    for encoding in ("explicit_mask", "zero_overload", "neg1_sentinel"):
        cfg = EncoderCfg(encoding, "army_only")  # type: ignore[arg-type]
        models = ("A1", "B1", "D1", _e1_flavor(encoding))
        for model in models:
            cells.append(Cell("mixed", encoding, model, cfg, _seeds(encoding, model)))
    return cells


def build_model(model: str, encoder_cfg: EncoderCfg, seed: int) -> ArgminModel:
    """Build a freshly-seeded `ArgminModel` for `(model, encoder_cfg)`. Seeding
    before construction makes each variant's init reproducible (mirrors run_probe)."""
    torch.manual_seed(seed)
    c = encoder_cfg.n_channels
    scorer: torch.nn.Module
    if model == "A1":
        scorer = A1(c)
    elif model == "B1":
        scorer = B1(c)
    elif model == "D1":
        scorer = D1(c)
    elif model == "E1_linear":
        scorer = E1(c, phi_hidden=None)
    elif model == "E1_mlp":
        scorer = E1(c, phi_hidden=PHI_HIDDEN)
    else:
        raise ValueError(f"unknown model {model!r}")
    return ArgminModel(encoder_cfg, scorer)


def run_cell(
    cell: Cell, table: FrameTable, hp: HParams, roles: Roles, eps: float,
    val_frac: float, device: torch.device,
) -> list[dict]:
    """Train every seed of one cell; return a per-seed result record list."""
    pop = all_alive(table) if cell.population == "all_alive" else mixed(table)
    train_t, val_t = split_by_game(pop, val_frac, seed=0)  # split fixed across seeds
    train_view, val_view = to_tensors(train_t), to_tensors(val_t)
    metrics_fn = make_metrics_fn(eps)

    records: list[dict] = []
    for seed in cell.seeds:
        print(f"\n=== {cell.population} / {cell.encoding} / {cell.model} / seed {seed} "
              f"(C={cell.encoder_cfg.n_channels}, "
              f"{train_view['label'].shape[0]} train / {val_view['label'].shape[0]} val) ===")
        model = build_model(cell.model, cell.encoder_cfg, seed)
        curves = train_argmin_model(
            model, train_view, val_view, roles, metrics_fn, hp, seed=seed, device=device
        )
        scores = _val_scores(model, val_view, roles, device)  # for the diagnostics
        aux = {k: val_view[k] for k in roles.aux}
        records.append({
            "population": cell.population,
            "encoding": cell.encoding,
            "model": cell.model,
            "seed": seed,
            "n_channels": cell.encoder_cfg.n_channels,
            "n_train": int(train_view["label"].shape[0]),
            "n_val": int(val_view["label"].shape[0]),
            "n_params": sum(p.numel() for p in model.parameters()),
            "curves": curves,
            "final": {k: v[-1] for k, v in curves.items()},
            "weights": weight_report(model.inspect()),
            "error_vs_margin": error_vs_margin(scores, val_view[roles.target], aux),
            "tie_stats": tie_stats(aux),
        })
    return records


@torch.no_grad()
def _val_scores(model: ArgminModel, val_view: dict, roles: Roles,
                device: torch.device) -> torch.Tensor:
    model.eval()
    inputs = [val_view[name].to(device) for name in roles.inputs]
    return model(*inputs).cpu()


class _Tee:
    """Mirror writes to several streams, so the per-epoch log lands in both the
    console and the run log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s: str) -> int:
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self) -> None:
        for st in self.streams:
            st.flush()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--slice", default="anchor", choices=("anchor", "full"))
    ap.add_argument("--manifest", type=Path,
                    default=TRAINING_DATA_DIR / "splits" / "probe_500.json")
    ap.add_argument("--split", default="val", choices=("train", "val"))
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=HParams.epochs)
    ap.add_argument("--lr", type=float, default=HParams.lr)
    ap.add_argument("--batch-size", type=int, default=HParams.batch_size)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--eps", type=float, default=0.0, help="strict-margin threshold (raw army)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", type=Path, default=TRAINING_DATA_DIR / "argmin_toy")
    args = ap.parse_args()

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out_dir = args.out_dir / f"{stamp}-{args.slice}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = (out_dir / "run.log").open("w")
    orig_stdout = sys.stdout
    sys.stdout = _Tee(orig_stdout, log_f)  # type: ignore[assignment]

    try:
        device = torch.device(args.device)
        hp = HParams(lr=args.lr, batch_size=args.batch_size, epochs=args.epochs)
        roles = Roles()

        man = load_manifest(args.manifest)
        samples = samples_for_split(man, args.split, Path(man["intermediate_root"]))
        print(f"building argmin_toy table [{args.split}] from {args.manifest.name} ...")
        table = build_frame_table(REGISTRY["argmin_toy"], samples, GROUND_TRUTH_OBS_CFG,
                                  args.max_games)
        n_pre = table.frame_t.size
        table = surrender_filter(table)
        print(f"  {n_pre} frames -> {table.frame_t.size} after surrender filter "
              f"({table.n_games} games)")

        cells = grid_cells(args.slice)
        print(f"slice '{args.slice}': {len(cells)} cells, "
              f"{sum(len(c.seeds) for c in cells)} runs")

        results: list[dict] = []
        for cell in cells:
            results.extend(run_cell(cell, table, hp, roles, args.eps,
                                    args.val_frac, device))

        artifact = {
            "config": {
                "slice": args.slice, "manifest": args.manifest.name, "split": args.split,
                "max_games": args.max_games, "epochs": hp.epochs, "lr": hp.lr,
                "batch_size": hp.batch_size, "val_frac": args.val_frac, "eps": args.eps,
                "n_frames_post_filter": int(table.frame_t.size), "n_games": table.n_games,
            },
            "cells": results,
        }
        (out_dir / "results.json").write_text(json.dumps(artifact, indent=2))

        print(f"\n{'='*70}\nsummary  (out_dir: {out_dir})")
        print(f"  {'cell':<46} {'acc_strict':>10} {'acc_inset':>10}")
        for r in results:
            name = f"{r['population']}/{r['encoding']}/{r['model']}/s{r['seed']}"
            print(f"  {name:<46} {r['final'].get('val_acc_strict', float('nan')):>10.3f} "
                  f"{r['final'].get('val_acc_inset', float('nan')):>10.3f}")
    finally:
        sys.stdout = orig_stdout
        log_f.close()


if __name__ == "__main__":
    main()
