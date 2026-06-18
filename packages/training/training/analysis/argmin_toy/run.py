#!/usr/bin/env -S uv run python
"""The argmin-toy driver: build the fq table once, then run an experiment's grid.

`main` is thin (arg-parse + tee logging + dispatch); `run_experiments` carries the
body — optionally surrender-filter, then for each target `prep_target` once and run
its cells. The MVP-vs-surrender differences are *data* (an `Experiment` config), not
control flow. Three experiments:

  - `anchor` — just A1 / all-alive, the sanity cell that must recover `W ≈ −I` at
    ~100% strict-margin before any grid means anything (6.18-1 §11 step 7).
  - `mvp`    — the original optimization-difficulty grid (filtered binary population):
    all-alive (4 models) + mixed (3 encodings × 4 models), the notch study (6.18-3).
  - `surrender` — the status-sufficiency grid (6.18-4 §6), on the full unfiltered
    population, run once per target (`alive` / `army_pos`) and sliced by surrender
    status in the metrics. `alive`: 4 encodings × spine; `army_pos`: 2 × spine.

Outputs a structured JSON (per-cell curves + final metrics + `.inspect()` dumps +
error-vs-margin + the none-ceiling reference) and a streamed log (tee'd to console).

    ./run.py --slice anchor                          # the sanity cell, ~seconds
    ./run.py --slice surrender                        # the status grid, on probe_500
    ./run.py --slice surrender --manifest <cloud_24k_sub_5k.json>   # the final read
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
    prep_target,
    split_by_game,
    surrender_filter,
    to_tensors,
)
from training.analysis.argmin_toy.encode import EncoderCfg
from training.analysis.argmin_toy.metrics import (
    error_vs_margin,
    make_metrics_fn,
    none_ceilings,
    tie_stats,
    weight_report,
)
from training.analysis.argmin_toy.models import A1, B1, D1, E1, ArgminModel
from training.analysis.argmin_toy.report import build_report
from training.analysis.argmin_toy.train import HParams, Roles, train_argmin_model
from training.analysis.families import REGISTRY  # imports families first → registers ARGMIN_TOY
from training.analysis.fq.frame_table import (
    GROUND_TRUTH_OBS_CFG,
    FrameTable,
    build_frame_table,
)
from training.bc.splits import load_manifest, samples_for_split


# Encodings whose dead-handling is a non-monotonic notch (need depth-1 to draw it):
# `none`/`capture_status` leave eliminated players at army 0, `neg1_sentinel` at −1.
NOTCH_ENCODINGS = {"none", "capture_status", "neg1_sentinel"}
NOTCH_CAPABLE = {"B1", "E1_mlp"}   # depth-1 models that can draw the notch
SEED_FUND = (0, 1, 2)              # ≥3 seeds on notch-encoding × notch-capable (Risk #3)
PHI_HIDDEN = 32                    # MLP-φ width for E1 on the notch encodings


@dataclass(frozen=True)
class Cell:
    target: str           # "alive" | "army_pos"
    population: str        # "all_alive" | "mixed" | "full"
    encoding: str          # "none" | "capture_status" | "boolean" | "categorical" | "neg1_sentinel"
    model: str             # "A1" | "B1" | "D1" | "E1_linear" | "E1_mlp"
    encoder_cfg: EncoderCfg
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class Experiment:
    name: str
    filter_surrender: bool   # MVP/anchor train on the filtered binary population
    cells: tuple[Cell, ...]


def _e1_flavor(encoding: str) -> str:
    """E1 uses a linear φ where the argmin is linear-solvable (boolean/categorical,
    all-alive army-only) and an MLP φ where the encoding needs the notch."""
    return "E1_mlp" if encoding in NOTCH_ENCODINGS else "E1_linear"


def _seeds(encoding: str, model: str) -> tuple[int, ...]:
    return SEED_FUND if (encoding in NOTCH_ENCODINGS and model in NOTCH_CAPABLE) else (0,)


def _spine(target: str, population: str, encoding: str) -> list[Cell]:
    """The A–E spine for one (target, population, encoding): A1 / B1 / D1 / E1, E1's
    φ flavored by whether the encoding needs the notch."""
    cfg = EncoderCfg(encoding, "army_only")  # type: ignore[arg-type]
    models = ("A1", "B1", "D1", _e1_flavor(encoding))
    return [Cell(target, population, encoding, m, cfg, _seeds(encoding, m)) for m in models]


def anchor_cells() -> list[Cell]:
    return [Cell("alive", "all_alive", "none", "A1", EncoderCfg("none", "army_only"), (0,))]


def mvp_cells() -> list[Cell]:
    """The optimization-difficulty grid (6.18-3), in the new taxonomy: all-alive
    (identity, the W≈−I oracle) + mixed across the three MVP encodings."""
    cells: list[Cell] = []
    aa_cfg = EncoderCfg("none", "army_only")
    for model in ("A1", "B1", "D1", "E1_linear"):
        cells.append(Cell("alive", "all_alive", "none", model, aa_cfg, (0,)))
    for encoding in ("boolean", "none", "neg1_sentinel"):
        cells.extend(_spine("alive", "mixed", encoding))
    return cells


def surrender_cells() -> list[Cell]:
    """The status-sufficiency grid (6.18-4 §6), full population, per target. `boolean`/
    `categorical` are pruned under `army_pos` (status is army-derivable there)."""
    cells: list[Cell] = []
    for encoding in ("none", "capture_status", "boolean", "categorical"):
        cells.extend(_spine("alive", "full", encoding))
    for encoding in ("none", "capture_status"):
        cells.extend(_spine("army_pos", "full", encoding))
    return cells


EXPERIMENTS = {
    "anchor": Experiment("anchor", True, tuple(anchor_cells())),
    "mvp": Experiment("mvp", True, tuple(mvp_cells())),
    "surrender": Experiment("surrender", False, tuple(surrender_cells())),
}


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


def _population(t: FrameTable, name: str) -> FrameTable:
    if name == "all_alive":
        return all_alive(t)
    if name == "mixed":
        return mixed(t)
    if name == "full":
        return t
    raise ValueError(f"unknown population {name!r}")


def run_cell(
    cell: Cell, prepped: FrameTable, hp: HParams, roles: Roles, eps: float,
    val_frac: float, device: torch.device,
) -> list[dict]:
    """Train every seed of one cell on a table already `prep_target`-ed for
    `cell.target`; return a per-seed result record list."""
    pop = _population(prepped, cell.population)
    train_t, val_t = split_by_game(pop, val_frac, seed=0)  # split fixed across seeds
    train_view, val_view = to_tensors(train_t), to_tensors(val_t)
    ceilings = none_ceilings(val_t, cell.target)
    metrics_fn = make_metrics_fn(eps)

    records: list[dict] = []
    for seed in cell.seeds:
        print(f"\n=== {cell.target} / {cell.population} / {cell.encoding} / {cell.model} "
              f"/ seed {seed} (C={cell.encoder_cfg.n_channels}, "
              f"{train_view['label'].shape[0]} train / {val_view['label'].shape[0]} val) ===")
        model = build_model(cell.model, cell.encoder_cfg, seed)
        res = train_argmin_model(
            model, train_view, val_view, roles, metrics_fn, hp, seed=seed, device=device
        )
        curves = res.curves
        scores = _val_scores(model, val_view, roles, device)  # for the diagnostics
        aux = {k: val_view[k] for k in roles.aux}
        records.append({
            "target": cell.target,
            "population": cell.population,
            "encoding": cell.encoding,
            "model": cell.model,
            "seed": seed,
            "n_channels": cell.encoder_cfg.n_channels,
            "n_train": int(train_view["label"].shape[0]),
            "n_val": int(val_view["label"].shape[0]),
            "n_params": sum(p.numel() for p in model.parameters()),
            "none_ceilings": ceilings,
            "curves": curves,
            "final": {k: v[-1] for k, v in curves.items()},          # last (stopped) epoch
            "best": {k: v[res.best_epoch] for k, v in curves.items()},  # lowest-val_loss epoch
            "best_epoch": res.best_epoch,
            "stopped_epoch": res.stopped_epoch,
            "weights": weight_report(model.inspect()),
            "error_vs_margin": error_vs_margin(scores, val_view[roles.target], aux),
            "tie_stats": tie_stats(aux),
        })
    return records


def run_experiments(
    exp: Experiment, table: FrameTable, hp: HParams, roles: Roles, eps: float,
    val_frac: float, device: torch.device,
) -> tuple[list[dict], FrameTable]:
    """Run one experiment's grid; return the per-cell records and the effective
    (post-filter) table. `prep_target` runs once per target (cells grouped by it)."""
    if exp.filter_surrender:
        n_pre = table.frame_t.size
        table = surrender_filter(table)
        print(f"  {n_pre} frames -> {table.frame_t.size} after surrender filter "
              f"({table.n_games} games)")
    else:
        print(f"  {table.frame_t.size} frames, no filter ({table.n_games} games)")

    records: list[dict] = []
    for target in dict.fromkeys(c.target for c in exp.cells):  # ordered-unique
        prepped = prep_target(table, target)
        for cell in (c for c in exp.cells if c.target == target):
            records.extend(run_cell(cell, prepped, hp, roles, eps, val_frac, device))
    return records, table


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
    ap.add_argument("--slice", default="anchor", choices=tuple(EXPERIMENTS))
    ap.add_argument("--manifest", type=Path,
                    default=TRAINING_DATA_DIR / "splits" / "probe_500.json")
    ap.add_argument("--split", default="val", choices=("train", "val"))
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=HParams.epochs)
    ap.add_argument("--lr", type=float, default=HParams.lr)
    ap.add_argument("--batch-size", type=int, default=HParams.batch_size)
    ap.add_argument("--patience", type=int, default=HParams.patience,
                    help="early-stop after N epochs of flat val_loss (<=0 disables)")
    ap.add_argument("--min-delta", type=float, default=HParams.min_delta)
    ap.add_argument("--min-epochs", type=int, default=HParams.min_epochs)
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
        hp = HParams(lr=args.lr, batch_size=args.batch_size, epochs=args.epochs,
                     patience=args.patience, min_delta=args.min_delta,
                     min_epochs=args.min_epochs)
        roles = Roles()
        exp = EXPERIMENTS[args.slice]

        man = load_manifest(args.manifest)
        samples = samples_for_split(man, args.split, Path(man["intermediate_root"]))
        print(f"building argmin_toy table [{args.split}] from {args.manifest.name} ...")
        table = build_frame_table(REGISTRY["argmin_toy"], samples, GROUND_TRUTH_OBS_CFG,
                                  args.max_games)
        print(f"experiment '{exp.name}': {len(exp.cells)} cells, "
              f"{sum(len(c.seeds) for c in exp.cells)} runs")

        results, eff_table = run_experiments(exp, table, hp, roles, args.eps,
                                             args.val_frac, device)

        artifact = {
            "config": {
                "slice": args.slice, "manifest": args.manifest.name, "split": args.split,
                "max_games": args.max_games, "epochs": hp.epochs, "lr": hp.lr,
                "batch_size": hp.batch_size, "val_frac": args.val_frac, "eps": args.eps,
                "patience": hp.patience, "min_delta": hp.min_delta,
                "min_epochs": hp.min_epochs,
                "n_frames": int(eff_table.frame_t.size), "n_games": eff_table.n_games,
            },
            "cells": results,
        }
        (out_dir / "results.json").write_text(json.dumps(artifact, indent=2))
        (out_dir / "report.md").write_text(build_report(artifact))

        print(f"\n{'='*78}\nsummary  (out_dir: {out_dir})")
        print(f"  {'cell':<52} {'acc_full':>9} {'surr':>7} {'ceil':>7}")
        for r in results:
            name = (f"{r['target']}/{r['population']}/{r['encoding']}/"
                    f"{r['model']}/s{r['seed']}")
            f = r["final"]
            print(f"  {name:<52} {f.get('val_acc_full', float('nan')):>9.3f} "
                  f"{f.get('val_acc_full_surr', float('nan')):>7.3f} "
                  f"{r['none_ceilings']['surr']:>7.3f}")
    finally:
        sys.stdout = orig_stdout
        log_f.close()


if __name__ == "__main__":
    main()
