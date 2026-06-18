"""Render a grid `results.json` artifact into a human-readable `report.md`.

One per-run table — `(population, encoding, model, seed)` keyed. Columns carry
both the *last* (stopped-epoch) state and the *best* (lowest-val_loss epoch) state,
so a late val-loss divergence stays visible rather than papered over:

  - `Train` / `Val` — final-epoch CE loss; `Gap` = `Val − Train` (the train-val gap,
    the overfit signal — positive means val worse than train).
  - `Best val` — lowest val_loss reached; `Best ep` / `Stop ep` — where that was and
    where early stop fired (`Stop ep < epochs` ⇒ it triggered).
  - `acc_full` / `acc_strict` — the two load-bearing accuracies at the final epoch.

Losses use `format_loss` (4dp under 1, 3dp above); accuracies render as percentages
via `format_pct`. Called from `run.py` after the JSON is written.
"""

from __future__ import annotations

from utils.format import format_count, format_loss, format_pct, md_table


def build_report(results: dict) -> str:
    """Markdown for one grid run: a provenance header + the per-run table."""
    cfg = results["config"]
    header = (
        f"`{cfg['manifest']}` [{cfg['split']}] · "
        f"{format_count(cfg['n_games'])} games · "
        f"{format_count(cfg['n_frames_post_filter'])} frames post-filter · "
        f"lr {cfg['lr']} · {cfg['epochs']} epochs (patience {cfg['patience']}) · "
        f"batch {cfg['batch_size']} · eps {cfg['eps']}"
    )

    headers = ["Population", "Encoding", "Model", "Seed", "Train", "Val", "Gap",
               "Best val", "Best ep", "Stop ep", "acc_full", "acc_strict"]
    align = ["left", "left", "left", *(["right"] * 9)]
    rows = []
    for r in results["cells"]:
        f, b = r["final"], r["best"]
        rows.append([
            r["population"], r["encoding"], r["model"], r["seed"],
            format_loss(f["train_loss"]), format_loss(f["val_loss"]),
            format_loss(f["val_loss"] - f["train_loss"]),
            format_loss(b["val_loss"]), r["best_epoch"], r["stopped_epoch"],
            format_pct(f["val_acc_full"]), format_pct(f["val_acc_strict"]),
        ])

    return f"# Argmin toy — grid results\n\n{header}\n\n{md_table(headers, rows, align=align)}\n"
