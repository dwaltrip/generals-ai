"""Render a grid `results.json` artifact into a human-readable `report.md`.

One per-run table — `(target, population, encoding, model, seed)` keyed. Columns:

  - `Train` / `Val` — final-epoch CE loss; `Gap` = `Val − Train` (the overfit signal,
    positive means val worse than train).
  - `acc_full` / `acc_inset` — the two load-bearing accuracies on the *all* slice.
  - `acc_surr` / `ceil_surr` — final accuracy on the surrender subpopulation against
    its none-ceiling reference (6.18-4 §5/§8): a `none`/`alive` cell floored at
    `ceil_surr` reads "at ceiling," not "broken." Empty slices render `—`.
  - `Best ep` / `Stop ep` — lowest-val_loss epoch and where early stop fired.

Losses use `format_loss` (4dp under 1, 3dp above); accuracies render as percentages
via `format_pct` (NaN → `—`). Called from `run.py` after the JSON is written.
"""

from __future__ import annotations

from utils.format import format_count, format_loss, format_pct, md_table


def build_report(results: dict) -> str:
    """Markdown for one grid run: a provenance header + the per-run table."""
    cfg = results["config"]
    header = (
        f"`{cfg['manifest']}` [{cfg['split']}] · slice `{cfg['slice']}` · "
        f"{format_count(cfg['n_games'])} games · "
        f"{format_count(cfg['n_frames'])} frames · "
        f"lr {cfg['lr']} · {cfg['epochs']} epochs (patience {cfg['patience']}) · "
        f"batch {cfg['batch_size']} · eps {cfg['eps']}"
    )

    headers = ["Target", "Pop", "Encoding", "Model", "Seed", "Train", "Val", "Gap",
               "Best ep", "Stop ep", "acc_full", "acc_inset", "acc_surr", "ceil_surr"]
    align = ["left", "left", "left", "left", *(["right"] * 10)]
    rows = []
    for r in results["cells"]:
        f = r["final"]
        rows.append([
            r["target"], r["population"], r["encoding"], r["model"], r["seed"],
            format_loss(f["train_loss"]), format_loss(f["val_loss"]),
            format_loss(f["val_loss"] - f["train_loss"]),
            r["best_epoch"], r["stopped_epoch"],
            format_pct(f["val_acc_full"]), format_pct(f["val_acc_inset"]),
            format_pct(f.get("val_acc_full_surr")),
            format_pct(r["none_ceilings"]["surr"]),
        ])

    return f"# Argmin toy — grid results\n\n{header}\n\n{md_table(headers, rows, align=align)}\n"
