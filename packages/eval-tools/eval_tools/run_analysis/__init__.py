"""Post-hoc analysis of finished eval runs.

Two-stage pipeline with flat row tables as the seam:

    loader   — eval-run dir → per-game records (npz arrays + results.jsonl join)
    extract  — per-game records → flat per-player-game and per-stage rows
    stats    — numpy-only aggregation helpers (bootstrap SE, sign test, percentiles)
    aggregate — rows → report-ready tables
    report   — tables → analysis/report.md, distributions.md, metrics.json, PNGs
    pipeline — drives the stages end to end (`analyze_run`), shared by the
               analyze_eval.py CLI and the eval runner

Everything downstream of extract reads only the row tables, never the npz.
"""
