"""One-off probe experiments built on the `probes/` framework.

Each module here is a self-contained, runnable probe: it contributes a
`ProbeTask` (target + heads + loss + metrics + baselines) and a thin `main` that
calls `probes.cli.run_probe_cli`. Disposable/experiment-specific by design — the
reusable machinery stays in the sibling `probes/` package.
"""
