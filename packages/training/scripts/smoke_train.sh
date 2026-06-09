#!/usr/bin/env bash
# Fresh-run smoke + loss fingerprint for the BC training loop.
#
# Runs a tiny fresh run — 2 epochs x 5 batches on the probe manifest, val
# skipped, single-process (num-workers 0) for determinism — and prints the
# per-batch loss fingerprint. The fingerprint is deterministic for unchanged
# training behavior, so a refactor's "no behavior change" claim is verified by
# diffing this output before vs. after the change.
#
# Arch + recipe come from the committed sibling smoke-config.json (its
# repo-relative data paths resolve because we cd to the repo root); the
# operational knobs stay CLI flags. Committing the config makes the
# fingerprint's inputs an auditable artifact.
#
# Output lands in a gitignored scratch run dir that is recreated each call.
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

smoke_dir="packages/training/data/runs/_smoke"
rm -rf "$smoke_dir"

# Run quietly (stdout -> /dev/null); the run still tees to its run.log, and
# errors surface on stderr. set -e aborts the script on a non-zero exit.
packages/training/scripts/run_bc_local.py \
  --config packages/training/scripts/smoke-config.json \
  --out-dir "$smoke_dir" \
  --max-batches 5 --skip-val --num-workers 0 \
  >/dev/null

run_dir="$(echo "$smoke_dir"/*/)"
uv run python - "$run_dir" <<'PY'
import json, sys, pathlib

run = pathlib.Path(sys.argv[1])
keys = ("epoch", "batch_idx", "batch_size", "policy", "value", "pass", "total", "n_non_pass")
for line in (run / "batches.jsonl").read_text().splitlines():
    r = json.loads(line)
    print("  ".join(f"{k}={r[k]}" for k in keys))
PY