#!/usr/bin/env bash
# Fresh-run smoke + loss fingerprint for the BC training loop.
#
# Runs a tiny fresh run — 2 epochs x 5 batches on the probe manifest,
# single-process (num-workers 0) for determinism — and prints the per-batch
# loss fingerprint. The fingerprint is deterministic for unchanged training
# behavior, so a refactor's "no behavior change" claim is verified by diffing
# this output before vs. after the change.
#
# `--with-val` additionally runs the val pass (skipped by default for speed)
# and appends a per-epoch val-metrics fingerprint — the same diff guarantee for
# refactors that touch `run_val` / the val path, which the train-step
# fingerprint alone can't see.
#
# Arch + recipe come from the committed sibling smoke-config.json (its
# repo-relative data paths resolve because we cd to the repo root); the
# operational knobs stay CLI flags. Committing the config makes the
# fingerprint's inputs an auditable artifact.
#
# Output lands in a gitignored scratch run dir that is recreated each call.
set -euo pipefail

# `$skip_val` stays empty under --with-val so the run does a real val pass;
# unquoted expansion drops the empty token (bash 3.2 safe, no empty-array trap).
with_val=0
skip_val="--skip-val"
if [[ "${1:-}" == "--with-val" ]]; then
  with_val=1
  skip_val=""
fi

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

smoke_dir="data/training/runs/_smoke"
rm -rf "$smoke_dir"

# Run quietly (stdout -> /dev/null); the run still tees to its run.log, and
# errors surface on stderr. set -e aborts the script on a non-zero exit.
packages/training/scripts/run_bc_local.py \
  --config packages/training/scripts/smoke-config.json \
  --out-dir "$smoke_dir" \
  --max-batches 5 $skip_val --num-workers 0 \
  >/dev/null

run_dir="$(echo "$smoke_dir"/*/)"
uv run python - "$run_dir" "$with_val" <<'PY'
import json, sys, pathlib

run = pathlib.Path(sys.argv[1])
with_val = sys.argv[2] == "1"
keys = ("epoch", "batch_idx", "batch_size", "policy", "value", "pass", "total", "n_non_pass")
for line in (run / "batches.jsonl").read_text().splitlines():
    r = json.loads(line)
    print("  ".join(f"{k}={r[k]}" for k in keys))

if with_val:
    # Val metrics live under the `val` key of each epochs.jsonl row. Keep only
    # deterministic scalars: drop the timing fields (wall-clock, differ every
    # run) and the histogram dicts (long; redundant with the scalar metrics for
    # a fingerprint).
    print("--- val ---")
    skip = {"duration_sec", "samples_per_sec"}
    for line in (run / "epochs.jsonl").read_text().splitlines():
        r = json.loads(line)
        val = r.get("val") or {}
        cells = [f"epoch={r['epoch']}"]
        cells += [
            f"{k}={val[k]}"
            for k in sorted(val)
            if k not in skip and not isinstance(val[k], (list, dict))
        ]
        print("  ".join(cells))
PY
