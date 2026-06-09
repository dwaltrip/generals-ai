#!/usr/bin/env bash
# Width-sweep head-to-head eval: 0.5× vs 0.75× vs 1× (DeepNash-relative).
# 3 pairwise matchups, 80 games each (40 maps × swap-slots), temp=0.5.
# All checkpoints use pyramid value head (E3 from 2026-05-31 width sweep).

set -euo pipefail

export PYTHONUNBUFFERED=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

CKPT_05x="training/data/runs-cloud/2026-05-31T23-08-15Z/checkpoints/epoch_003.pt"
CKPT_075x="training/data/runs-cloud/2026-05-31T23-08-27Z/checkpoints/epoch_003.pt"
CKPT_1x="training/data/runs-cloud/2026-05-31T23-08-41Z/checkpoints/epoch_003.pt"

for ckpt in "$CKPT_05x" "$CKPT_075x" "$CKPT_1x"; do
  [ -f "$ckpt" ] || { echo "ERROR: checkpoint not found: $ckpt" >&2; exit 1; }
done

SWEEP_TS="$(date +%Y-%m-%dT%H-%M-%S)"
SWEEP_DIR="data/eval-runs/sweep-$SWEEP_TS"
mkdir -p "$SWEEP_DIR"
SUMMARY="$SWEEP_DIR/summary.log"
RUN_DIRS_FILE="$SWEEP_DIR/run_dirs.txt"
: > "$RUN_DIRS_FILE"

{
  echo "Width eval sweep started: $(date)"
  echo "Sweep dir:   $SWEEP_DIR"
  echo "0.5× ckpt:   $CKPT_05x"
  echo "0.75× ckpt:  $CKPT_075x"
  echo "1× ckpt:     $CKPT_1x"
} | tee "$SUMMARY"

TEMP=0.5
MAP_SEED=42
N_MAPS=40

spec() { echo "checkpoint:$1:temperature=$TEMP,sample,value_head_variant=pyramid"; }

run_matchup() {
  local label="$1"
  local spec0="$2"
  local spec1="$3"

  {
    echo ""
    echo "============================================================"
    echo "  $label  ($((N_MAPS * 2)) games, map_seed=$MAP_SEED)"
    echo "============================================================"
  } | tee -a "$SUMMARY"

  ./packages/eval-tools/scripts/run_adhoc.py \
    --policy "$spec0" \
    --policy "$spec1" \
    --maps "random:$N_MAPS" \
    --map-seed "$MAP_SEED" \
    --games-per-map 1 \
    --swap-slots 2>&1 | tee -a "$SUMMARY"

  local run_dir
  run_dir=$(grep "output:" "$SUMMARY" | tail -1 | sed 's/.*output: //')
  if [ -n "$run_dir" ]; then
    echo "$run_dir" >> "$RUN_DIRS_FILE"
  else
    echo "WARNING: could not extract run dir for matchup '$label'" >&2
  fi
}

# 1. 0.5× vs 0.75×
run_matchup "0.5x vs 0.75x @ temp=$TEMP" \
  "$(spec "$CKPT_05x")" "$(spec "$CKPT_075x")"

# 2. 0.5× vs 1×
run_matchup "0.5x vs 1x @ temp=$TEMP" \
  "$(spec "$CKPT_05x")" "$(spec "$CKPT_1x")"

# 3. 0.75× vs 1×
run_matchup "0.75x vs 1x @ temp=$TEMP" \
  "$(spec "$CKPT_075x")" "$(spec "$CKPT_1x")"

{
  echo ""
  echo "============================================================"
  echo "  Width eval sweep complete: $(date)"
  echo "  Master log:     $SUMMARY"
  echo "  Run dirs:       $RUN_DIRS_FILE"
  echo "  Per-matchup metrics in each run dir's games/*.metrics.json"
  echo "============================================================"
} | tee -a "$SUMMARY"
