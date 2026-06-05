#!/usr/bin/env bash
# Recommended eval sweep: 200 games over 5 matchups (~50 min MPS).
#
#   1. pyramid vs direct  @ temp=0.8   (80 games)  primary win-rate question
#   2. pyramid vs direct  @ temp=0.5   (40 games)  temp sensitivity
#   3. pyramid vs direct  @ temp=1.0   (40 games)  temp sensitivity
#   4. direct  vs direct  @ temp=0.8   (20 games)  noise floor (should ~50/50)
#   5. pyramid vs pyramid @ temp=0.8   (20 games)  noise floor (should ~50/50)
#
# Each matchup lands in its own data/eval-runs/<timestamp>/ dir (full
# metrics.json + per-tick diagnostics per game). This script writes a
# master log + pointer file to data/eval-runs/sweep-<timestamp>/.

set -euo pipefail

# Force run_adhoc.py's stdout to stay line-buffered when piped through `tee`.
# Without this, per-game progress lines are held in Python's block buffer
# (~8 KB) and only flush at end of each matchup.
export PYTHONUNBUFFERED=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

DIRECT_CKPT="training/data/runs-cloud/2026-05-27T21-39-06Z/checkpoints/epoch_005.pt"
PYRAMID_CKPT="training/data/runs-cloud/2026-05-27T21-50-45Z/checkpoints/epoch_005.pt"

for ckpt in "$DIRECT_CKPT" "$PYRAMID_CKPT"; do
  [ -f "$ckpt" ] || { echo "ERROR: checkpoint not found: $ckpt" >&2; exit 1; }
done

SWEEP_TS="$(date +%Y-%m-%dT%H-%M-%S)"
SWEEP_DIR="data/eval-runs/sweep-$SWEEP_TS"
mkdir -p "$SWEEP_DIR"
SUMMARY="$SWEEP_DIR/summary.log"
RUN_DIRS_FILE="$SWEEP_DIR/run_dirs.txt"
: > "$RUN_DIRS_FILE"  # truncate; per-matchup paths appended by run_matchup

{
  echo "Sweep started: $(date)"
  echo "Sweep dir:     $SWEEP_DIR"
  echo "Direct ckpt:   $DIRECT_CKPT"
  echo "Pyramid ckpt:  $PYRAMID_CKPT"
} | tee "$SUMMARY"

direct_spec()  { echo "checkpoint:$DIRECT_CKPT:temperature=$1,sample"; }
pyramid_spec() { echo "checkpoint:$PYRAMID_CKPT:temperature=$1,sample,value_head_variant=pyramid"; }

run_matchup() {
  local label="$1"
  local spec0="$2"
  local spec1="$3"
  local n_maps="$4"
  local map_seed="$5"

  {
    echo ""
    echo "============================================================"
    echo "  $label  ($((n_maps * 2)) games, map_seed=$map_seed)"
    echo "============================================================"
  } | tee -a "$SUMMARY"

  ./packages/eval-tools/scripts/run_adhoc.py \
    --policy "$spec0" \
    --policy "$spec1" \
    --maps "random:$n_maps" \
    --map-seed "$map_seed" \
    --games-per-map 1 \
    --swap-slots 2>&1 | tee -a "$SUMMARY"

  # run_adhoc.py prints exactly one `output: <dir>` line per invocation
  # near the top. Pull the most recent one out of SUMMARY and append to
  # the sweep's run_dirs pointer file.
  local run_dir
  run_dir=$(grep "output:" "$SUMMARY" | tail -1 | sed 's/.*output: //')
  if [ -n "$run_dir" ]; then
    echo "$run_dir" >> "$RUN_DIRS_FILE"
  else
    echo "WARNING: could not extract run dir for matchup '$label'" >&2
  fi
}

# 1. Primary matchup (80 games)
run_matchup "pyramid vs direct @ temp=0.8" \
  "$(direct_spec 0.8)" "$(pyramid_spec 0.8)" 40 801

# 2-3. Temperature sweep (40 games each)
run_matchup "pyramid vs direct @ temp=0.5" \
  "$(direct_spec 0.5)" "$(pyramid_spec 0.5)" 20 501

run_matchup "pyramid vs direct @ temp=1.0" \
  "$(direct_spec 1.0)" "$(pyramid_spec 1.0)" 20 1001

# 4-5. Noise-floor baselines (20 games each)
run_matchup "direct vs direct @ temp=0.8 (noise floor)" \
  "$(direct_spec 0.8)" "$(direct_spec 0.8)" 10 2001

run_matchup "pyramid vs pyramid @ temp=0.8 (noise floor)" \
  "$(pyramid_spec 0.8)" "$(pyramid_spec 0.8)" 10 2002

{
  echo ""
  echo "============================================================"
  echo "  Sweep complete: $(date)"
  echo "  Master log:     $SUMMARY"
  echo "  Run dirs:       $RUN_DIRS_FILE"
  echo "  Per-matchup metrics in each run dir's games/*.metrics.json"
  echo "============================================================"
} | tee -a "$SUMMARY"
