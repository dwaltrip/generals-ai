#!/usr/bin/env bash
# Width sweep — fan out 4 BC training runs (one per trunk width) on Modal, hold
# everything else fixed, and read the resulting loss curves across widths.
#
# Dated one-shot experiment: the width set is baked in; the recipe knobs are
# flags (defaults below). Design + rationale: docs/2026-05/5.31-3-width-sweep-plan.md.
#
# Launch phase only. Head-to-head eval is deferred — this is the cheap initial
# screen (final val_policy + pass_frac across widths). Note 5.28-1: loss is a
# poor proxy for skill, so a promising/ambiguous result graduates to eval before
# any big-run width decision.
#
# Jobs run detached on Modal and train in parallel (~30-35 min at the defaults).
# Each cell's run id is recorded to the dated sweep dir for the pull/compare step
# printed at the end. Re-running relaunches all four cells — mind the double-spend.

set -euo pipefail

# --- recipe knobs (defaults = the locked sweep recipe; override to tweak) ----
EPOCHS=2
MAX_BATCHES=800
BATCH_SIZE=512                           # value-head class-diversity floor (5.28-1)
GPU="A100"                               # uniform card — removes GPU as a confound
SEED=0
VALUE_HEAD="pyramid"                     # the head-A/B winner; what the big run uses
MANIFEST="/data/cloud_24k.json"          # already on the parsed-replays Volume
INTERMEDIATE="/data/intermediate-cloud-20k"

usage() {
  cat <<EOF
Width sweep launcher — fan out 4 BC training runs (0.5x/1x/1.5x/2x) on Modal.

Recipe knobs (defaults are the locked sweep recipe; override to tweak):
  --epochs N         (default $EPOCHS)
  --max-batches N    (default $MAX_BATCHES)
  --batch-size N     (default $BATCH_SIZE)
  --gpu CLASS        (default $GPU)
  --seed N           (default $SEED)
  --value-head V     direct|pyramid (default $VALUE_HEAD)
  -h, --help

Example:  ./training/scripts/width_sweep_2026-05-31.sh --epochs 3 --max-batches 600
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --epochs)      EPOCHS="$2"; shift 2 ;;
    --max-batches) MAX_BATCHES="$2"; shift 2 ;;
    --batch-size)  BATCH_SIZE="$2"; shift 2 ;;
    --gpu)         GPU="$2"; shift 2 ;;
    --seed)        SEED="$2"; shift 2 ;;
    --value-head)  VALUE_HEAD="$2"; shift 2 ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

export PYTHONUNBUFFERED=1  # keep modal/local-entrypoint output live through tee

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# label:outer:middle:inner — 0.5x / 1x / 1.5x / 2x of the 128/128/160 trunk.
# 0.5x is the regime control: if 0.5x ~= 1x we're not capacity-bound, and a flat
# high end reads as "under-budget" rather than "width doesn't help".
WIDTHS=(
  "0.5x:64:64:80"
  "1x:128:128:160"
  "1.5x:192:192:240"
  "2x:256:256:320"
)

SWEEP_DIR="data/width-sweep-2026-05-31"
CONFIG_DIR="$SWEEP_DIR/configs"
RUN_IDS_FILE="$SWEEP_DIR/run_ids.txt"
LAUNCH_LOG="$SWEEP_DIR/launch.log"
mkdir -p "$CONFIG_DIR"
: > "$RUN_IDS_FILE"   # truncate; per-cell `label run_id` lines appended below

{
  echo "Width sweep launch — $(date)"
  echo "Sweep dir: $SWEEP_DIR"
  echo "Recipe:    bs=$BATCH_SIZE epochs=$EPOCHS max_batches=$MAX_BATCHES gpu=$GPU head=$VALUE_HEAD seed=$SEED"
  echo "Slice:     $MANIFEST"
  echo
} | tee "$LAUNCH_LOG"

for entry in "${WIDTHS[@]}"; do
  IFS=":" read -r label outer middle inner <<< "$entry"
  config="$CONFIG_DIR/$label.json"

  # The whole cell config — only the arch block differs across cells. Partial
  # over defaults: lr/weight_decay/shuffle_buffer_size/precision stay default.
  cat > "$config" <<JSON
{
  "manifest": "$MANIFEST",
  "intermediate": "$INTERMEDIATE",
  "batch_size": $BATCH_SIZE,
  "epochs": $EPOCHS,
  "seed": $SEED,
  "gpu": "$GPU",
  "arch": {
    "outer_width": $outer,
    "middle_width": $middle,
    "inner_width": $inner,
    "value_head_variant": "$VALUE_HEAD"
  }
}
JSON

  {
    echo "============================================================"
    echo "  launching $label  ($outer/$middle/$inner, $GPU)"
    echo "============================================================"
  } | tee -a "$LAUNCH_LOG"

  uv run modal run --detach training/scripts/run_bc_modal.py \
    --config "$config" --max-batches "$MAX_BATCHES" 2>&1 | tee -a "$LAUNCH_LOG"

  # run_bc_modal prints `run_dir: /runs/<run_id>`; pull the latest one back out.
  run_id=$(grep "run_dir:" "$LAUNCH_LOG" | tail -1 | sed 's#.*/##' || true)
  if [ -n "$run_id" ]; then
    echo "$label $run_id" >> "$RUN_IDS_FILE"
  else
    echo "WARNING: could not capture run_id for $label" >&2
  fi
  echo | tee -a "$LAUNCH_LOG"
done

echo "All cells launched. Run ids ($RUN_IDS_FILE):"
cat "$RUN_IDS_FILE"
echo
echo "Once the jobs finish (~30-35 min on Modal, parallel), pull each run dir:"
while read -r label run_id; do
  echo "  uv run modal volume get generals-ai.training-runs /$run_id training/data/runs-cloud   # $label"
done < "$RUN_IDS_FILE"
echo
echo "Then compare across widths (epoch-$EPOCHS val_policy + pass_frac in epochs.jsonl;"
echo "train curves via training/scripts/plot_run.py). Promising/ambiguous -> graduate to eval."
