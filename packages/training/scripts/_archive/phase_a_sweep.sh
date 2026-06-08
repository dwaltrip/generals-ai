#!/usr/bin/env bash
# Phase A FP16 batch-size sweep on T4 (cloud_24k slice).
# Kicks off 3 parallel detached Modal runs (bs=256/512/1024), logs to per-run files.
# Local CLI processes survive shell exit via nohup; remote functions survive via --detach.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

TIMESTAMP=$(date -u +%Y-%m-%dT%H-%M-%SZ)
LOGS_DIR="tmp/phase-a-logs/$TIMESTAMP"
mkdir -p "$LOGS_DIR"

BATCH_SIZES=(256 512 1024)

echo "Phase A FP16 bs sweep — $TIMESTAMP"
echo "Logs dir: $LOGS_DIR"
echo

for BS in "${BATCH_SIZES[@]}"; do
    LOG="$LOGS_DIR/bs${BS}.log"
    echo "  launching bs=$BS → $LOG"
    nohup uv run modal run --detach training/scripts/run_bc_modal.py \
        --modal-gpu T4 \
        --precision fp16 \
        --num-workers 4 \
        --pin-memory auto \
        --epochs 2 \
        --max-batches 100 \
        --batch-size "$BS" \
        --intermediate /data/intermediate-cloud-20k \
        --manifest /data/cloud_24k.json \
        > "$LOG" 2>&1 &
    disown
    sleep 5  # avoid run_id collision (make_run_id is second-resolution)
done

echo
echo "Launched 3 detached runs."
echo
echo "Tail all:  tail -f $LOGS_DIR/bs*.log"
echo "Tail one:  tail -f $LOGS_DIR/bs256.log"
echo
echo "Local CLI processes are nohup'd; remote functions are --detach'd."
echo "Close this shell anytime — runs continue. Find them later via 'modal app list' or the web UI."
