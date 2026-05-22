#!/usr/bin/env bash
# Regenerate <member>/modal_requirements.txt files from the workspace uv.lock.
#
# Each output is a fully-pinned requirements file (direct + transitive deps)
# suitable for Modal's `uv_pip_install(requirements=[...])`. The single source
# of truth is the workspace uv.lock; this script just exports a per-package
# slice of it.
#
# Re-run after changing any dependency that ships to Modal.

set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p tmp
LOG="tmp/regen_modal_reqs.log"
echo "generated at: $(date '+%Y-%m-%d %H:%M:%S %Z')" > "$LOG"

regen() {
    local package="$1"
    local out="$2"
    echo "regenerating $out (package: $package)"
    echo "=== $package ===" >> "$LOG"
    uv export \
        --package "$package" \
        --format requirements-txt \
        --no-dev \
        --no-emit-project \
        --no-emit-workspace \
        -o "$out" \
        >> "$LOG"
}

regen cloud-train-poc cloud-poc/modal_requirements.txt
regen training training/modal_requirements.txt

echo "(uv export stdout captured in $LOG)"
