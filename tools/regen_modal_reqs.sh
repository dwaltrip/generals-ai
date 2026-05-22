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

regen() {
    local package="$1"
    local out="$2"
    echo "regenerating $out (package: $package)"
    uv export \
        --package "$package" \
        --format requirements-txt \
        --no-dev \
        --no-emit-project \
        -o "$out"
}

regen cloud-train-poc cloud-poc/modal_requirements.txt
regen training training/modal_requirements.txt
