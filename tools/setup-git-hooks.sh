#!/usr/bin/env bash
# Point git at the repo's tracked hooks under .githooks/.
# Idempotent — safe to re-run.
set -euo pipefail

cd "$(dirname "$0")/.."

git config core.hooksPath .githooks
echo "git core.hooksPath -> .githooks"
