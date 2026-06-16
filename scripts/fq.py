#!/usr/bin/env -S uv run python
"""fq CLI entry: hand the analysis-families registry to fq's generic CLI.

    ./scripts/fq.py --table elim_head_debug --split val \
        -e "t.cols['army_sim'][t.cols['alive']].mean()"

    ./scripts/fq.py --table elim_head_debug <<'PY'
    imm = select(t, (t.cols['dt'] >= 0) & (t.cols['dt'] < 10))
    print(imm.n_games, np.median(imm.cols['margin'][imm.cols['margin'] >= 0]))
    PY
"""

from __future__ import annotations

from training.analysis.families import DEFAULT_MANIFEST, REGISTRY
from training.analysis.fq.cli import build_cli


main = build_cli(REGISTRY, default_manifest=DEFAULT_MANIFEST)


if __name__ == "__main__":
    main()
