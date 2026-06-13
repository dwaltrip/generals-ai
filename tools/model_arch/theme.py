"""Color palette for the architecture diagrams — the diagram-semantic
counterpart to svgkit's generic primitives.

svgkit knows how to render a box of some `kind` and how to turn a palette into
CSS; this module decides which kinds exist (encoder / decoder / auxiliary) and
their colors. New box kinds — e.g. a head color for the top-level diagram — are
added here, not in svgkit.
"""

from __future__ import annotations

from model_arch.svgkit import KindStyle


PALETTE: dict[str, KindStyle] = {
    "enc": KindStyle(fill="#e1f5ee", stroke="#0f6e56", title="#085041", sub="#0f6e56"),
    "dec": KindStyle(fill="#eeedfe", stroke="#534ab7", title="#3c3489", sub="#534ab7"),
    "aux": KindStyle(fill="#f1efe8", stroke="#5f5e5a", title="#444441", sub="#5f5e5a"),
    # Top-level overview kinds: io = input/output tensors, head = output heads.
    "io": KindStyle(fill="#f4f4f2", stroke="#9a9893", title="#444441", sub="#6b6a66"),
    "head": KindStyle(fill="#e7f0fb", stroke="#2f6cb5", title="#1f4f88", sub="#2f6cb5"),
}
