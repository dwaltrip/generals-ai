"""Render a self-play game as a self-contained timestep-viewer HTML.

Mirrors `timestep_viewer.render.build_html` but skips the DB-fetch step:
instead of `extract(replay_id)` (which looks up wire_data, parses, and
re-simulates), we hand the already-finished sim State directly to
`_build_payload(replay_id, state, fake_replay)`.

`_build_payload` reads the map spec off `fake_replay.static.map`, so we
wrap the `StaticMap` we seeded the game with (via `sim_core.new_state`)
in a matching `.static.map` shape.

`timestep_viewer` lives under `replay-parser/tools/` and isn't a Python
package (no `__init__.py`, not in the replay-parser wheel build). We
locate it relative to the installed `replay_parser` package and add
its dir to `sys.path` so `extract` and `constants` import.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from typing import Any

from game_types import StaticMap

import replay_parser
import sim_core


_REPLAY_PARSER_ROOT = Path(replay_parser.__file__).resolve().parent.parent
_TIMESTEP_VIEWER_DIR = _REPLAY_PARSER_ROOT / "tools" / "timestep_viewer"
_ASSETS_DIR = _TIMESTEP_VIEWER_DIR / "assets"
_TEMPLATE_PATH = _TIMESTEP_VIEWER_DIR / "template.html"

if str(_TIMESTEP_VIEWER_DIR) not in sys.path:
    sys.path.insert(0, str(_TIMESTEP_VIEWER_DIR))

import constants as _viewer_constants  # noqa: E402  (path-dependent import)
from extract import _build_payload  # noqa: E402


def _svg_inner(svg_text: str) -> str:
    match = re.search(r"<svg\b[^>]*>(.*)</svg>", svg_text, re.DOTALL)
    if not match:
        raise ValueError("could not find <svg>...</svg> in asset")
    return match.group(1).strip()


def _slot_color_rules(colors: list[str]) -> str:
    return "\n".join(
        f"  .tile.slot-{i} {{ background: {c}; }}"
        for i, c in enumerate(colors)
    )


def _safe_json(payload: dict) -> str:
    """JSON-encode for embedding inside an inline <script>. Escapes </ so
    the browser's HTML parser doesn't see a </script> inside string data."""
    return json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")


def build_html_from_state(
    replay_id: str,
    state: sim_core.State,
    map_data: StaticMap,
) -> str:
    """Render a finished self-play State as a viewer HTML string.

    `replay_id` is purely a display label (used in the page title and in
    the embedded payload's static record). For self-play games a value
    like `"selfplay-<timestamp>"` works.

    `map_data` must be the same `StaticMap` used to build `state` — the
    payload pulls map dims, usernames, mountains, initial cities, and
    initial generals off it, so any divergence would render a board that
    doesn't match the snapshots.
    """
    fake_static = dataclasses.replace(
        map_data,
        usernames=[f"Player {i+1}" for i, _ in enumerate(map_data.usernames)],
    )
    return render_viewer_html(replay_id, state, fake_static)


def render_viewer_html(
    label: str,
    state: Any,
    static: Any,
) -> str:
    """Lower-level renderer that accepts any state/static duck types.

    Unlike build_html_from_state, does not call dataclasses.replace on
    static — the caller is responsible for providing the right usernames.
    Works with SimpleNamespace mocks (e.g. from loading a saved .npz).
    """
    fake_replay = SimpleNamespace(static=SimpleNamespace(map=static))
    payload = _build_payload(label, state, fake_replay)

    template = _TEMPLATE_PATH.read_text()
    substitutions = {
        "$REPLAY_ID": label,
        "$NEUTRAL_BG": _viewer_constants.NEUTRAL_TILE_BG,
        "$MOUNTAIN_BG": _viewer_constants.MOUNTAIN_TILE_BG,
        "$NEUTRAL_CITY_BG": _viewer_constants.NEUTRAL_CITY_BG,
        "$TILE_BORDER": _viewer_constants.TILE_BORDER,
        "$DEAD_PLAYER_BG": _viewer_constants.DEAD_PLAYER_BG,
        "$EVENT_LOG_BG": _viewer_constants.EVENT_LOG_BG,
        "$SLOT_COLOR_RULES": _slot_color_rules(_viewer_constants.SLOT_COLORS),
        "$SLOT_COLORS_JSON": json.dumps(_viewer_constants.SLOT_COLORS),
        "$SVG_MOUNTAIN_INNER": _svg_inner((_ASSETS_DIR / "mountain.svg").read_text()),
        "$SVG_CROWN_INNER": _svg_inner((_ASSETS_DIR / "crown.svg").read_text()),
        "$SVG_CITY_INNER": _svg_inner((_ASSETS_DIR / "city.svg").read_text()),
        "$DATA_JSON": _safe_json(payload),
    }
    for placeholder, value in substitutions.items():
        template = template.replace(placeholder, value)
    return template
