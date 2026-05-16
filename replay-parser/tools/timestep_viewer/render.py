"""Build a self-contained timestep-viewer HTML for a given replay.

Usage (from replay-parser/):
    uv run python tools/timestep_viewer/render.py <replay_id>
    uv run python tools/timestep_viewer/render.py <replay_id> --out path.html

Reads template.html + assets/*.svg from this directory, decodes the replay
from the collector DB via extract.extract, and substitutes the payload +
palette into the template. Default output: replay-parser/tmp/viewer-<id>.html.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import constants
from extract import extract


HERE = Path(__file__).resolve().parent
TEMPLATE_PATH = HERE / "template.html"
ASSETS_DIR = HERE / "assets"
DEFAULT_OUT_DIR = HERE.parent.parent / "tmp"


def svg_inner(svg_text: str) -> str:
    """Strip the outer <svg ...> wrapper, returning just the inner content
    so it can be embedded inside a <symbol viewBox=...>...</symbol>."""
    match = re.search(r"<svg\b[^>]*>(.*)</svg>", svg_text, re.DOTALL)
    if not match:
        raise ValueError("could not find <svg>...</svg> in asset")
    return match.group(1).strip()


def slot_color_rules(colors: list[str]) -> str:
    """CSS rules mapping .tile.slot-{i} to its slot background color. Indents
    to match the rest of the <style> block so the output reads cleanly."""
    return "\n".join(
        f"  .tile.slot-{i} {{ background: {c}; }}"
        for i, c in enumerate(colors)
    )


def safe_json(payload: dict) -> str:
    """JSON-encode for embedding inside an inline <script>. Escapes </ to
    prevent the browser's HTML parser from treating any string content
    containing </script> as ending the script tag."""
    return json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")


def build_html(replay_id: str) -> str:
    payload = extract(replay_id)
    template = TEMPLATE_PATH.read_text()
    substitutions = {
        "$REPLAY_ID": replay_id,
        "$NEUTRAL_BG": constants.NEUTRAL_TILE_BG,
        "$MOUNTAIN_BG": constants.MOUNTAIN_TILE_BG,
        "$NEUTRAL_CITY_BG": constants.NEUTRAL_CITY_BG,
        "$TILE_BORDER": constants.TILE_BORDER,
        "$DEAD_PLAYER_BG": constants.DEAD_PLAYER_BG,
        "$EVENT_LOG_BG": constants.EVENT_LOG_BG,
        "$SLOT_COLOR_RULES": slot_color_rules(constants.SLOT_COLORS),
        "$SLOT_COLORS_JSON": json.dumps(constants.SLOT_COLORS),
        "$SVG_MOUNTAIN_INNER": svg_inner((ASSETS_DIR / "mountain.svg").read_text()),
        "$SVG_CROWN_INNER": svg_inner((ASSETS_DIR / "crown.svg").read_text()),
        "$SVG_CITY_INNER": svg_inner((ASSETS_DIR / "city.svg").read_text()),
        "$DATA_JSON": safe_json(payload),
    }
    for placeholder, value in substitutions.items():
        template = template.replace(placeholder, value)
    return template


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay_id")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output path (default: replay-parser/tmp/viewer-<id>.html)",
    )
    args = parser.parse_args()

    out = args.out or (DEFAULT_OUT_DIR / f"viewer-{args.replay_id}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(args.replay_id))
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
