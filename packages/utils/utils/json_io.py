"""Write JSON to disk, formatted compactly via `fjson` when available."""

import json
import subprocess
import sys
from pathlib import Path


def write_json(
    path: Path,
    data: object,
    *,
    indent: int = 2,
    width: int = 100,
) -> None:
    """Write `data` to `path` as JSON.

    Writes plain `json.dump` first, then reformats in-place via `fjson` to
    collapse inner sequences onto aligned rows (much nicer to diff than the
    one-element-per-line `indent=2` output). Falls back to the plain JSON
    on disk — with a WARNING to stderr — if `fjson` isn't installed or
    errors out. The fallback file is still valid JSON, just verbose.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fp:
        json.dump(data, fp, indent=indent)
        fp.write("\n")
    try:
        subprocess.run(
            ["fjson", "-i", str(indent), "-w", str(width), str(path), "-o", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        print(
            f"WARNING: fjson not found on PATH; wrote unformatted JSON to {path}. "
            f"Install with: cargo install fjson",
            file=sys.stderr,
        )
    except subprocess.CalledProcessError as e:
        print(
            f"WARNING: fjson failed on {path}: {e.stderr.strip()}; "
            f"left unformatted JSON in place.",
            file=sys.stderr,
        )
