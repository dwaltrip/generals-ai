"""Shared helpers for training/investigations/* scripts.

TODO: load_replay_months() reaches into replay-collector's sqlite DB to fetch
the per-replay `started` timestamp, because that timestamp is NOT stored in
the per-game meta sidecar. If month-bucketing becomes a recurring need
across training-side code, the cleaner fix is to add `started` (or an
equivalent date field) to the meta sidecar at parse time, so training-side
code doesn't depend on the collector DB. Until then, this is an investigation
workaround.
"""

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
import random
import sqlite3
from typing import Literal

from settings import DB_PATH, INTERMEDIATE_DIR


FileKind = Literal["sim", "meta"]


def iter_intermediate_files(kind: FileKind) -> Iterator[Path]:
    for prefix_dir in sorted(INTERMEDIATE_DIR.iterdir()):
        if not prefix_dir.is_dir():
            continue
        for path in sorted(prefix_dir.iterdir()):
            name = path.name
            is_meta = name.endswith(".meta.npz")
            if kind == "meta" and is_meta:
                yield path
            elif kind == "sim" and name.endswith(".npz") and not is_meta:
                yield path


def sample_files(kind: FileKind, n: int | None, seed: int) -> list[Path]:
    all_files = list(iter_intermediate_files(kind))
    if n is None or n >= len(all_files):
        return all_files
    rng = random.Random(seed)
    return rng.sample(all_files, n)


def meta_path_for(sim_path: Path) -> Path:
    return sim_path.with_name(sim_path.stem + ".meta.npz")


def replay_id_from_path(path: Path) -> str:
    name = path.name
    if name.endswith(".meta.npz"):
        return name[: -len(".meta.npz")]
    if name.endswith(".npz"):
        return name[: -len(".npz")]
    raise ValueError(f"unexpected filename: {name}")


def load_replay_months(db_path: Path = DB_PATH) -> dict[str, str]:
    """{replay_id: "YYYY-MM"} for every row in `replays`. Bulk-fetched once."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT id, started FROM replays").fetchall()
    finally:
        conn.close()
    return {
        rid: datetime.fromtimestamp(ms / 1000, tz=UTC).strftime("%Y-%m")
        for rid, ms in rows
    }
