"""JSONL writers for a BC training run.

`RunLogger` is the single source of truth for the per-batch / per-epoch
JSONL record format. It owns `batches.jsonl` and `epochs.jsonl` for one
run segment; the `suffix` distinguishes resume segments (`""` for the
original run, `"_resume_01"` for the first resume, etc.).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TextIO


class RunLogger:
    """Owns the `batches.jsonl` + `epochs.jsonl` writers for one run segment.

    Files are opened by `open()` (so creation is deferred to run start) and
    closed by `close()`. Both are line-buffered, so a `tail -f` sees each
    record as soon as its newline lands. `RunArtifacts` drives the
    open/close lifecycle as a context manager.
    """

    def __init__(self, run_dir: Path, suffix: str = "") -> None:
        self._batches_path = run_dir / f"batches{suffix}.jsonl"
        self._epochs_path = run_dir / f"epochs{suffix}.jsonl"
        self._batches_fp: TextIO | None = None
        self._epochs_fp: TextIO | None = None

    def open(self) -> None:
        self._batches_fp = self._batches_path.open("w", buffering=1)
        self._epochs_fp = self._epochs_path.open("w", buffering=1)

    def log_batch(self, record: dict) -> None:
        assert self._batches_fp is not None, "RunLogger.open() not called"
        self._batches_fp.write(json.dumps(record) + "\n")

    def log_epoch(self, record: dict) -> None:
        assert self._epochs_fp is not None, "RunLogger.open() not called"
        self._epochs_fp.write(json.dumps(record) + "\n")

    def close(self) -> None:
        for fp in (self._batches_fp, self._epochs_fp):
            if fp is not None:
                fp.close()
