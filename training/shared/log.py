"""Stdout/stderr capture utilities for CLI scripts."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO


class _Tee:
    """Fan-out writes to multiple text streams."""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, s: str) -> int:
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self) -> None:
        for st in self.streams:
            st.flush()


@contextmanager
def tee_stdio(log_path: Path) -> Iterator[None]:
    """Mirror stdout + stderr into `log_path` for the duration of the block.

    Line-buffered so `tail -f` sees output live. Restores the original
    streams on exit, including on exception.

    Caveat: this swaps `sys.stdout`/`sys.stderr` at the Python level, so
    any code that stashed a reference to the original streams before
    the `with` block will bypass the tee. Fine for CLI entrypoints
    wrapping their own work; not for library code. Fd-level redirection
    (`os.dup2`) would close that gap but isn't worth the complexity here.
    """
    log = log_path.open("w", buffering=1)
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(orig_out, log)
    sys.stderr = _Tee(orig_err, log)
    try:
        yield
    finally:
        sys.stdout = orig_out
        sys.stderr = orig_err
        log.close()
