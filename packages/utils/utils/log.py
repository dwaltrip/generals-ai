"""Stdout/stderr capture utilities for CLI scripts."""

from __future__ import annotations

import sys
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Protocol


class _Writable(Protocol):
    """Structural type for the bits of TextIO that `_Tee` /
    `_TimestampedStream` actually use. Avoids requiring full TextIO
    conformance (seek, read, isatty, ...) for our internal wrappers,
    which are write-only sinks.
    """

    def write(self, s: str, /) -> int: ...
    def flush(self) -> None: ...


class _Tee:
    """Fan-out writes to multiple text streams."""

    def __init__(self, *streams: _Writable) -> None:
        self.streams = streams

    def write(self, s: str) -> int:
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self) -> None:
        for st in self.streams:
            st.flush()


class _TimestampedStream:
    """Prefix each output line with a UTC `[HH:MM:SS]` timestamp.

    Wraps a downstream stream. Buffers partial writes until a newline
    arrives, so callers using `print` (which emits text then newline as
    separate writes) get one timestamp per line rather than per write.

    Blank lines (just `"\\n"`) pass through without a timestamp prefix
    so visual spacing in the log isn't cluttered with timestamps on
    empty rows.
    """

    def __init__(self, downstream: _Writable) -> None:
        self.downstream = downstream
        self._buffer = ""

    @staticmethod
    def _now_prefix() -> str:
        return datetime.now(timezone.utc).strftime("[%H:%M:%S] ")

    def write(self, s: str) -> int:
        n = len(s)
        self._buffer += s
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line == "":
                self.downstream.write("\n")
            else:
                self.downstream.write(self._now_prefix() + line + "\n")
        return n

    def flush(self) -> None:
        # Flush any pending partial line so a `tail -f` viewer sees it
        # even if the producer didn't terminate with a newline. Mostly
        # matters for traceback tails / progress bars; routine prints
        # always end with a newline.
        if self._buffer:
            self.downstream.write(self._now_prefix() + self._buffer)
            self._buffer = ""
        self.downstream.flush()


@contextmanager
def tee_stdio(log_path: Path) -> Iterator[None]:
    """Mirror stdout + stderr into `log_path` for the duration of the block.

    Output goes to both terminal and file, with each line prefixed by a
    UTC `[HH:MM:SS]` timestamp on both sides. The file is line-buffered
    so `tail -f` sees output live.

    On exception, the traceback is written into the log file (with
    timestamps) before `finally` restores the streams — otherwise
    Python's default handler would only fire after restoration and the
    log would miss it. The traceback write targets the log file only;
    Python's default handler prints the (untimestamped) terminal copy
    after we re-raise.

    Caveat: this swaps `sys.stdout`/`sys.stderr` at the Python level, so
    any code that stashed a reference to the original streams before
    the `with` block will bypass the tee. Fine for CLI entrypoints
    wrapping their own work; not for library code. Fd-level redirection
    (`os.dup2`) would close that gap but isn't worth the complexity here.
    """
    log = log_path.open("w", buffering=1)
    orig_out, orig_err = sys.stdout, sys.stderr
    # Single timestamper at the top of each chain, fanning out to both
    # terminal and log via `_Tee`. One source of timestamp truth per
    # stream — terminal and log see the same prefix for any given line.
    ts_stdout = _TimestampedStream(_Tee(orig_out, log))
    ts_stderr = _TimestampedStream(_Tee(orig_err, log))
    sys.stdout = ts_stdout
    sys.stderr = ts_stderr
    try:
        yield
    except BaseException:
        # Log file only — Python's default exception handler will print
        # the (untimestamped) traceback to the terminal after we re-raise.
        log_only = _TimestampedStream(log)
        traceback.print_exc(file=log_only)
        log_only.flush()
        raise
    finally:
        ts_stdout.flush()
        ts_stderr.flush()
        sys.stdout = orig_out
        sys.stderr = orig_err
        log.close()
