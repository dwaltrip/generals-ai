"""Generic formatting helpers: human-readable sizes/durations/percentages and
a thin `tabulate` wrapper for GitHub-flavored markdown tables.

Deliberately domain-free — no knowledge of runs, GPUs, or any project schema.
Anything that needs to know *what* it's formatting belongs with that domain's
code; this module only turns numbers into strings.
"""

from __future__ import annotations

from collections.abc import Sequence

import tabulate as _tabulate
from tabulate import tabulate


def format_bytes(n: float) -> str:
    """Human-readable IEC size (e.g. `252.0 MiB`)."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}"
        n /= 1024
    raise AssertionError("unreachable")  # the TiB branch always returns


def format_count(n: float) -> str:
    """Integer with thousands separators (e.g. `1,234`)."""
    return f"{round(n):,}"


def format_duration(seconds: float) -> str:
    """Compact duration: sub-minute stays in seconds (`47.0s`), longer rolls
    up to `3m 12s` / `1h 04m`."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def format_ns(ns: float | int | None) -> str:
    """Format a nanosecond value as a human-readable approximate duration."""
    if ns is None:
        return "—"
    if ns < 1_000:
        return f"~{ns:.0f} ns"
    if ns < 1_000_000:
        return f"~{ns / 1e3:.1f} µs"
    if ns < 1_000_000_000:
        return f"~{ns / 1e6:.1f} ms"
    return f"~{ns / 1e9:.2f} s"


def format_ns_exact(ns: int | None) -> str:
    """Format an exact nanosecond value (no ~ prefix)."""
    if ns is None:
        return "—"
    if ns < 1_000:
        return f"{ns} ns"
    if ns < 1_000_000:
        return f"{ns / 1e3:.1f} µs"
    if ns < 1_000_000_000:
        return f"{ns / 1e6:.1f} ms"
    return f"{ns / 1e9:.2f} s"


def format_loss(val: float, *, dp: int | None = None) -> str:
    """Format a loss value.  When *dp* is None (default), uses 4 decimal places
    under 1 and 3 above (`0.4190`, `1.223`).  When set, uses that many decimal
    places everywhere."""
    if dp is not None:
        return f"{val:.{dp}f}"
    if val < 1:
        return f"{val:.4f}"
    return f"{val:.3f}"


def format_rate(val: float) -> str:
    """Format a throughput rate with no decimal places (`2209.49 -> 2,209`)."""
    return f"{round(val):,}"


def format_pct(fraction: float, *, digits: int = 1) -> str:
    """Format a 0–1 fraction as a percentage (`0.179 -> 17.9%`)."""
    return f"{fraction * 100:.{digits}f}%"


def md_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
    *,
    align: Sequence[str] | None = None,
) -> str:
    """GitHub-flavored markdown table via `tabulate`. `align` is an optional
    per-column alignment (`"left"`/`"right"`/`"center"`), handy for right-
    aligning numeric columns.

    Uses the `pipe` format rather than `github`: both render on GitHub, but
    `github` always emits a plain `|----|` separator and drops the alignment,
    while `pipe` emits the `|:---|---:|` markers GFM honors — so `align`
    actually takes effect in rendered output, not just in the raw whitespace.
    """
    # disable_numparse: cells are pre-formatted strings; without this tabulate
    # re-parses numeric-looking ones and reformats them (e.g. "666.0" -> "666"),
    # clobbering our chosen precision.
    kwargs: dict = {"headers": list(headers), "tablefmt": "pipe", "disable_numparse": True}
    if align is not None:
        kwargs["colalign"] = tuple(align)
    # tabulate sizes each column to max(len(header) + MIN_PADDING, widest cell),
    # with MIN_PADDING defaulting to 2 — so columns sized by their header (wider
    # than the data) carry 2 extra spaces. We drop it to 0 for tighter tables and
    # restore the value afterwards so we don't affect other tabulate callers.
    # NOTE: mutating the global is not thread-safe — concurrent md_table calls
    # would race on MIN_PADDING. Fine for synchronous CLI use.
    saved_min_padding = _tabulate.MIN_PADDING
    _tabulate.MIN_PADDING = 0
    try:
        return tabulate(rows, **kwargs)
    finally:
        _tabulate.MIN_PADDING = saved_min_padding
