"""Stdout summary stats + ASCII histogram for numeric distributions."""

from collections.abc import Sequence


def print_counts_with_percents(
    items: Sequence[tuple[object, int]],
    cumulative: bool = False,
) -> None:
    """Print a (label, count, %) table from ordered (label, count) pairs.

    Set cumulative=True to add a running cumulative-percent column.
    """
    if not items:
        return
    total = sum(c for _, c in items)
    labels = [str(lbl) for lbl, _ in items]
    width = max(len(s) for s in labels)
    cum = 0
    for (_, count), label_str in zip(items, labels):
        pct = 100 * count / total
        if cumulative:
            cum += count
            cpct = 100 * cum / total
            print(f"  {label_str:>{width}}: {count:>7}  ({pct:5.2f}%)  cum {cpct:6.2f}%")
        else:
            print(f"  {label_str:>{width}}: {count:>7}  ({pct:5.2f}%)")


def print_distribution(
    name: str,
    values: Sequence[float],
    bins: int = 10,
) -> None:
    if not values:
        print(f"{name}: empty")
        return

    n = len(values)
    sorted_vals = sorted(values)

    def pct(p: float) -> float:
        return sorted_vals[int(p * (n - 1))]

    mean = sum(values) / n
    print(f"{name}: n={n}  mean={mean:.4f}")
    print(
        f"  p10={pct(0.10):.4f}  p25={pct(0.25):.4f}  p50={pct(0.50):.4f}"
        f"  p75={pct(0.75):.4f}  p90={pct(0.90):.4f}"
    )

    lo, hi = sorted_vals[0], sorted_vals[-1]
    if hi <= lo:
        return

    width = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        idx = min(int((v - lo) / width), bins - 1)
        counts[idx] += 1

    max_count = max(counts)
    bar_w = 40
    for i, c in enumerate(counts):
        bin_lo = lo + i * width
        bin_hi = bin_lo + width
        bar = "#" * int(bar_w * c / max_count) if max_count > 0 else ""
        print(f"  [{bin_lo:6.3f}, {bin_hi:6.3f}): {c:>6} ({c / n * 100:5.2f}%) {bar}")
