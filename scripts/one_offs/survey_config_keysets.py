#!/usr/bin/env python3
"""Survey unique config key-sets across historical run args.json files.

Scoped to data/training/{runs,runs-cloud,sweeps}. For sweeps, args.json lives
under <sweep>/runs/<run>/ — a plain rglob for args.json catches all three roots
and naturally skips sweeps' base-config.json / sweep.json.

Groups runs by the frozenset of dotted key paths in their resolved args.json, to
see how many distinct config shapes exist historically and how they progressed.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

BASE = Path("/Users/danielwaltrip/all-files/projects/code/generals-ai/data/training")
ROOTS = [BASE / "runs", BASE / "runs-cloud", BASE / "sweeps"]


def dotted_keys(obj: dict, prefix: str = "") -> set[str]:
    """Dotted key paths for a JSON object. Recurses into dicts only; lists/tuples
    and scalars are leaf values (not descended into)."""
    keys: set[str] = set()
    for k, v in obj.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            keys |= dotted_keys(v, path)
        else:
            keys.add(path)
    return keys


def fmt(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def main() -> None:
    files: list[Path] = []
    for root in ROOTS:
        if not root.exists():
            print(f"(missing root: {root})", file=sys.stderr)
            continue
        files.extend(sorted(root.rglob("args.json")))

    groups: dict[frozenset[str], list[tuple[Path, float]]] = defaultdict(list)
    errors: list[tuple[Path, str]] = []
    for p in files:
        try:
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError) as e:
            errors.append((p, str(e)))
            continue
        if not isinstance(data, dict):
            errors.append((p, f"top-level is {type(data).__name__}, not dict"))
            continue
        data.pop("_resume_meta", None)  # defensive; primary args.json shouldn't carry it
        groups[frozenset(dotted_keys(data))].append((p, p.stat().st_mtime))

    print(f"Scanned {len(files)} args.json files; "
          f"{len(groups)} unique key-sets; {len(errors)} errors\n")

    # Order unique sets by earliest mtime; treat the latest-earliest as "current".
    ordered = sorted(groups.items(), key=lambda kv: min(m for _, m in kv[1]))
    current = ordered[-1][0]

    for i, (ks, members) in enumerate(ordered):
        mtimes = [m for _, m in members]
        example = min(members, key=lambda pm: pm[1])[0]
        rel = example.relative_to(BASE)
        print(f"=== set #{i}  runs={len(members):>3}  "
              f"{fmt(min(mtimes))}..{fmt(max(mtimes))}  n_keys={len(ks)}")
        print(f"    e.g. {rel}")
        extra = sorted(ks - current)    # present here, absent in current -> removed/renamed
        missing = sorted(current - ks)  # in current, absent here -> added later
        if extra:
            print(f"    NOT in current (removed/renamed/homeless candidates): {extra}")
        if missing:
            print(f"    added after this set: {missing}")
        print()

    print("--- naive chronological delta chain (consecutive unique sets) ---")
    for a, b in zip(ordered, ordered[1:]):
        added = sorted(b[0] - a[0])
        removed = sorted(a[0] - b[0])
        la = fmt(min(m for _, m in a[1]))
        lb = fmt(min(m for _, m in b[1]))
        print(f"\n{la} -> {lb}:")
        if added:
            print(f"  + {added}")
        if removed:
            print(f"  - {removed}")
        if not added and not removed:
            print("  (same keys, different values only)")

    if errors:
        print("\n--- errors ---")
        for p, e in errors:
            print(f"  {p.relative_to(BASE)}: {e}")


if __name__ == "__main__":
    main()
