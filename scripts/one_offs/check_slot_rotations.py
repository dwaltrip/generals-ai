"""Exercise run_adhoc's slot-rotation resolver against hand-checked configs.

`run_adhoc.py --slot-rotations K` plays each map K times, cyclically rotating
which policy sits in which slot (offset o → slot s runs policy (s+o)%N, offsets
stepped by N/K). This prints the offsets + resulting lineups for a spread of
policy configs so the behavior can be eyeballed against the whiteboard:

  - AABB  K=2 → the antithetic complement pair (AABB / BBAA)
  - AABB  K=4 → full slot balance
  - ABAB  K=2 → collision: rotating an interleaved pair by N/2 is a no-op, so
                offset 2 ≡ offset 0. Errors by default; --skip-dupes drops it.
  - ABAB  K=4 + skip → the order-robust balanced pair (ABAB / BABA)
  - AABC, ABABCC → 3-distinct and 6-slot cases
  - AABB  K=3 → K must divide the player count, so this errors

Run from the repo root: `uv run python scripts/one_offs/check_slot_rotations.py`
"""

from eval_tools.runner import make_slot_map, rotation_offsets


def lineups(specs: list[str], offsets: list[int]) -> list[str]:
    """Render each offset's slot lineup as a spaced string (e.g. 'A A B B')."""
    n = len(specs)
    return [" ".join(specs[make_slot_map(n, o)[s]] for s in range(n)) for o in offsets]


def show(specs: list[str], k: int, skip: bool = False) -> None:
    tag = f"{''.join(specs)} K={k}" + (" --skip-dupes" if skip else "")
    try:
        offs, note = rotation_offsets(specs, len(specs), k, skip)
    except ValueError as e:
        print(f"{tag}: ERROR\n    " + str(e).replace("\n", "\n    "))
        return
    print(f"{tag}: offsets={offs}  lineups={lineups(specs, offs)}"
          + (f"  note='{note}'" if note else ""))


def main() -> None:
    show(list("AABB"), 2)
    show(list("AABB"), 4)
    show(list("ABAB"), 2)
    show(list("ABAB"), 2, skip=True)
    show(list("ABAB"), 4, skip=True)
    show(list("AABC"), 2)
    show(list("AABC"), 4)
    show(list("ABABCC"), 3)
    show(list("ABABCC"), 6)
    show(list("AABB"), 3)   # K does not divide N -> error


if __name__ == "__main__":
    main()
