"""
Train/val split manifest for the BC training corpus.

Produces and consumes `training/data/splits/<name>.json` — a deterministic,
seeded shuffle of eligible `(replay_id, perspective_index)` pairs split 95/5.

The manifest is the contract for what a training run learns from. Training
re-reads it on startup; the manifest's `filter_version` + `git_sha` +
`corpus_size` fields are the provenance hook for catching "corpus moved
under me" footguns — training can warn (or fail) if the on-disk corpus
doesn't match the snapshot the manifest was built against.

Manifest JSON schema (v1):
  - `version` (int): schema version, currently 1.
  - `seed` (int): the shuffle seed.
  - `built_at` (str): ISO 8601 UTC timestamp.
  - `intermediate_root` (str): path the manifest was built from. Informational
    only — `samples_for_split` resolves rids against the caller's root.
  - `filter_version` (str): `bc.filters.FILTER_VERSION` at build time.
  - `git_sha` (str): short git SHA when built; "unknown" if git fails.
  - `corpus_size` (int): total sim files scanned.
  - `dropped_games` (int): games rejected by the filter (`eligible_perspectives` → []).
  - `kept_pairs` (int): total `(rid, k)` pairs after filtering.
  - `val_frac` (float): requested validation fraction (actual ratio may
    differ by one pair due to rounding).
  - `train` (list[[str, int]]): list of `[replay_id, perspective_index]`.
  - `val` (list[[str, int]]): same shape.

CLI:
    uv run python -m bc.splits build --seed 42 --out training/data/splits/v1.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
import subprocess
from pathlib import Path

from bc.filters import FILTER_VERSION, eligible_perspectives
from bc.utils import list_sim_paths, meta_path_for, sim_path_for


MANIFEST_VERSION = 1
DEFAULT_VAL_FRAC = 0.05


def _git_sha() -> str:
    """Best-effort short git SHA; 'unknown' if git can't answer."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def build_manifest(
    intermediate_root: Path,
    seed: int,
    val_frac: float = DEFAULT_VAL_FRAC,
    sim_paths: list[Path] | None = None,
    log_every: int | None = None,
) -> dict:
    """
    Scan corpus, apply filters, seeded-shuffle eligible `(rid, k)` pairs, split.

    `sim_paths`, if provided, bypasses `list_sim_paths(intermediate_root)` —
    the same seam tests use to share a cached session-scoped sim list. The
    recorded `intermediate_root` field still reflects the caller's argument
    so the manifest is machine-portable.

    `log_every` prints "scanned X / Y" progress every N games; useful for the
    CLI on the full corpus. Pass `None` to silence (tests don't need it).
    """
    assert 0.0 < val_frac < 1.0, f"val_frac must be in (0, 1), got {val_frac}"

    paths = list(sim_paths) if sim_paths is not None else list_sim_paths(intermediate_root)
    n_total = len(paths)

    pairs: list[tuple[str, int]] = []
    dropped_games = 0
    for i, sim_path in enumerate(paths):
        ks = eligible_perspectives(sim_path, meta_path_for(sim_path))
        if not ks:
            dropped_games += 1
        else:
            rid = sim_path.stem
            for k in ks:
                pairs.append((rid, k))
        if log_every is not None and (i + 1) % log_every == 0:
            print(f"  scanned {i + 1:,} / {n_total:,} games | kept {len(pairs):,} pairs")

    rng = random.Random(seed)
    rng.shuffle(pairs)

    n_val = round(len(pairs) * val_frac)
    val = pairs[:n_val]
    train = pairs[n_val:]

    return {
        "version": MANIFEST_VERSION,
        "seed": seed,
        "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "intermediate_root": str(intermediate_root),
        "filter_version": FILTER_VERSION,
        "git_sha": _git_sha(),
        "corpus_size": n_total,
        "dropped_games": dropped_games,
        "kept_pairs": len(pairs),
        "val_frac": val_frac,
        "train": [list(p) for p in train],
        "val": [list(p) for p in val],
    }


def load_manifest(path: Path) -> dict:
    """
    Load a manifest JSON. Basic shape check only — provenance validation
    (filter_version match, corpus_size sanity) is the training script's job.
    """
    with path.open() as fp:
        m = json.load(fp)
    for key in ("version", "seed", "filter_version", "train", "val"):
        assert key in m, f"manifest missing required key: {key}"
    assert m["version"] == MANIFEST_VERSION, (
        f"manifest schema version mismatch: file has {m['version']}, code expects {MANIFEST_VERSION}"
    )
    return m


def samples_for_split(
    manifest: dict,
    split: str,
    intermediate_root: Path,
) -> list[tuple[Path, int]]:
    """
    Resolve a manifest's split into `(sim_path, perspective_k)` tuples.

    The caller supplies `intermediate_root` (not the manifest's recorded path)
    so the manifest stays portable across machines / checkout locations.
    """
    assert split in ("train", "val"), f"unknown split: {split}"
    return [(sim_path_for(rid, intermediate_root), int(k)) for rid, k in manifest[split]]


def _cmd_build(args: argparse.Namespace) -> None:
    intermediate_root = args.intermediate.resolve()
    if not intermediate_root.exists():
        raise SystemExit(f"intermediate corpus not found: {intermediate_root}")

    out_path = args.out.resolve()
    if out_path.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite existing manifest: {out_path} (pass --force)")

    print(f"building manifest from {intermediate_root}")
    print(f"  seed={args.seed}  val_frac={args.val_frac}  filter_version={FILTER_VERSION}")
    manifest = build_manifest(
        intermediate_root=intermediate_root,
        seed=args.seed,
        val_frac=args.val_frac,
        log_every=args.log_every,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fp:
        json.dump(manifest, fp, indent=2)

    n_train = len(manifest["train"])
    n_val = len(manifest["val"])
    print()
    print(f"wrote {out_path}")
    print(
        f"  corpus: {manifest['corpus_size']:,} games | "
        f"dropped: {manifest['dropped_games']:,} | "
        f"kept pairs: {manifest['kept_pairs']:,}"
    )
    print(f"  train: {n_train:,}  |  val: {n_val:,}  ({n_val / (n_train + n_val):.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(prog="bc.splits", description=__doc__.splitlines()[1])
    sub = parser.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build", help="Build a new train/val split manifest")
    build.add_argument("--seed", type=int, required=True, help="Required. No silent default.")
    build.add_argument("--out", type=Path, required=True, help="Path to write the manifest JSON.")
    build.add_argument(
        "--intermediate",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent / "replay-parser" / "data" / "intermediate",
        help="Intermediate corpus root (defaults to replay-parser/data/intermediate).",
    )
    build.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC)
    build.add_argument(
        "--log-every",
        type=int,
        default=10_000,
        help="Print scan progress every N games. 0 to silence.",
    )
    build.add_argument("--force", action="store_true", help="Overwrite an existing manifest.")
    build.set_defaults(func=_cmd_build)

    args = parser.parse_args()
    if args.log_every == 0:
        args.log_every = None
    args.func(args)


if __name__ == "__main__":
    main()
