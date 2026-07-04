"""
Train/val split manifest for the BC training corpus.

Produces and consumes `data/training/splits/<name>.json` — a deterministic,
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
  - `git_sha` (str): short git SHA when built.
  - `corpus_size` (int): total sim files scanned (post-`scan_limit` if set).
  - `dropped_games` (int): games rejected by the filter (`eligible_perspectives` → []).
  - `kept_pairs` (int): total `(rid, k)` pairs after filtering.
  - `curated_names_count` (int): size of the curated-name set the filter
    intersected against — provenance for "which curated lists were in
    effect when this manifest was built."
  - `val_frac` (float): requested validation fraction (actual ratio may
    differ by one pair due to rounding).
  - `scan_limit` (int | null): if set, scans a seeded random sample of N
    sim files instead of the full corpus. `null` means a full-corpus scan.
    With the same `seed`, smaller scan_limits are prefixes of larger ones
    (subset property holds across runs).
  - `eval_map_sets` (list[str]): names of the published eval map sets whose
    replay ids were excluded from this manifest (eval maps stay disjoint
    from training data).
  - `eval_map_ids_excluded` (int): corpus games dropped by that exclusion.
  - `train` (list[[str, int]]): list of `[replay_id, perspective_index]`.
  - `val` (list[[str, int]]): same shape.

Usage via CLI:
    uv run python -m training.bc.splits build --seed 42 --out <data-dir>/splits/v1.json
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import random

from settings import (
    CURATED_LISTS_MANIFEST,
    EVAL_MAP_SETS_DIR,
    INTERMEDIATE_DIR,
    PROJECT_ROOT,
)
from training.bc.filters import DROP_REASONS, FILTER_VERSION, eligible_perspectives
from training.bc.utils import list_sim_paths, meta_path_for
from utils.docstring import doc_summary
from utils.git import git_sha
from utils.json_io import write_json
from utils.player_name_lists import load_union


MANIFEST_VERSION = 1
DEFAULT_VAL_FRAC = 0.05


def load_curated_names() -> set[str]:
    """Load the default curated-player-name union as a set for O(1)
    membership tests. The set is the cross-list union (insertion-order
    dedup) at the current state of `curated-player-lists.txt`.

    Callers can pass an explicit `curated_names=` to `build_manifest` to
    override — useful for tests that want a small known fixture set.
    """
    return set(load_union(CURATED_LISTS_MANIFEST, PROJECT_ROOT))


def load_eval_map_ids(
    map_sets_dir: Path = EVAL_MAP_SETS_DIR,
) -> tuple[set[str], list[str]]:
    """Replay ids reserved by published eval map sets, as (id union, set names).

    Eval skill-runs play on these maps, so they must never enter training data
    — a checkpoint that trained on a map could in principle have memorized
    hidden details (enemy spawns, city locations) the eval assumes are unknown.
    Built from every `<set>/manifest.json` under `map_sets_dir`; an absent dir
    means no sets are published yet (empty union).
    """
    ids: set[str] = set()
    names: list[str] = []
    if not map_sets_dir.exists():
        return ids, names
    for manifest_path in sorted(map_sets_dir.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        names.append(manifest["name"])
        for bucket in manifest["buckets"].values():
            ids.update(entry[0] for entry in bucket)
    return ids, names


def build_manifest(
    intermediate_root: Path,
    seed: int,
    val_frac: float = DEFAULT_VAL_FRAC,
    sim_paths: list[Path] | None = None,
    scan_limit: int | None = None,
    log_every: int | None = None,
    curated_names: set[str] | None = None,
    eval_map_ids: set[str] | None = None,
    eval_map_set_names: list[str] | None = None,
) -> dict:
    """
    Scan corpus, apply filters, seeded-shuffle eligible `(rid, k)` pairs, split.

    `sim_paths`, if provided, bypasses `list_sim_paths(intermediate_root)` —
    the same seam tests use to share a cached session-scoped sim list. The
    recorded `intermediate_root` field still reflects the caller's argument
    so the manifest is machine-portable.

    `scan_limit`, if set, caps the resolved path list to the first N entries
    — produces a smoke manifest in seconds rather than minutes. Applied after
    `sim_paths` resolution, so the two compose.

    `log_every` prints "scanned X / Y" progress every N games; useful for the
    CLI on the full corpus. Pass `None` to silence (tests don't need it).

    `curated_names`, if `None`, defaults to `load_curated_names()` — the
    cross-list union from `curated-player-lists.txt`. Tests pass an explicit
    set to keep fixtures small and known.

    `eval_map_ids` / `eval_map_set_names`, if `None`, default to
    `load_eval_map_ids()` — every published eval map set's ids. Same explicit
    seam for tests.
    """
    assert 0.0 < val_frac < 1.0, f"val_frac must be in (0, 1), got {val_frac}"

    if curated_names is None:
        curated_names = load_curated_names()
    if eval_map_ids is None:
        eval_map_ids, eval_map_set_names = load_eval_map_ids()
    eval_map_set_names = eval_map_set_names or []

    paths = list(sim_paths) if sim_paths is not None else list_sim_paths(intermediate_root)
    # Eval-map exclusion comes before the scan_limit sample so a capped scan
    # still fills its full quota from eligible games.
    n_before = len(paths)
    paths = [p for p in paths if p.stem not in eval_map_ids]
    eval_excluded = n_before - len(paths)
    if scan_limit is not None:
        # Seeded shuffle so scan_limit yields a random sample — otherwise it
        # would be the first N lexicographically sorted replays. Fresh RNG
        # keeps this independent of the pair shuffle below; both reproducible
        # from `seed`.
        random.Random(seed).shuffle(paths)
        paths = paths[:scan_limit]
    n_total = len(paths)

    pairs: list[tuple[str, int]] = []
    dropped_games = 0
    drop_counts: dict[str, int] = {r: 0 for r in DROP_REASONS}
    for i, sim_path in enumerate(paths):
        ks = eligible_perspectives(
            sim_path, meta_path_for(sim_path), curated_names, drop_counts,
        )
        if not ks:
            dropped_games += 1
        else:
            rid = sim_path.stem
            for k in ks:
                pairs.append((rid, k))
        if log_every is not None and (i + 1) % log_every == 0:
            print(f"  scanned {i + 1:,} / {n_total:,} games | kept {len(pairs):,} pairs")

    if log_every is not None:
        parts = "  ".join(f"{r}={drop_counts[r]:,}" for r in DROP_REASONS)
        print(f"  perspective drops: {parts}")

    if not pairs:
        # Catches "pointed CLI at the wrong directory" + "filters rejected
        # everything" before downstream code (CLI print, training loop) hits
        # a div-by-zero or silent-no-op on empty splits.
        raise ValueError(
            f"no eligible (game, perspective) pairs found under {intermediate_root}. "
            f"Scanned {n_total} games; filters dropped all of them."
        )

    rng = random.Random(seed)
    rng.shuffle(pairs)

    n_val = round(len(pairs) * val_frac)
    if n_val == 0:
        # Banker's rounding (`round(10 * 0.05) == 0`) silently empties the val
        # split at tiny `kept_pairs` — chunk 5's per-epoch val iteration would
        # then yield zero batches without complaint. Loud is better.
        raise ValueError(
            f"val split would be empty: kept_pairs={len(pairs)}, val_frac={val_frac}. "
            "Use a larger corpus or a higher val_frac."
        )
    val = pairs[:n_val]
    train = pairs[n_val:]

    return {
        "version": MANIFEST_VERSION,
        "seed": seed,
        "built_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "intermediate_root": str(intermediate_root),
        "filter_version": FILTER_VERSION,
        "git_sha": git_sha(),
        "corpus_size": n_total,
        "dropped_games": dropped_games,
        "kept_pairs": len(pairs),
        "curated_names_count": len(curated_names),
        "val_frac": val_frac,
        "scan_limit": scan_limit,
        "eval_map_sets": eval_map_set_names,
        "eval_map_ids_excluded": eval_excluded,
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
    for key in ("version", "seed", "filter_version", "kept_pairs", "train", "val"):
        assert key in m, f"manifest missing required key: {key}"
    assert m["version"] == MANIFEST_VERSION, (
        f"manifest schema version mismatch: file has {m['version']}, expected {MANIFEST_VERSION}"
    )
    # Catches truncation / hand-edit damage that would silently shrink the train
    # or val list. A non-matching count means the file isn't what its provenance
    # claims it is.
    n_train_val = len(m["train"]) + len(m["val"])
    assert n_train_val == m["kept_pairs"], (
        f"manifest length mismatch: train+val={n_train_val}, kept_pairs={m['kept_pairs']}"
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

    Shard directories are resolved case-insensitively. macOS APFS is
    case-insensitive ("first-writer-wins" on dir name), so a slice built on
    macOS can have shard dirs whose case doesn't match the rid's prefix —
    e.g., rid `jCxxx` expects `jC/` but the dir on disk is `JC/` because an
    uppercase-prefix rid created it first. On Linux readers (our Modal
    Volume mounts) the rid-derived path would 404. We build a one-shot
    `lower(prefix) → actual_prefix` map from the on-disk shard dirs and
    route lookups through it.
    """
    assert split in ("train", "val"), f"unknown split: {split}"
    shard_case_map = {
        p.name.lower(): p.name for p in intermediate_root.iterdir() if p.is_dir()
    }
    out: list[tuple[Path, int]] = []
    for rid, k in manifest[split]:
        actual_prefix = shard_case_map.get(rid[:2].lower(), rid[:2])
        assert actual_prefix is not None, f"bad prefix for {rid}"
        sim_path = intermediate_root / actual_prefix / f"{rid}.npz"
        out.append((sim_path, int(k)))
    return out


def _cmd_build(args: argparse.Namespace) -> None:
    intermediate_root = args.intermediate.resolve()
    if not intermediate_root.exists():
        raise SystemExit(f"intermediate corpus not found: {intermediate_root}")

    out_path = args.out.resolve()
    if out_path.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite existing manifest: {out_path} (pass --force)")

    print(f"building manifest from {intermediate_root}")
    print(
        f"  seed={args.seed}  val_frac={args.val_frac}  "
        f"filter_version={FILTER_VERSION}  scan_limit={args.scan_limit}"
    )
    manifest = build_manifest(
        intermediate_root=intermediate_root,
        seed=args.seed,
        val_frac=args.val_frac,
        scan_limit=args.scan_limit,
        log_every=args.log_every,
    )

    write_json(out_path, manifest)

    # Function-level import: analysis depends on this module for manifest
    # loading, so importing it at module level would be a cycle.
    from training.analysis.run_metrics import sidecar_path, write_manifest_metrics

    write_manifest_metrics(out_path, intermediate_root)

    n_train = len(manifest["train"])
    n_val = len(manifest["val"])
    print()
    print(f"wrote {out_path}")
    print(f"wrote {sidecar_path(out_path)}")
    print(
        f"  corpus: {manifest['corpus_size']:,} games | "
        f"dropped: {manifest['dropped_games']:,} | "
        f"kept pairs: {manifest['kept_pairs']:,}"
    )
    if manifest["eval_map_sets"]:
        print(
            f"  eval-map exclusion: {manifest['eval_map_ids_excluded']:,} games "
            f"({', '.join(manifest['eval_map_sets'])})"
        )
    print(f"  train: {n_train:,}  |  val: {n_val:,}  ({n_val / (n_train + n_val):.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(prog="training.bc.splits", description=doc_summary(__doc__, 1))
    sub = parser.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build", help="Build a new train/val split manifest")
    build.add_argument("--seed", type=int, required=True, help="Required. No silent default.")
    build.add_argument("--out", type=Path, required=True, help="Path to write the manifest JSON.")
    build.add_argument(
        "--intermediate",
        type=Path,
        default=INTERMEDIATE_DIR,
        help=f"Intermediate corpus root (default: {INTERMEDIATE_DIR}).",
    )
    build.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC)
    build.add_argument(
        "--scan-limit",
        type=int,
        default=None,
        help=(
            "Scan a seeded random sample of N sim files instead of the full corpus. "
            "Used to build smoke or random-subset manifests cheaply. Omit for full scan."
        ),
    )
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
