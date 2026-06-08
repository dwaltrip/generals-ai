from pathlib import Path


def meta_path_for(sim_path: Path) -> Path:
    return sim_path.with_name(sim_path.stem + ".meta.npz")


def sim_path_for(replay_id: str, intermediate_root: Path) -> Path:
    """Replay-id → on-disk sim path. Mirrors parser's `intermediate_dir / rid[:2]` convention."""
    return intermediate_root / replay_id[:2] / f"{replay_id}.npz"


def list_sim_paths(intermediate_root: Path) -> list[Path]:
    """Sorted list of sim file paths under the intermediate root."""
    out: list[Path] = []
    for prefix_dir in sorted(intermediate_root.iterdir()):
        if not prefix_dir.is_dir():
            continue
        for path in sorted(prefix_dir.iterdir()):
            name = path.name
            if name.endswith(".npz") and not name.endswith(".meta.npz"):
                out.append(path)
    return out
