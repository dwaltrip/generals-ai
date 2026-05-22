"""Inputs to a BC training run + small helpers tied to the run identity.

`TrainConfig` is the public contract: every entry point (local CLI wrapper,
Modal cloud entry, notebook) builds one of these and hands it to `bc.train.bc_run`.

`make_run_id` is the UTC-timestamp run-id factory used by the CLI bridge to
fill in `TrainConfig.run_dir`. Caller-side generation lets the cloud entry
print the run path *before* the remote call kicks off, so the operator knows
where to look on the outputs Volume.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    """Inputs to a BC training run. Structural invariants checked at
    construction; existence checks for `manifest`/`intermediate` happen
    in `bc_run` (so cloud runs check after the Volume is mounted)."""

    # Required — no default that makes sense across environments.
    manifest: Path
    intermediate: Path
    # Full path to this run's directory. `bc_run` creates it with
    # `exist_ok=False`, so two runs landing in the same wall-clock
    # second still collide explicitly. Wrappers usually compute this
    # as `<parent>/<make_run_id()>`.
    run_dir: Path
    # Optional — sensible defaults independent of environment.
    epochs: int = 1
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    device: str = "auto"
    seed: int = 0
    shuffle_buffer_size: int = 2048
    log_every: int = 50
    max_batches: int | None = None

    def __post_init__(self) -> None:
        valid_devices = ("auto", "cuda", "mps", "cpu")
        if self.device not in valid_devices:
            raise ValueError(f"device must be one of {valid_devices}; got {self.device!r}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1; got {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1; got {self.batch_size}")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0; got {self.lr}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0; got {self.weight_decay}")
        if self.shuffle_buffer_size < 0:
            raise ValueError(f"shuffle_buffer_size must be >= 0; got {self.shuffle_buffer_size}")
        if self.log_every < 1:
            raise ValueError(f"log_every must be >= 1; got {self.log_every}")
        if self.max_batches is not None and self.max_batches < 1:
            raise ValueError(f"max_batches must be >= 1 or None; got {self.max_batches}")


def make_run_id() -> str:
    """UTC-timestamp run-id, e.g. `2026-05-22T19-30-00Z`.

    Dash-separated time component (not the `:` ISO 8601 uses) so the id
    is safe to use as a filesystem path on every OS we care about.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def json_default(obj: object) -> str:
    """JSON `default=` for `asdict(TrainConfig)`. Stringifies `Path`; anything
    else that lands here is an unexpected type — raise to surface it."""
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"unserializable: {type(obj).__name__}: {obj!r}")
