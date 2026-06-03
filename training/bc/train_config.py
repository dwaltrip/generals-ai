"""Inputs to a BC training run + small helpers tied to the run identity.

`TrainConfig` is the public contract: every entry point (local CLI wrapper,
Modal cloud entry, notebook) builds one of these and hands it to `bc.train.bc_run`.

`make_run_id` is the UTC-timestamp run-id factory used by the CLI bridge to
fill in `TrainConfig.run_dir`. Caller-side generation lets the cloud entry
print the run path *before* the remote call kicks off, so the operator knows
where to look on the outputs Volume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from bc.model_config import ModelConfig, build_model_cfg


@dataclass(frozen=True)
class TrainConfig:
    """Inputs to a BC training run. Structural invariants checked at
    construction; existence checks for `manifest`/`intermediate` happen
    in `bc_run` (so cloud runs check after the Volume is mounted).

    Fields fall into three groups that decide how they're supplied:
      - `arch` (model design) — from the `--config` file; recorded in the
        checkpoint's `arch` key so a checkpoint self-describes its model.
      - recipe (`lr`, `batch_size`, `epochs`, `seed`, `precision`,
        `shuffle_buffer_size`, `gpu`, `manifest`, `intermediate`) — from the
        `--config` file; the run's reproducible identity, not in the checkpoint.
      - operational / invocation-local (`run_dir`, `device`, `num_workers`,
        `pin_memory`, `prefetch_factor`, `log_every`, `skip_val`,
        `max_batches`) — from CLI flags; where/how this invocation runs.
    The config-file fields (arch + recipe) and the operational CLI fields are
    disjoint domains, so `from_file` (file) + the CLI overrides merge as a plain
    union.
    """

    # --- recipe: required paths (no default that makes sense across envs) ---
    manifest: Path
    intermediate: Path
    # --- operational: required ---
    # Full path to this run's directory. `bc_run` creates it with
    # `exist_ok=False`, so two runs landing in the same wall-clock
    # second still collide explicitly. Wrappers usually compute this
    # as `<parent>/<make_run_id()>`.
    run_dir: Path
    # --- arch: model design ---
    # Self-describing architecture. Recorded in each checkpoint's `arch` key;
    # validated by `ModelConfig.__post_init__`.
    arch: ModelConfig = field(default_factory=build_model_cfg)
    # --- recipe (optional — env-independent defaults) ---
    epochs: int = 1
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 0
    shuffle_buffer_size: int = 2048
    # Modal GPU class for the run. Recorded provenance that rides into the
    # remote inertly; the Modal launcher reads it locally for
    # `with_options(gpu=...)`. Inert on local runs. In `_DRIFT_EXCLUDED` —
    # the card may change freely on resume.
    gpu: str = "T4"
    # --- Mixed-precision knob ---
    # "auto" picks fp16 on CUDA (engages tensor cores), fp32 elsewhere.
    # Explicit "fp32" or "fp16" overrides. fp16 on a non-CUDA device is
    # allowed but unlikely to win much — autocast still runs, but the
    # tensor-core ceiling that motivates AMP is CUDA-only.
    precision: str = "auto"
    # --- operational / invocation-local (optional) ---
    device: str = "auto"
    log_every: int = 50
    max_batches: int | None = None
    # --- DataLoader knobs ---
    # Number of subprocess workers that prefetch batches in parallel with
    # the main training loop. 0 = main-process iteration (no prefetch).
    # Default `2` provides overlap without consuming the whole CPU budget;
    # raise to 4+ on boxes with more cores.
    num_workers: int = 2
    # pin_memory controls how the DataLoader hands tensors to the GPU.
    # Only meaningful for CUDA — MPS and CPU don't benefit (no PCIe
    # transfer to overlap). `None` resolves to `True` iff the chosen
    # device is CUDA; explicit True/False overrides.
    #
    # The mechanic: CPU-side tensors normally live in "pageable" memory —
    # memory the OS is free to swap to disk under pressure. CUDA can't
    # DMA (direct memory access) directly from pageable memory to GPU
    # VRAM; it has to copy into a "pinned" (page-locked) staging buffer
    # first, then DMA from there. Two steps, with the CPU thread blocked
    # waiting for the staging copy to finish.
    #
    # pin_memory=True tells the DataLoader to allocate its batches in
    # pinned memory directly. Two wins:
    #   1. The pageable→pinned staging copy is skipped — one less hop.
    #   2. With pinned memory, `tensor.to(device, non_blocking=True)`
    #      actually runs async — the H2D transfer is queued on a CUDA
    #      stream and can overlap with GPU compute on the previous batch.
    #      Without pinned memory, `non_blocking=True` is silently ignored.
    #
    # Cost: pinned memory is a limited system-wide resource. A handful
    # of training batches (~hundreds of MB) is fine; allocating GBs of
    # it can starve other processes / cause allocation failures.
    #
    # Note on provenance: `args.json` stores the operator's input (which
    # may be `None` for "auto"); the booled-out value actually used by
    # the DataLoader is logged to `run.log` at run start. Re-running
    # the same config on a different device intentionally re-resolves —
    # `args.json` is the canonical input, not the resolved state.
    pin_memory: bool | None = None
    # Number of batches each worker prefetches ahead of the consumer.
    # PyTorch's default is 2; ignored when num_workers == 0.
    prefetch_factor: int = 2
    # Skip the per-epoch val pass entirely. For perf-testing runs where
    # val cost dominates wall time (val sweeps the full val split and
    # can be ~10x slower than a short train segment) and we don't care
    # about val metrics. Off by default — val is wanted for real runs.
    skip_val: bool = False

    def __post_init__(self) -> None:
        valid_devices = ("auto", "cuda", "mps", "cpu")
        if self.device not in valid_devices:
            raise ValueError(f"device must be one of {valid_devices}; got {self.device!r}")
        valid_precisions = ("auto", "fp32", "fp16")
        if self.precision not in valid_precisions:
            raise ValueError(f"precision must be one of {valid_precisions}; got {self.precision!r}")
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
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0; got {self.num_workers}")
        if self.prefetch_factor < 1:
            raise ValueError(f"prefetch_factor must be >= 1; got {self.prefetch_factor}")
        # `arch` validates itself in `ModelConfig.__post_init__`.

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        # `overrides` is `Any` rather than PEP 692 `Unpack[_Overrides]`. The
        # precise form would type-check each override key, but only by
        # duplicating the operational-field list in a TypedDict and keeping it
        # in sync with the fields above — drift risk not worth the boundary
        # check, since these kwargs forward straight into the type-checked
        # constructor below.
        **overrides: Any,
    ) -> TrainConfig:
        """Build a `TrainConfig` from a `--config` JSON file plus invocation-local
        overrides.

        The file supplies the arch + recipe (`arch` as a nested object, plus the
        recipe knobs + `manifest`/`intermediate` paths) and may be partial —
        unset recipe fields fall through to this dataclass's defaults, unset arch
        fields to `build_model_cfg`'s (`MODEL_CONFIG_DEFAULTS`).
        `overrides` supplies operational (invocation-local) fields (`run_dir`,
        `device`, `num_workers`, …). The two domains are disjoint, so this is a
        plain union; an unknown file key or a field set in both surfaces as a
        `TypeError` from the constructor.
        """
        data = json.loads(Path(path).read_text())
        arch = build_model_cfg(**data.pop("arch", {}))
        for path_field in ("manifest", "intermediate"):
            if path_field in data:
                data[path_field] = Path(data[path_field])
        return cls(arch=arch, **data, **overrides)


def make_run_id() -> str:
    """UTC-timestamp run-id, e.g. `2026-05-22T19-30-00Z`.

    Dash-separated time component (not the `:` ISO 8601 uses) so the id
    is safe to use as a filesystem path on every OS we care about.
    """
    return datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")


def json_default(obj: object) -> str:
    """JSON `default=` for `asdict(TrainConfig)`. Stringifies `Path`; anything
    else that lands here is an unexpected type — raise to surface it."""
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"unserializable: {type(obj).__name__}: {obj!r}")
