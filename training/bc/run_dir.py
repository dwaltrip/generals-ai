"""Run-directory lifecycle for BC training.

`initialize_run_dir` creates the run dir and drops provenance before
training starts. `RunArtifacts` bundles the per-run runtime resources
(file handles via `RunLogger`, paths, run-start measurements) and owns
their open/close lifecycle as a context manager.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bc.run_logger import RunLogger
from bc.train_config import TrainConfig, json_default


def initialize_run_dir(config: TrainConfig) -> None:
    """Create `config.run_dir` and persist run provenance.

    Mkdirs the run dir with `exist_ok=False` so two runs landing in the
    same wall-clock second collide explicitly. Writes `args.json` (full
    `TrainConfig` as JSON) and announces the path. Call before `bc_run`.

    Split out from `bc_run` so cloud callers can drop sibling provenance
    files (e.g. `args_cloud.json`) into the run dir *before* training
    starts, instead of relying on a try/finally cleanup hook.
    """
    config.run_dir.mkdir(parents=True, exist_ok=False)
    print(f"run dir: {config.run_dir}")
    with (config.run_dir / "args.json").open("x") as fp:
        json.dump(asdict(config), fp, default=json_default, indent=2)


@dataclass
class RunArtifacts:
    """Per-run runtime: born at run start, released at run end.

    Wraps the file handles (via `RunLogger`), paths, and run-start
    measurements (FLOPs/sample, device peak) that aren't part of the
    serialized training state. Acts as a context manager: `__enter__`
    opens the logger's JSONL writers, `__exit__` closes them.

    `suffix` distinguishes resume segments ("" for the original run,
    "_resume_NN" for resumes) and is the single source for this segment's
    artifact filenames — the `RunLogger`, `log_path`, and `gpu_util_path`
    all derive from it.

    TODO: consider folding the `gpu_util_sidecar` lifecycle in here too —
    it's per-run runtime by the same definition. Currently it's a sibling
    context manager at the call site because it needs `device`.
    """

    run_dir: Path
    ckpt_dir: Path
    flops_per_sample: int
    peak_tflops: float | None
    suffix: str = ""
    resume_segment: int = 0
    logger: RunLogger = field(init=False)

    def __post_init__(self) -> None:
        self.logger = RunLogger(self.run_dir, self.suffix)

    @property
    def log_path(self) -> Path:
        return self.run_dir / f"run{self.suffix}.log"

    @property
    def gpu_util_path(self) -> Path:
        return self.run_dir / f"gpu_util{self.suffix}.jsonl"

    def __enter__(self) -> RunArtifacts:
        self.logger.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.logger.close()
