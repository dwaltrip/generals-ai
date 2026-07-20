
## BC training runs

**Consumes**

- A subset of parsed replay data, as described by a manifest json file (train / val split).

**Produces**

- Model checkpoints per epoch (e.g. `epoch_001.pt`)
- Logs, generated reports (`quality.md`, `perf.md`)
- Per-epoch stratified dumps (optionally)

**Downstream consumers**

Eval tools, analysis work, etc...

**Overview and key code**

- Flow:
    - `bc_run` → `run_training` → `train_loop`. configured by `TrainConfig`.
    - CLI entry points:
        - `run_bc_modal.py`: primary, cloud runs via Modal.
        - `run_bc_local.py`: used mostly for smoke testing.
- Resume: rarely used, basic implementation for resuming / extending a run (`bc/resume.py`).
- Model (`BCModel`):
    - Architecture is a DeepNash-style pyramid U-Net trunk, fully described by `ModelConfig`.
    - Three primary heads:
        - Policy head: what move to make.
        - Pass head: should we move or not.
        - Value head: expected placement outcome for the perspective in this match (e.g. 1st place, 2nd place).
    - Auxilliary heads:
        - Currently just one experimental / WIP aux head, the "elim-head". Predicts which player will be eliminated next. Intended to help the value head (and the trunk overall).
- Input Data:
    - `IterableDataset` walks the parser corpus per split. Obs and targets are encoded on the fly (`bc/obs/`, `bc/targets/`).
- Loss: `bc_loss` in `loss.py` sums the per-head loss values.

**Outputs**

- Checkpoints: written via `bc/storage/checkpoint.py`
    - Fully self-describing via a stored TrainConfig, since `ckpt-config-refactor` landed (the "shape-axis" work). A legacy read path (`bc/checkpoint.py`) exists for older checkpoints.
- In-loop validation per epoch (`bc/eval/run.py`).
- Optionally capturing stratified dumps (`bc/eval/dump.py`) — An important artifact created specifically for certain analyses.
- Run logging + report generation (`run_logger.py`, `run_instrumentation.py`).

**Additional notes**

- Sweeps: `prep_sweep.py` generates per-cell configs and a runner script for kicking off each cell as a separate Modal run. The run dirs for a sweep are grouped under a dedicated sweep dir, unlike the flat list in `runs-cloud`.
- Perf instrumentation in `training/shared/` (timing spans, MFU, GPU sidecar).
