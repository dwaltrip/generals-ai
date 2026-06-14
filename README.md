# generals-ai

Building an AI bot to play the [generals.io](https://generals.io) strategy game in FFA mode.

> **NOTE (2026-06): this README is out of date — a cleanup is now underway.** Older sections may not reflect the current state of the project; trust the linked docs and `AGENTS.md` over this file where they conflict. New, current material is being appended at the end as the rewrite proceeds.

## Setup (one-time, after clone)

```sh
./tools/setup-git-hooks.sh
```

Points git at the repo's tracked hooks under `.githooks/`, which includes a pre-commit hook that regenerates `modal_requirements.txt` files whenever `uv.lock` changes.

## Where to look next

- [`AGENTS.md`](./AGENTS.md) — project overview, sub-projects, key entry points, tooling notes.
- [`docs/`](./docs/) — design docs and references (game format, API, network architecture, etc.).
- [`replay-collector/README.md`](./replay-collector/README.md) — operator guide for the replay collector sub-project.

## Frozen-representation probes

Diagnostic tooling for asking *is signal X decodable from a frozen representation, and at what capacity?* — built to investigate whether a trained trunk preserves a given signal (e.g. per-player army) or discards it. A probe freezes a representation, caches it over a held-out validation sub-split, then trains small swap-in heads on it and reads their generalization.

- **`packages/training/training/analysis/probes/`** — the reusable framework. A `FeatureSource` selects *what representation* to read (`TrunkSource` wraps a frozen checkpoint's trunk; `RawObsSource` is the identity control that reads the obs directly), and a `ProbeTask` defines *what signal* to decode (the per-frame target, the head variants spanning capacities — linear / deployed-shape / fat — the loss, the metrics, and the model-free baselines). `core.py` holds the orchestrator; `cli.py` the shared run flow.
- **`packages/training/training/analysis/probe_runs/`** — one-off experiment tasks built on the framework, each a self-contained runnable script (e.g. `who_dies_next.py`). A new probe is one new `ProbeTask`; reusing it across representations is a swapped `FeatureSource`.

Reading linear vs. fat across feature sources localizes a failure: the raw-obs control sets the floor (the signal is in the inputs), a linear readout reaching it off the trunk means the trunk preserves the signal, only-fat means it's present but entangled, and neither means the trunk discarded it.

Runs write to `data/analysis/probes/TIMESTAMP-task-source/` (`summary.json` + a full `run.log`) by default. Example:

```sh
cd packages/training
./training/analysis/probe_runs/who_dies_next.py \
    --source trunk --checkpoint <ckpt.pt> --manifest <split.json> \
    --epochs 6 --max-train-frames 24000 --max-val-frames 10000
```
