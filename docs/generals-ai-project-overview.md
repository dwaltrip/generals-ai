
# Generals-AI Project Overview

**This overview attempts to answer the following:**

- What does this project do?
- What is the purpose of the code?

## Use case 1: Data collection and processing

**TL;DR** Gather data for use during training. Do any needed pre-processing.

- Collection: replay-collector
- Processing:
    - replay-parser
    - sim-core
    - `packages/training/training/bc/splits.py`

...

## Use case 2: Training runs

**TL;DR** Train the behavior-cloning (bc) model using curated expert replays. Produce model "checkpoints". Record useful metrics.

See `training/bc/README.md`, which directly came from an attempt to add more detail to this section.

## Use case 3: Gameplay

### "Real" gameplay (in the wild)

e.g. On the main server against human players, on the bot server, and so on. We have not yet done this.

### Eval

Evaluating the performance of a checkpoint in actual games (simulated). Often done over many games. Players in the eval match may be:

- Any number of models (specific checkpoint and inference config) that are "under evaluation"
- Other fixed reference points that have been used in previous evals (eg. checkpoints from the eval set, custom-bots )
- win-rates and a wide variety of other metrics are reported and anaylzed, as we attempt to understand the performance and capabilities of the model under evaluattion.

...

## Use case 4: Analysis

1. Looking through existing data and metrics
2. Frozen probes
4. Ad-hoc analysis scripts
    1. fq toolkit

...
