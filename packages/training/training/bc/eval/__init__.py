"""Per-epoch validation pass for BC training.

`run` drives the pass (loader, loop, summary assembly); `metrics` holds the
diagnostic meters that ride it; `dump` holds the per-frame stratified record
capture + artifact IO shared with the offline analysis harness. This facade
re-exports the public surface — external callers import from
`training.bc.eval`, package internals import from the specific module.
"""

from training.bc.eval.dump import FrameRecordCapture, dump_path, save_dump
from training.bc.eval.metrics import ActionDistMeter, PolicyEntropyMeter
from training.bc.eval.run import run_val


__all__ = [
    "ActionDistMeter",
    "FrameRecordCapture",
    "PolicyEntropyMeter",
    "dump_path",
    "run_val",
    "save_dump",
]
