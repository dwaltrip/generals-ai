"""Per-epoch validation pass for BC training.

`run` drives the in-training pass (summary assembly); `metrics` holds the
diagnostic meters that ride it; `dump` holds the shared val forward loop
(`iter_val_forward`), the per-frame stratified record capture, the offline
`capture_val_frames` driver, and artifact IO. This facade re-exports the public
surface — external callers import from `training.bc.eval`, package internals
import from the specific module.
"""

from training.bc.eval.dump import (
    FrameRecordCapture,
    capture_val_frames,
    dump_path,
    save_dump,
)
from training.bc.eval.metrics import ActionDistMeter, PolicyEntropyMeter
from training.bc.eval.run import run_val


__all__ = [
    "ActionDistMeter",
    "FrameRecordCapture",
    "PolicyEntropyMeter",
    "capture_val_frames",
    "dump_path",
    "run_val",
    "save_dump",
]
