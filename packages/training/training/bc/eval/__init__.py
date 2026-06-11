"""Per-epoch validation pass for BC training.

`run` drives the pass (loader, loop, summary assembly); `metrics` holds the
diagnostic meters that ride it. This facade re-exports the public surface —
external callers import from `training.bc.eval`, package internals import
from the specific module.
"""

from training.bc.eval.metrics import ActionDistMeter, PolicyEntropyMeter
from training.bc.eval.run import run_val


__all__ = ["ActionDistMeter", "PolicyEntropyMeter", "run_val"]
