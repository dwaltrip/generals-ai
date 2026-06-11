"""
Behavioral-cloning model: trunk (Pyramid Module) + policy / pass / value heads.

`trunk` holds the DeepNash U-Net and its residual blocks, `heads/` one module
per head, `bc_model` the composition root. This facade re-exports the public
surface — external callers import from `training.bc.model`, package internals
import from the specific module.
"""

from training.bc.model.bc_model import BCModel
from training.bc.model.heads.pass_head import PassHead
from training.bc.model.heads.policy import MASK_NEG, PolicyHead, flatten_policy_logits
from training.bc.model.heads.value import ValueHead
from training.bc.model.trunk import PyramidModule


__all__ = [
    "MASK_NEG",
    "BCModel",
    "PassHead",
    "PolicyHead",
    "PyramidModule",
    "ValueHead",
    "flatten_policy_logits",
]
