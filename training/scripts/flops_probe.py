#!/usr/bin/env -S uv run python
"""Diagnostic: report BCModel parameter count + fwd / fwd+bwd FLOPs per sample.

Used to answer "did my changes to the model alter compute cost?" and as the
ground-truth input to MFU calculations. Cheap to run (one fwd + one bwd at
batch=1). No CUDA required — runs on CPU.

Sample output:

    params: 4,185,442 (4.19M)
    input shape per sample: [C=96, H=32, W=32]

    forward FLOPs (B=1): 3,599,188,224 (3.599 GFLOPs)
    fwd+bwd FLOPs (B=1): 10,571,072,256 (10.571 GFLOPs)
    bwd / fwd ratio:     1.94x

    fwd+bwd FLOPs (B=8): 84,568,578,048 (84.569 GFLOPs)
    per-sample at B=8:   10.571 GFLOPs/sample
"""

import argparse

import torch

from bc.constants import H_PADDED, W_PADDED
from bc.model import BCModel
from bc.model_config import ModelConfig
from shared.perf import measure_total_flops


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--value-head", choices=("direct", "pyramid"), default="direct",
        help="Value-head variant to probe.",
    )
    args = parser.parse_args()

    model = BCModel(ModelConfig(value_head_variant=args.value_head))
    in_ch = model.cfg.in_ch
    print(f"value_head variant: {args.value_head}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,} ({n_params / 1e6:.2f}M)")
    print(f"input shape per sample: [C={in_ch}, H={H_PADDED}, W={W_PADDED}]")

    # Forward-only FLOPs (eval mode + no backward). Smaller number; useful
    # for inference-cost framing, and for the bwd/fwd ratio sanity check.
    model.eval()
    x = torch.zeros(1, in_ch, H_PADDED, W_PADDED)
    # Worst-case fully-used grid for FLOPs counting.
    vm1 = torch.ones(1, 1, H_PADDED, W_PADDED, dtype=torch.bool)
    with torch.no_grad():
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc:
            _ = model(x, vm1)
        fwd_flops = fc.get_total_flops()
    print(f"\nforward FLOPs (B=1): {fwd_flops:,} ({fwd_flops / 1e9:.3f} GFLOPs)")

    # Forward + backward: this is the number that goes into MFU. Surrogate
    # loss is sum of all three head outputs so backward touches every path.
    model.train()
    def fwd_bwd_b1() -> torch.Tensor:
        x = torch.zeros(1, in_ch, H_PADDED, W_PADDED)
        vm = torch.ones(1, 1, H_PADDED, W_PADDED, dtype=torch.bool)
        out = model(x, vm)
        return out["policy_logits"].sum() + out["pass_logit"].sum() + out["value_logits"].sum()

    fb_flops = measure_total_flops(fwd_bwd_b1)
    print(f"fwd+bwd FLOPs (B=1): {fb_flops:,} ({fb_flops / 1e9:.3f} GFLOPs)")
    print(f"bwd / fwd ratio:     {(fb_flops - fwd_flops) / fwd_flops:.2f}x")

    # Linearity sanity check at a larger batch.
    model.zero_grad()
    def fwd_bwd_b8() -> torch.Tensor:
        x = torch.zeros(8, in_ch, H_PADDED, W_PADDED)
        vm = torch.ones(8, 1, H_PADDED, W_PADDED, dtype=torch.bool)
        out = model(x, vm)
        return out["policy_logits"].sum() + out["pass_logit"].sum() + out["value_logits"].sum()

    b8_flops = measure_total_flops(fwd_bwd_b8)
    print(f"\nfwd+bwd FLOPs (B=8): {b8_flops:,} ({b8_flops / 1e9:.3f} GFLOPs)")
    print(f"per-sample at B=8:   {b8_flops / 8 / 1e9:.3f} GFLOPs/sample")


if __name__ == "__main__":
    main()
