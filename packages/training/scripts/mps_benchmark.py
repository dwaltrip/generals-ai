"""MPS benchmark — placeholder U-Net forward + backward + optimizer step on
synthetic input. Settles whether MPS speedup over CPU is pleasant enough
to use for the spike, and surfaces any MPS-fallback ops loudly.

The model here is NOT the real Pyramid Module — it's a U-Net-shaped
placeholder using the same ops (Conv2d, ConvTranspose2d, GroupNorm,
ReLU) and roughly the same width spec (128/128/160), so timing is
representative.

We explicitly unset PYTORCH_ENABLE_MPS_FALLBACK before running; without
that env var, unsupported MPS ops raise NotImplementedError. That's the
signal we want — silent CPU fallback would defeat the benchmark.

Run from repo root:
    uv run python training/scripts/mps_benchmark.py
    uv run python training/scripts/mps_benchmark.py --warmup 10 --measure 50

Note: The example config of 10, 50 currentlyruns in a bit over 1 minute.
"""

import argparse
import time

import torch
from torch import nn

from training.shared.device import disable_mps_fallback


IN_CHANNELS = 96
SPATIAL = 32
BATCH = 64


def conv_block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.GroupNorm(1, out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        nn.GroupNorm(1, out_c),
        nn.ReLU(inplace=True),
    )


class PlaceholderUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        c1, c2, c3 = 128, 128, 160
        self.stem = nn.Conv2d(IN_CHANNELS, c1, kernel_size=3, padding=1)
        self.enc1 = conv_block(c1, c1)
        self.down1 = nn.Conv2d(c1, c2, kernel_size=2, stride=2)
        self.enc2 = conv_block(c2, c2)
        self.down2 = nn.Conv2d(c2, c3, kernel_size=2, stride=2)
        self.bottle = conv_block(c3, c3)
        self.up1 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec1 = conv_block(c2 + c2, c2)
        self.up2 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec2 = conv_block(c1 + c1, c1)
        self.head = nn.Conv2d(c1, 8, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.stem(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(self.down1(e1))
        b = self.bottle(self.down2(e2))
        x1 = self.dec1(torch.cat([self.up1(b), e2], dim=1))
        x2 = self.dec2(torch.cat([self.up2(x1), e1], dim=1))
        return self.head(x2)


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def benchmark(device: str, warmup: int, measure: int) -> dict[str, float]:
    torch.manual_seed(0)
    model = PlaceholderUNet().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x = torch.randn(BATCH, IN_CHANNELS, SPATIAL, SPATIAL, device=device)

    for _ in range(warmup):
        optim.zero_grad()
        loss = model(x).pow(2).mean()
        loss.backward()
        optim.step()
    sync(device)

    t0 = time.perf_counter()
    for _ in range(measure):
        optim.zero_grad()
        loss = model(x).pow(2).mean()
        loss.backward()
        optim.step()
    sync(device)
    elapsed = time.perf_counter() - t0

    return {
        "time": elapsed,
        "iters": measure,
        "iters_per_sec": measure / elapsed,
        "ms_per_iter": 1000 * elapsed / measure,
    }


def report(device: str, stats: dict[str, float]) -> None:
    print(
        f"  {device:5}: {stats['iters_per_sec']:6.2f} iter/s  "
        f"({stats['ms_per_iter']:6.2f} ms/iter, "
        f"{int(stats['iters'])} iters in {stats['time']:.2f}s)"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--measure", type=int, default=100)
    args = parser.parse_args()

    disable_mps_fallback()

    n_params = sum(p.numel() for p in PlaceholderUNet().parameters())
    print(
        f"PlaceholderUNet  in_ch={IN_CHANNELS} HW={SPATIAL} batch={BATCH} "
        f"params={n_params:,}"
    )
    print(f"warmup={args.warmup} measure={args.measure}")
    print()

    print("Benchmarking CPU ...")
    cpu_stats = benchmark("cpu", args.warmup, args.measure)
    report("cpu", cpu_stats)

    if not torch.backends.mps.is_available():
        print("MPS not available — skipping.")
        return

    print("Benchmarking MPS ...")
    try:
        mps_stats = benchmark("mps", args.warmup, args.measure)
    except Exception as e:
        print(f"  MPS failed: {type(e).__name__}: {e}")
        return
    report("mps", mps_stats)

    ratio = mps_stats["iters_per_sec"] / cpu_stats["iters_per_sec"]
    print(f"\nMPS / CPU throughput ratio: {ratio:.2f}x")


if __name__ == "__main__":
    main()
