"""Print Python + torch + MPS availability. Sanity check after `uv sync`."""

import platform
import sys

import torch


def main() -> None:
    print(f"python:        {sys.version.split()[0]} ({platform.machine()})")
    print(f"torch:         {torch.__version__}")
    print(f"mps.available: {torch.backends.mps.is_available()}")
    print(f"mps.built:     {torch.backends.mps.is_built()}")
    print(f"cuda.available: {torch.cuda.is_available()}")


if __name__ == "__main__":
    main()
