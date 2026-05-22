import platform
import sys


def hello() -> dict:
    import torch

    x = torch.arange(10, dtype=torch.float32)
    return {
        "msg": "hello from cloud_train_poc",
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "tensor_sum": float(x.sum().item()),
        "cuda_available": torch.cuda.is_available(),
    }
