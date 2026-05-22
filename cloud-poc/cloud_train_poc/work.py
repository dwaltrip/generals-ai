import platform
import sys


def hello() -> dict:
    return {
        "msg": "hello from cloud_train_poc",
        "python": sys.version,
        "platform": platform.platform(),
    }
