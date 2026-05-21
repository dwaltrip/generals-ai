from pathlib import Path


# tmp/ at the subproject root, where log files land.
TMP_DIR = Path(__file__).resolve().parent.parent.parent / "tmp"


def fmt_duration(seconds: int) -> str:
    if seconds < 60:
        return f"~{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"~{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"~{h}h {m:02d}m {s:02d}s"
