"""Feasibility probe: can py-spy attach inside a Modal container?

Result (2026-06-06): NO. Every configuration — self+subprocesses, parent->child,
and --nonblocking — fails with `Permission denied (os error 13)` reading the
target's memory. ptrace_scope is 1, but parent->child (which scope=1 permits)
fails too, and --nonblocking uses process_vm_readv rather than ptrace — so it's
the Modal sandbox (gVisor) denying cross-process memory reads at the syscall
level, not YAMA. This rules out py-spy *and* the wrapper-launch fallback, which
is why the obs-pipeline blind spot is profiled in-process (cProfile inside the
DataLoader worker) instead. Archived as the record behind that call.

CPU-only — sandbox behavior isn't GPU-specific.
Run: `uv run modal run training/scripts/_archive/pyspy_probe.py`.
"""

import modal

image = modal.Image.debian_slim(python_version="3.14").pip_install("py-spy")
app = modal.App("pyspy-probe", image=image)


@app.function()
def probe() -> dict:
    import os
    import subprocess
    import sys
    import time

    # A separate, recognizable child process spinning in `busyloop`.
    busy_src = "def busyloop():\n x=0\n while True: x=(x+1)%1000000\nbusyloop()"
    child = subprocess.Popen([sys.executable, "-c", busy_src])
    time.sleep(1.0)  # let it start spinning

    def run(label: str, *args: str) -> dict:
        try:
            p = subprocess.run(["py-spy", *args], capture_output=True, text=True, timeout=60)
            return {
                "rc": p.returncode,
                "saw_busyloop": "busyloop" in p.stdout,
                "stdout_head": p.stdout[:500],
                "stderr_tail": p.stderr[-400:],
            }
        except Exception as exc:
            return {"error": repr(exc)}

    me = str(os.getpid())
    results = {
        "py_spy_version": run("version", "--version"),
        "ptrace_scope": _read("/proc/sys/kernel/yama/ptrace_scope"),
        "self+subprocesses (blocking)": run("a", "dump", "--pid", me, "--subprocesses"),
        "self+subprocesses (nonblocking)": run(
            "b", "dump", "--pid", me, "--subprocesses", "--nonblocking"
        ),
        "child (parent->child)": run("c", "dump", "--pid", str(child.pid)),
    }
    child.terminate()
    return results


def _read(path: str) -> str:
    try:
        with open(path) as fp:
            return fp.read().strip()
    except OSError as exc:
        return f"(absent: {exc})"


@app.local_entrypoint()
def main() -> None:
    import json

    print(json.dumps(probe.remote(), indent=2))
