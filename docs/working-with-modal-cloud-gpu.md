# Working with Modal (cloud GPU runs)

Reference for running our training code on Modal's serverless GPU infrastructure: the mental model, the image-build patterns we use, and the gotchas worth knowing.

Official docs: [Modal guide root](https://modal.com/docs/guide) · [Image reference](https://modal.com/docs/reference/modal.Image) · [Pricing](https://modal.com/pricing).

## Mental model

Modal is **serverless functions, with GPUs**. Conceptually closer to AWS Lambda than to a rented VM — you don't have a persistent server. You define a function, decorate it with `@app.function(...)`, and Modal spins up a container with your image, runs the function, returns the result, and tears the container down. You're billed per second of container clock-time.

The important differences vs. Lambda:

| | Lambda | Modal |
|---|---|---|
| Max runtime | 15 minutes | 24 hours (longer with `--detach`) |
| GPUs | No | Yes — H100/A100/T4/L4/etc. per function |
| Image | ~250 MB layers, narrow base options | Arbitrary container images |
| Cold start | ~100ms–1s | ~5-30s typically (image-dependent) |
| Persistent state | Use S3/EFS as separate services | Modal Volumes mount into the container |
| Pricing granularity | 1ms | per-second of container clock-time |

The "long runtime + dedicated GPU + per-second billing" combo is what makes Modal good for our case. An overnight 8-hour H100 training run costs ~$32 and you don't pay for boot, debug, or idle time.

## Pricing reality

GPU clock-time dominates everything else by orders of magnitude. As of writing:

| Resource | Rate |
|---|---|
| H100 | $0.001097 / sec ≈ **$3.95 / hour** |
| A100 (80 GB) | $0.000694 / sec ≈ **$2.50 / hour** |
| A100 (40 GB) | $0.000583 / sec ≈ **$2.10 / hour** |
| L40S | $0.000542 / sec ≈ **$1.95 / hour** |
| A10 | $0.000306 / sec ≈ **$1.10 / hour** |
| T4 | $0.000164 / sec ≈ **$0.59 / hour** |
| CPU | $0.0000131 / core / sec ≈ $0.047 / core / hour |
| Memory | $0.00000222 / GiB / sec |
| Volume storage | $0.09 / GiB / month (first 1 TiB free) |
| Data transfer | Not on the pricing page — *appears* to be free, but unconfirmed |

Source: [modal.com/pricing](https://modal.com/pricing).

Modal's free tier gives $30/mo of credit. For comparison: our parsed corpus (~10 GB) sits inside the storage free tier ~100× over; iterating with smoke runs on a T4 costs sub-penny per run; a real H100 training run is the cost driver (~$15–40 per overnight run).

**Practical implication.** Don't sweat upload costs or storage. Do sweat GPU wall-clock — be deliberate about which run actually needs the H100 and which can validate on a T4.

## Run modes

A handful of ways to invoke a Modal function:

| Mode | Command | When |
|---|---|---|
| One-shot, tethered | `modal run script.py` | Smoke runs, dev iteration; you watch stdout |
| One-shot, detached | `modal run --detach script.py` | Long runs (training overnight); function survives your laptop sleeping |
| Deployed | `modal deploy app.py` then call by name/API | Services that need to be callable from elsewhere |
| Fan-out | `function.map(args_list)` | Parameter sweeps, parallel evaluations |

For training runs, **detached is the default** — kick off, walk away, check back. The Modal web UI shows live logs for detached runs; you can also tail JSONL files from the Volume mid-run.

## The image pattern we use

**Modal-blessed install path with full lockfile-driven reproducibility.** See the [Images guide](https://modal.com/docs/guide/images) and [Image reference](https://modal.com/docs/reference/modal.Image) for the underlying API. The pattern:

```python
import modal

image = (
    modal.Image.debian_slim(python_version="3.14")
    # 1. Heavy deps — installed from a fully-pinned requirements file
    .uv_pip_install(requirements=["<package>/modal_requirements.txt"])
    # 2. Package source (copied to /pkg)
    .add_local_file("<package>/pyproject.toml", "/pkg/pyproject.toml", copy=True)
    .add_local_dir(
        "<package>/<source>",
        "/pkg/<source>",
        copy=True,
        ignore=["__pycache__", "*.pyc"],
    )
    # 3. Install the package itself — its deps are already present
    .uv_pip_install("/pkg")
)
```

Why this shape:

- **[`debian_slim`](https://modal.com/docs/reference/modal.Image#debian_slim) is Modal's recommended base.** Supports Python 3.14.
- **[`uv_pip_install`](https://modal.com/docs/reference/modal.Image#uv_pip_install) is Modal's blessed install method.** Better cache behavior than running `uv pip install` via `run_commands`.
- **The `modal_requirements.txt` is generated** from the workspace's `uv.lock` via `tools/regen_modal_reqs.sh`. Single source of truth: our workspace lock. No drift.
- **Layer ordering matters for caching:** heavy + rarely-changing layers first (the torch install), light + frequently-changing layers last (our package source). When you edit `work.py` and re-run, only the last `uv_pip_install("/pkg")` layer is invalidated.

### What `modal_requirements.txt` looks like

A `--hash`-bearing pip-style requirements file with every direct + transitive dep pinned:

```
torch==2.12.0 \
    --hash=sha256:...
nvidia-cublas==13.1.1.3 ; sys_platform == 'linux' \
    --hash=sha256:...
sympy==1.14.0 \
    --hash=sha256:...
...
```

Generated by:
```bash
uv export --package <pkg> --format requirements-txt --no-dev --no-emit-project -o <pkg>/modal_requirements.txt
```

`--no-emit-project` excludes the workspace member itself (which is installed separately from its local path). `--no-dev` excludes dev-only deps.

### Workflow for adding/changing deps

1. Edit `<pkg>/pyproject.toml` (or the relevant workspace member)
2. `uv lock` (updates `uv.lock`)
3. `tools/regen_modal_reqs.sh` (regenerates all tracked `modal_requirements.txt` files from the lock)
4. Commit `uv.lock` + the regenerated requirements together

Future improvement: a pre-commit / CI check that fails if any `modal_requirements.txt` is stale vs. the lock.

## Native vs shell-out (function design)

When Modal calls into our code, two patterns are available:

- **Native** — Modal function does `from training.scripts.train_bc import run; run(args)`. Real Python imports, real tracebacks, type-safe.
- **Shell-out** — Modal function does `subprocess.run(["uv", "run", "python", "training/scripts/train_bc.py", ...])`. Treats the script as a black box CLI.

**We use the hybrid: native call with CLI-style args.** The Modal `local_entrypoint` takes Python-typed parameters, builds an `argparse.Namespace` via the script's existing parser, and passes it to `run(args)` inside the function. This gives us real tracebacks while keeping the local and cloud invocation shapes symmetric.

```python
@app.function(gpu="H100")
def train_remote(cli_args: list[str]):
    from training.scripts.train_bc import _build_arg_parser, run
    args = _build_arg_parser().parse_args(cli_args)
    run(args)

@app.local_entrypoint()
def cli(args: str):
    train_remote.remote(args.split())
```

## Volumes (state across runs)

The container's local filesystem is **ephemeral** — anything written there disappears when the container exits. Persistent state lives in [Modal Volumes](https://modal.com/docs/guide/volumes), which mount as a path inside the container.

```python
vol = modal.Volume.from_name("generals-corpus", create_if_missing=True)

@app.function(volumes={"/data": vol})
def train_remote(...):
    # /data is now a persistent shared filesystem
    ...
```

Typical patterns:

- **Read-only mount for input data** (parsed corpus): one-time upload via `modal volume put`, then every training run reads from the same mount.
- **Read-write mount for outputs** (checkpoints, JSONL, logs): training writes during the run; you pull artifacts back via `modal volume get` afterward.

Volumes are content-addressed and snapshotted daily for billing; deletes take up to four days to drop from the bill.

## Gotchas worth knowing

### [`add_local_dir`](https://modal.com/docs/reference/modal.Image#add_local_dir) copies *everything* by default

It does **not** respect `.gitignore` or `.dockerignore`. With `copy=True` it uploads the full directory tree into an image layer; with `copy=False` it does the same on every container cold start (worse!). For our repo, blindly calling `add_local_dir(REPO_ROOT, ...)` would have shoveled 19 GB (mostly the replay corpus in `replay-collector/data/`) up to Modal.

**Always be targeted.** Use `add_local_file` for single files, `add_local_dir` on the narrowest source directory, and the `ignore=[...]` parameter (dockerignore-style patterns) when you need to exclude `__pycache__` / `*.pyc` / etc.

### `uv_pip_install(requirements=[...])` takes a *local* path

The path is read from your laptop's filesystem at build-planning time and shipped as part of the install — not a path inside the image. If you also `add_local_file` the requirements file to the image, you're doing double-work (and the install step won't find it where you think).

### `add_local_dir` followed by `run_commands` works, but feels fragile

If you mix `.run_commands(...)` between `add_local_*` calls, the order is preserved but it's easy to author layers in the wrong sequence. Prefer `uv_pip_install` / Modal's high-level methods where possible — they handle ordering and caching better than raw shell commands.

### The `modal_entry.py` file itself is auto-mounted

You don't need to `add_local_file` your entrypoint script — Modal mounts it automatically. (Functions defined in other modules do need to be reachable in the image, of course.)

### Failed image builds cost essentially nothing

Modal only bills for successful image builds and function runs. If you're iterating on an image definition and hitting errors, you're not burning money. Iterate freely.

### Lockfile vs. pinning vs. ranges

Three reproducibility levels:

- `uv_pip_install("torch")` — resolved at build time, drifts over time.
- `uv_pip_install("torch==2.12.0")` — direct dep pinned, transitive deps still drift.
- `uv_pip_install(requirements=["modal_requirements.txt"])` — every dep including transitive pinned, with hashes.

Always use the last one for anything that ships to Modal. The pinned requirements file is what guarantees the cloud install matches what you tested locally.

## Quick reference

| Need | Method |
|---|---|
| Run on CPU, foreground | `uv run modal run <entry>.py` |
| Run on GPU | Add `gpu="H100"` (or T4/A100/etc.) to `@app.function(...)` |
| Run detached | `uv run modal run --detach <entry>.py` |
| Pass args to remote | `local_entrypoint` takes Python-typed params, calls `.remote(...)` on the function |
| Persistent storage | `modal.Volume.from_name(...)`, mount via `volumes={...}` |
| Upload to Volume | `modal volume put <vol-name> <local-path> <remote-path>` |
| Download from Volume | `modal volume get <vol-name> <remote-path> <local-path>` |
| Inspect a running app | https://modal.com/apps (web UI) |
