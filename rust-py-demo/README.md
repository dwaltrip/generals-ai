# rust-py-demo — Notes on Rust-in-Python with maturin + uv

A working reference for integrating Rust into a Python project in this repo. The package itself is a tiny standalone demo (`add`, `sum_squares`, `mean_of_squares`); the real value is the patterns and gotchas captured here. Intended audience: anyone (human or assistant) bootstrapping a real Rust hot loop — e.g. for the replay sim.

## Toolchain

- **PyO3** — Rust crate providing the bindings via `#[pyfunction]` / `#[pymodule]`.
- **maturin** — build backend; compiles the Rust crate into a Python wheel or editable install.
- **uv** — env/package manager (already used throughout this repo).

## Layout (mixed Python + Rust package)

```
rust-py-demo/
├── Cargo.toml                 # Rust manifest
├── pyproject.toml             # build backend + uv config
├── src/lib.rs                 # Rust — exports a `_native` pymodule
└── python/rust_py_demo/
    ├── __init__.py            # public Python API; re-exports + wrappers
    ├── _native.pyi            # type stubs for the Rust functions
    └── py.typed               # PEP 561 marker (REQUIRED for IDE types)
```

The Rust extension is named `_native` and lives as a private submodule. Consumers do `import rust_py_demo` and use the Pythonic top-level API; the Rust internals are an implementation detail. This mirrors what pydantic-core, polars, ruff, and cryptography do.

**Alternative — pure Rust extension** (no `python/` directory): `src/lib.rs` exports the module directly and consumers import it raw. Simpler, suitable when the Rust *is* the API and there's no Python ergonomics layer to add.

## pyproject.toml essentials

```toml
[tool.maturin]
module-name = "rust_py_demo._native"   # private submodule
python-source = "python"
features = ["pyo3/extension-module"]

[tool.uv]
cache-keys = [
    { file = "pyproject.toml" },
    { file = "Cargo.toml" },
    { file = "**/*.rs" },
]
```

The `cache-keys` block is **critical**. Without it, `uv run` reinstalls the project from uv's wheel cache on every invocation and clobbers any fresh `maturin develop` build. With it, uv invalidates its cached wheel whenever any `.rs` file changes and rebuilds via the maturin backend automatically. Endorsed in [astral-sh/uv#11390](https://github.com/astral-sh/uv/issues/11390) and [PyO3/maturin#2314](https://github.com/PyO3/maturin/issues/2314).

## Dev loop

```bash
# from rust-py-demo/
uv run python demo.py                    # debug build, auto-rebuilds on Rust changes
uv run maturin develop --release         # release build (for perf benchmarks)
```

Cargo handles incremental compilation transparently — small edits rebuild in well under a second after the first compile (which downloads + builds all deps, ~5–15s on a clean checkout).

**Debug vs release matters a lot.** Debug Rust builds can be 10–100× slower than release. Always use `--release` for any perf measurement; the default `uv run` flow gives you debug builds.

## Python version compat

PyO3 0.22 supports Python 3.10–3.13. Python 3.14 is **not** supported by 0.22. Pin the venv accordingly:

```bash
uv venv --python 3.13
```

Bump PyO3 (and re-check support) when starting a new package; check [PyO3 release notes](https://github.com/PyO3/pyo3/releases).

## Type stubs for IDE / type checkers

Rust extensions don't carry type info into Python. Pylance/mypy see the functions as untyped unless you ship:

1. A `.pyi` stub file next to the `.so` (e.g. `python/rust_py_demo/_native.pyi`)
2. An empty `py.typed` marker file in the package directory — **without this**, type checkers ignore the `.pyi` regardless. PEP 561.

Hand-writing stubs is tedious but worth it for any module used across the codebase.

## VS Code setup (this repo)

The repo is a uv workspace where members share the root `.venv`. rust-py-demo is intentionally **not** a workspace member — it lives standalone with its own venv as a learning artifact. To make Pylance work with this layout, the repo includes a multi-root `generals-ai.code-workspace` file declaring both the root folder and `rust-py-demo` as separate roots. Open via "File → Open Workspace from File" (not "Open Folder"). The Python extension auto-detects each folder's venv.

If/when a Rust crate becomes a real workspace member, the multi-root entry can be removed — it'd use the shared venv like the others.

## When integrating Rust into the real project

Decisions to make at that point:

1. **Workspace member or standalone?** For a real integration, almost certainly a workspace member — matches the existing replay-collector / replay-parser pattern, uses the shared venv, simpler deps.

2. **Where does the crate live?**
   - Own subproject (`sim-core/` alongside the others) — best when multiple Python subprojects consume it.
   - Nested inside one subproject (`replay-parser/rust/`) — best when only one consumer.

3. **Pure Rust or mixed layout?** Pure for a tight 1–2-function utility. Mixed when you want a Python API surface with validation, helpers, fallbacks, or graceful errors over an austere Rust core.

4. **What goes in Rust?**
   - **Worth it:** tight numeric loops, array math (use the `numpy` crate for zero-copy ndarray views), bitset/state-machine inner loops, anything currently bottlenecked by Python interpreter overhead.
   - **Not worth it:** I/O-bound code, glue code, anything dominated by allocating Python objects across the boundary (the FFI cost dominates).

5. **GIL release for parallelism.** If the hot loop is CPU-bound pure Rust, wrap it in `py.allow_threads(|| { ... })` to let other Python threads run concurrently. Required to actually parallelize across threads.

6. **Data shapes at the boundary.** PyO3 converts scalars, strings, `Vec<T>`, tuples, dicts automatically. For numpy arrays, use the `numpy` crate (`PyReadonlyArray` for zero-copy reads). For custom Python classes ↔ Rust structs, `#[pyclass]` (more boilerplate, fine when needed).

## References

- [PyO3 user guide](https://pyo3.rs/) — bindings reference
- [maturin docs](https://www.maturin.rs/) — layout, configuration, build options
- [maturin project layout guide](https://www.maturin.rs/project_layout) — pure vs mixed layout
- [uv cache concepts](https://docs.astral.sh/uv/concepts/cache/) — `cache-keys` reference
- [PyO3 ↔ maturin ↔ uv interaction](https://github.com/PyO3/maturin/issues/2314)
