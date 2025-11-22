# fastcma

[![CI](https://github.com/Dicklesworthstone/fast_cmaes/actions/workflows/build-wheels.yml/badge.svg)](https://github.com/Dicklesworthstone/fast_cmaes/actions/workflows/build-wheels.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](#license)
[![PyPI](https://img.shields.io/pypi/v/fast-cmaes.svg)](https://pypi.org/project/fast-cmaes/)
[![Python](https://img.shields.io/badge/Python-3.12%20--%203.14-blue)](#installation-python)
[![PyPI version](https://img.shields.io/pypi/v/fast-cmaes.svg)](https://pypi.org/project/fast-cmaes/)
[![Rust](https://img.shields.io/badge/Rust-nightly-orange)](#rust-usage-library)

Hyper-optimized CMA-ES in Rust with a first-class Python experience. SIMD, rayon, deterministic seeds, vectorized objectives, restarts, constraints, and a Rich-powered TUI — all while keeping the Rust core available for native use. Published to PyPI as `fast-cmaes` (module name: `fastcma`). Latest release: **0.1.4**.

## Table of contents
- [Why CMA-ES](#why-cma-es)
- [Architecture (Mermaid)](#architecture-mermaid)
- [Features](#features)
- [Installation (Python)](#installation-python)
- [Quickstart (Python)](#quickstart-python)
- [Vectorized & Constraints](#vectorized--constraints)
- [Rust usage](#rust-usage-library)
- [Demos & TUI](#demos--visualization)
- [Baselines & Benchmarks](#baselines--benchmarks)
- [Performance choices](#performance-considerations)
- [Feature flags](#feature-flags)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Why CMA-ES
- Derivative-free, handles noisy/non-convex landscapes.
- Adapts step size (sigma) and covariance to follow curved valleys.
- Parallel-friendly: candidate evaluations are embarrassingly parallel.

## Architecture (Mermaid)
```mermaid
flowchart LR
    subgraph Python API
    A[fmin / fmin_vec / CMAES class]
    B[Constraints & restarts]
    C[Naive baseline (pure Python)]
    end

    subgraph Rust Core (src/lib.rs)
    D[Ask/Tell loop]
    E[Covariance update (full/diag)]
    F[Sigma adaptation]
    G[SIMD dot + rayon fitness]
    H[Deterministic seeds]
    end

    subgraph Tests & Demos
    I[Benchmarks: sphere, rosenbrock, rastrigin, ackley, schwefel, griewank]
    J[Rich TUI demo]
    K[Python smoke]
    end

    A --> D
    A --> B
    B --> D
    D --> E --> F --> D
    D --> G
    D --> H
    I --> D
    J --> A
    K --> A
    C --> A
```

## Features
- **Python-first API**: `fmin`, `fmin_vec`, constrained, restart modes, and a `CMAES` class.
- **SIMD + rayon**: portable_simd accelerates dot products; rayon parallelizes fitness calls.
- **Full/Diagonal covariance**: switch via `covariance_mode`.
- **Deterministic seeds**: `new_with_seed` + `test_utils` for reproducible runs and benchmarks.
- **Pure-Python baseline**: `fastcma.cma_es` for speed comparisons.
- **Rich TUI**: live, colorful CMA-ES progress view.
- **Cross-platform wheels**: CI builds for Linux/macOS/Windows, Python 3.12–3.14.

## Installation (Python)
Fastest path (PyPI):
```bash
python -m pip install fast-cmaes==0.1.4  # installs module `fastcma`
```

Build locally (needed only if hacking on Rust):
```bash
python -m pip install maturin
maturin develop --release

# Optional: NumPy fast paths
maturin develop --release --features numpy_support

# Optional: LAPACK eigen backend (requires a Fortran toolchain)
maturin develop --release --features eigen_lapack

# Demo extras (Rich TUI)
python -m pip install .[demo]
```

One-liner setup + Rich TUI demo (auto-installs nightly Rust, uv, venv, builds, runs):
```bash
./scripts/setup_and_demo.sh
```

## Quickstart (Python)
```python
from fastcma import fmin
from fastcma_baseline import benchmark_sphere

def sphere(x):
    return sum(v*v for v in x)

xmin, es = fmin(sphere, [0.5, -0.2, 0.8], sigma=0.3, maxfevals=4000, ftarget=1e-12)
print("xmin", xmin)

# Pure-Python baseline
print(benchmark_sphere(dim=20, iters=120))
```

## Vectorized & Constraints
```python
from fastcma import fmin_vec

def sphere_vec(X):
    return [sum(v*v for v in x) for x in X]

xmin, _ = fmin_vec(sphere_vec, [0.4, -0.1, 0.3], sigma=0.25, maxfevals=3000)
```

Constrained run:
```python
from fastcma import fmin_constrained

def sphere(x): return sum(v*v for v in x)
constraints = {"lower_bounds": [-1,-1,-1], "upper_bounds": [1,1,1]}
xmin, _ = fmin_constrained(sphere, [0.5,0.5,0.5], 0.3, constraints)
```

## Rust usage (library)
```rust
use fastcma::{optimize_rust, CovarianceModeKind};

let (xmin, _state) = optimize_rust(
    vec![0.5, -0.2, 0.8],
    0.3,
    None,
    Some(4000),
    Some(1e-12),
    CovarianceModeKind::Full,
    |x| x.iter().map(|v| v*v).sum()
);
println!("xmin = {:?}", xmin);
```

## Demos & visualization
- `examples/python_quickstart.py` – minimal sphere + vectorized demo.
- `examples/python_benchmarks.py` – Rust vs naive Python on sphere; naive on Rastrigin.
- `examples/rich_tui_demo.py` – Rich TUI streaming sigma/fbest/evals while minimizing Rosenbrock.

One-shot setup + demo runner:
```bash
./scripts/setup_and_demo.sh
```
What it does: ensures nightly Rust, creates a uv venv on Python 3.13, installs maturin + demo extras, builds the extension, and launches the Rich TUI.

Run the TUI with uv + Python 3.13:
```bash
uv venv --python 3.13
uv pip install .[demo]
uv run python examples/rich_tui_demo.py
```

## Baselines & benchmarks
- Pure Python baseline: `fastcma_baseline.cma_es`, `fastcma_baseline.benchmark_sphere` (see `python/fastcma_baseline/naive_cma.py`).
- Integration benchmarks (fixed seeds): sphere, Rosenbrock, Rastrigin, Ackley, Schwefel, Griewank in `tests/benchmarks.rs`.
- Hard-suite (20 classic tough functions, seeded, higher dims): Zakharov, Levy, Dixon-Price, Powell, Styblinski–Tang, Bohachevsky, Bukin6, Dropwave, Alpine N.1, Elliptic, Salomon, Quartic, Schwefel 1.2/2.22, Bent Cigar, Rastrigin 10D, Ackley 10D, Griewank 10D, Rosenbrock 6D, Sum Squares 12D in `tests/hard_benchmarks.rs`.
- Very-hard (ignored by default, ~20 more high-dim/fractal/ill-conditioned cases such as Katsuura, Weierstrass, Schwefel 2.26 @ 30D, HappyCat/HGBat, Expanded Schaffer F6/F7, Discus, Different Powers, 30D Rastrigin/Ackley/Rosenbrock/Griewank/Elliptic) in the same file; run with `cargo test --test hard_benchmarks -- --ignored`.
- Rich TUI demo for live insight.

## Performance considerations
- **SIMD** for dot products; **rayon** for parallel ask/tell evaluations.
- **Lazy eigensystem updates** to reduce eigen decompositions.
- **Diagonal covariance** option for higher dimensions / speed.
- **Determinism**: seeded RNG to make tests non-flaky and benchmarks comparable.
- **Restart helper** (`test_utils::run_with_restarts`) to escape local minima without huge budgets.

## Feature flags
- `numpy_support`: NumPy array support in vectorized objectives.
- `eigen_lapack`: LAPACK eigen backend.
- `test_utils`: expose deterministic helpers externally.
- `demo`: pulls in `rich` for the TUI.

## Testing
- Rust quick suite: `cargo test`
- Extended hard benchmarks (20 classic functions): `cargo test --test hard_benchmarks` (takes ~90s)
- Very-hard (ignored) suite (high-dim/fractal/ill-conditioned, +~20 cases): `cargo test --test hard_benchmarks -- --ignored`
- CI runs the core suite on every push/PR and executes the ignored very-hard suite on GitHub Actions workers (job: `very-hard`) so heavy compute stays off your local machine; expect PR CI to take a few minutes longer because of that job and wheel matrix builds.
- Python smoke (local wheel): `pytest tests/python_smoke.py`
- CI: GitHub Actions builds wheels on nightly Rust; runs smoke tests; publishes to PyPI on tags when `PYPI_API_TOKEN` is set; PyPI smoke follows publish.

## Contributing
- Nightly Rust required (see `rust-toolchain.toml`).
- Please include failing cases or perf comparisons in issues/PRs.

## License
MIT (c) 2025 Jeffrey Emanuel
