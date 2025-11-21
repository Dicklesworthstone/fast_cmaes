# fastcma

Hyper-optimized CMA-ES in Rust with a first-class Python experience. SIMD, rayon, deterministic seeds, vectorized objectives, restart strategies, and constraint handling — all accessible from Python while keeping the Rust core available for native use.

## Why CMA-ES (quick refresher)
- **What it is:** Covariance Matrix Adaptation Evolution Strategy — a derivative-free, stochastic optimizer that adapts step size and search covariance to follow curved landscapes.
- **Why it’s great:** Works on noisy, non-convex, non-separable problems; needs only objective evaluations; naturally parallelizable.
- **Core loop:** sample candidates → evaluate → rank → update mean / evolution paths / covariance → adapt sigma → repeat until tolerance/target/fevals.

## Project architecture
- **Rust core** (`src/lib.rs`): portable_simd for fast dot products, rayon for parallel fitness evaluation, optional LAPACK eigen backend (`eigen_lapack` feature), full/diagonal covariance modes, deterministic `new_with_seed` for tests and reproducible runs.
- **Python binding** (PyO3, maturin): Python surface mirrors `purecma`-style APIs: `fmin`, `fmin_vec`, constrained and restart helpers. Wheels built for Py3.9–3.12 on Linux/macOS/Windows.
- **Deterministic test utilities** (`test_utils`): seeded helpers, multiseed sweeps, restart helper with Gaussian perturbations.
- **Naive baseline** (`python/naive_cma.py`): pure-Python CMA-ES for speed comparisons (`fastcma.cma_es`, `fastcma.benchmark_sphere`).
- **Examples**: `examples/python_quickstart.py`, `examples/python_benchmarks.py`, plus Rust-side examples live in the library tests/benchmarks.
- **CI**: `.github/workflows/build-wheels.yml` builds wheels on nightly Rust (portable_simd), runs smoke tests, and can publish to PyPI with `PYPI_API_TOKEN`.

## Installation (Python)
```bash
python -m pip install maturin  # if building locally
maturin develop --release              # or: pip install fastcma (when published)

# Optional: NumPy fast paths
maturin develop --release --features numpy_support

# Optional: LAPACK eigen backend
maturin develop --release --features eigen_lapack
```

## Quickstart (Python)
```python
from fastcma import fmin

def sphere(x):
    return sum(v*v for v in x)

xmin, es = fmin(sphere, [0.5, -0.2, 0.8], sigma=0.3, maxfevals=4000, ftarget=1e-12)
print("xmin", xmin)
```

### Vectorized objectives
```python
from fastcma import fmin_vec

def sphere_vec(X):
    return [sum(v*v for v in x) for x in X]

xmin, es = fmin_vec(sphere_vec, [0.4, -0.1, 0.3], sigma=0.25, maxfevals=3000)
print("xmin", xmin)
```

### Rust vs. pure-Python baseline
```python
from fastcma import benchmark_sphere

print(benchmark_sphere(dim=20, iters=120))
```

### Ready-to-run scripts
- `examples/python_quickstart.py` – minimal sphere example + vectorized demo.
- `examples/python_benchmarks.py` – compares Rust vs naive Python on sphere and runs naive CMA-ES on Rastrigin.
- `examples/rich_tui_demo.py` – Rich-powered TUI that live-visualizes CMA-ES optimizing Rosenbrock.

## Rust usage (library)
```rust
use fastcma::{optimize_rust, CovarianceModeKind};

let (xmin, state) = optimize_rust(vec![0.5, -0.2, 0.8], 0.3, None, Some(4000), Some(1e-12), CovarianceModeKind::Full, |x| x.iter().map(|v| v*v).sum());
println!("xmin = {:?}", xmin);
```

## Performance considerations
- **SIMD dot products** (portable_simd) and **rayon** parallel fitness evaluation.
- **Lazy eigensystem updates** to avoid unnecessary decompositions.
- **Full vs diagonal covariance**: choose via `covariance_mode="diag"` for speed on high dimensions.
- **Determinism**: `new_with_seed` plus seeded test utilities make benchmarks repeatable.
- **Restart helper**: `test_utils::run_with_restarts` supports simple restarted runs without exploding eval budgets.

## Feature flags
- `numpy_support`: enable NumPy array returns for vectorized objectives.
- `eigen_lapack`: use LAPACK-based eigen solver (via nalgebra-lapack) instead of pure Rust.
- `test_utils`: expose deterministic constructors/helpers to downstream tests.

## Benchmarks & tests
- Integration benchmarks (sphere, Rosenbrock, Rastrigin, Ackley, Schwefel, Griewank) in `tests/benchmarks.rs` use fixed seeds for stability.
- Python smoke test: `tests/python_smoke.py` ensures wheel import + basic optimization + baseline.
- Run everything: `cargo test` (Rust) and `pytest tests/python_smoke.py` (Python).

## Rich TUI demo (Python 3.13 with uv)
```bash
uv venv --python 3.13
uv pip install .[demo]
uv run python examples/rich_tui_demo.py
```
Streams live CMA-ES progress (sigma, fbest, evals) in color while minimizing Rosenbrock.

## Demos & visualization
- Scripts above print timing and objective values for Rust vs Python baselines. You can drop the results into a notebook for plots; outputs are plain dicts for easy pandas ingestion.

## Design choices (high level)
- **Portable SIMD + rayon**: better CPU utilization without custom C/AVX intrinsics.
- **NALgebra covariance / eigen**: leverages battle-tested linear algebra; LAPACK optional for parity with reference implementations.
- **Configurable covariance (full/diag)**: switch depending on dimension and conditioning.
- **Deterministic testing**: seeded RNG to make convergence tests non-flaky.
- **Python-first surface**: API mirrors familiar `purecma` functions while keeping Rust ergonomic.

## Contributing
- Tests: `cargo test` (Rust) and `pytest tests/python_smoke.py` (Python).
- Nightly Rust required (see `rust-toolchain.toml`).
- Issues/PRs welcome; please include failing cases or perf comparisons when relevant.

## License
MIT
