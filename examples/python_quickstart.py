"""Minimal Python quickstart for fastcma.

Run with:

    pip install fastcma  # from PyPI once published, or `maturin develop`
    python examples/python_quickstart.py
"""

from pprint import pprint

import fastcma
from fastcma_baseline import benchmark_sphere


def sphere(x):
    return sum(v * v for v in x)


def main():
    x0 = [0.5, -0.2, 0.8]
    xmin, es = fastcma.fmin(sphere, x0, sigma=0.3, maxfevals=4000, ftarget=1e-12)
    print("Best point (Rust CMA-ES):", xmin)
    print("f(xmin) =", sphere(xmin))

    # Pure-Python baseline for comparison
    baseline = benchmark_sphere(dim=10, iters=80)
    print("\nPure-Python baseline vs Rust:")
    pprint(baseline)

    # Vectorized evaluation demo (evaluate batch at once)
    def sphere_vec(X):
        return [sphere(x) for x in X]

    xmin_vec, _es2 = fastcma.fmin_vec(sphere_vec, [0.4, -0.1, 0.3], sigma=0.25, maxfevals=3000)
    print("\nVectorized run best point:", xmin_vec)


if __name__ == "__main__":
    main()
