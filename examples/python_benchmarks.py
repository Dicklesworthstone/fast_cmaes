"""Showcase hard objective functions with Rust fastcma vs naive Python.

Run with:

    python examples/python_benchmarks.py

The Rust-backed runs use higher dimensions; the naive baseline stays tiny
to keep wall-clock time reasonable.
"""

import math
import time
from pprint import pprint

import fastcma
from fastcma_baseline import benchmark_sphere, cma_es


def rastrigin(x):
    return 10 * len(x) + sum(v * v - 10 * math.cos(2 * math.pi * v) for v in x)


def ackley(x):
    n = len(x)
    sum_sq = sum(v * v for v in x)
    sum_cos = sum(math.cos(2 * math.pi * v) for v in x)
    term1 = -20 * math.exp(-0.2 * math.sqrt(sum_sq / n))
    term2 = -math.exp(sum_cos / n)
    return term1 + term2 + 20 + math.e


def schwefel_2_26(x):
    n = len(x)
    return 418.9828872724339 * n - sum(v * math.sin(abs(v) ** 0.5) for v in x)


def katsuura(x):
    n = float(len(x))
    prod = 1.0
    for i, xi in enumerate(x):
        s = 0.0
        for j in range(1, 33):
            two = 2.0**j
            val = two * xi
            s += abs(val - round(val)) / two
        prod *= (1.0 + (i + 1) * s) ** (10.0 / (n ** 1.2))
    return prod - 1.0


def weierstrass(x):
    a = 0.5
    b = 3.0
    kmax = 20
    sum1 = 0.0
    for xi in x:
        for k in range(kmax + 1):
            ak = a**k
            bk = b**k
            sum1 += ak * math.cos(2 * math.pi * bk * (xi + 0.5))
    sum2 = sum(a**k * math.cos(2 * math.pi * (b**k) * 0.5) for k in range(kmax + 1))
    return sum1 - len(x) * sum2


def happy_cat(x):
    norm2 = sum(v * v for v in x)
    sumx = sum(x)
    n = float(len(x))
    return abs(norm2 - n) ** 0.25 + (0.5 * norm2 + sumx) / n + 0.5


def run_fastcma(name, f, x0, sigma, maxfevals, covariance_mode=None):
    t0 = time.time()
    xmin, es = fastcma.fmin(f, x0, sigma=sigma, maxfevals=maxfevals, ftarget=1e-12, covariance_mode=covariance_mode)
    xbest, fbest, _evals_best, counteval, _iters, _xmean, _stds = es.result
    return {
        "fbest": fbest,
        "xmin_head": xmin[:4],
        "dims": len(x0),
        "evals": counteval,
        "elapsed_s": round(time.time() - t0, 3),
        "mode": covariance_mode or "full",
        "name": name,
    }


def main():
    rust_cases = [
        ("ackley-20d", ackley, [1.2] * 20, 0.45, 40_000, None),
        ("rastrigin-20d", rastrigin, [0.3] * 20, 0.5, 60_000, None),
        ("schwefel-2.26-18d", schwefel_2_26, [420.0] * 18, 40.0, 80_000, "diag"),
        ("katsuura-12d", katsuura, [0.25] * 12, 0.45, 70_000, "diag"),
        ("weierstrass-14d", weierstrass, [0.2] * 14, 0.35, 60_000, "diag"),
        ("happy-cat-16d", happy_cat, [-1.5] * 16, 0.55, 60_000, "diag"),
    ]

    print("Rust fastcma on hard/high-dim functions (quick budgets):")
    rust_results = [run_fastcma(*case) for case in rust_cases]
    pprint(rust_results)

    print("\nNaive Python CMA-ES (tiny dims to keep it fast):")
    xbest, fbest, evals, elapsed = cma_es(rastrigin, [0.4] * 4, sigma=0.25, max_iter=400, seed=0)
    print({"fbest": fbest, "xbest": xbest, "evals": evals, "elapsed": round(elapsed, 3)})

    print("\nPure-Python sphere baseline (for reference):")
    pprint(benchmark_sphere(dim=20, iters=120))


if __name__ == "__main__":
    main()
