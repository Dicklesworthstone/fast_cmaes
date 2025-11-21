"""A minimal, pure-Python CMA-ES baseline (no SIMD, no LAPACK).

This is intentionally simple so it stays readable and provides a speed/quality
baseline against the optimized Rust implementation.
"""

from __future__ import annotations

import math
import random
import time
from typing import Callable, Iterable, List, Sequence, Tuple


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _square_sum(x: Sequence[float]) -> float:
    return _dot(x, x)


def cma_es(
    objective: Callable[[Sequence[float]], float],
    x0: Sequence[float],
    sigma: float = 0.5,
    lam: int | None = None,
    max_iter: int = 200,
    seed: int | None = 0,
) -> Tuple[List[float], float, int, float]:
    """Naive CMA-ES (diagonal-free) mainly for benchmarking.

    Returns (xbest, fbest, evals, elapsed_seconds).
    """

    if seed is not None:
        random.seed(seed)

    n = len(x0)
    lam = lam or max(4, int(4 + 3 * math.log(n)))
    mu = lam // 2
    weights = [math.log(lam / 2.0 + 0.5) - math.log(i + 1) if i < mu else 0.0 for i in range(lam)]
    w_sum = sum(weights[:mu])
    if w_sum != 0:
        weights = [w / w_sum for w in weights]

    xmean = list(x0)
    fbest = float("inf")
    xbest = list(x0)
    evals = 0
    start = time.perf_counter()

    for _ in range(max_iter):
        # Sample population (isotropic Gaussian)
        population: List[List[float]] = []
        fitness: List[float] = []
        for _ in range(lam):
            x = [xm + sigma * random.gauss(0.0, 1.0) for xm in xmean]
            fx = objective(x)
            population.append(x)
            fitness.append(fx)
        evals += lam

        # Rank and update mean
        ranked = sorted(zip(fitness, population), key=lambda t: t[0])
        fitness_sorted, pop_sorted = zip(*ranked)
        new_mean = [0.0] * n
        for k in range(mu):
            wk = weights[k]
            xk = pop_sorted[k]
            for i in range(n):
                new_mean[i] += wk * xk[i]
        xmean = new_mean

        # Track best
        if fitness_sorted[0] < fbest:
            fbest = fitness_sorted[0]
            xbest = list(pop_sorted[0])

        # Very light step-size damping toward smaller steps if improving slowly
        if fitness_sorted[0] < fbest * 1.01:
            sigma *= 0.99
        else:
            sigma *= 1.01

        # Simple termination: if steps are tiny or objective very low
        if sigma < 1e-12 or fbest < 1e-12:
            break

    elapsed = time.perf_counter() - start
    return xbest, fbest, evals, elapsed


def benchmark_sphere(dim: int = 20, iters: int = 200) -> dict:
    """Benchmark naive Python vs Rust extension on the sphere function.

    The Rust extension portion imports fastcma lazily; if unavailable, it is skipped.
    """

    def sphere(x: Sequence[float]) -> float:
        return _square_sum(x)

    x0 = [0.5] * dim

    py_x, py_f, py_evals, py_t = cma_es(sphere, x0, sigma=0.3, max_iter=iters, seed=0)

    rust = None
    try:
        import fastcma

        t0 = time.perf_counter()
        xmin, _es = fastcma.fmin(sphere, x0, 0.3, maxfevals=py_evals, ftarget=1e-12)
        rust_t = time.perf_counter() - t0
        rust_f = sphere(xmin)
        rust = {
            "elapsed": rust_t,
            "fbest": rust_f,
            "evals": py_evals,  # fastcma stops when done; keep comparable budget cap
        }
    except Exception:
        rust = None

    return {
        "python": {"elapsed": py_t, "fbest": py_f, "evals": py_evals, "xbest": py_x},
        "rust": rust,
    }


if __name__ == "__main__":
    import pprint

    pprint.pp(benchmark_sphere())
