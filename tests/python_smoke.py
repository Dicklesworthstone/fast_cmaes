import math


def test_python_sphere():
import fastcma
from fastcma_baseline import benchmark_sphere

    def sphere(x):
        return sum(v * v for v in x)

    xmin, _es = fastcma.fmin(sphere, [0.4, -0.6, 0.2], 0.3, maxfevals=4000, ftarget=1e-12)
    assert max(abs(v) for v in xmin) < 1e-3


def test_naive_python_baseline_runs():
    from fastcma import benchmark_sphere

    res = benchmark_sphere(dim=10, iters=60)
    assert res["python"]["elapsed"] > 0.0
    # Rust path is optional; only assert the pure python path exists
    assert res["python"]["fbest"] >= 0.0


if __name__ == "__main__":
    test_python_sphere()
