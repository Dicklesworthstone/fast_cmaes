import math

import fastcma


def test_python_sphere():
    def sphere(x):
        return sum(v * v for v in x)

    xmin, _es = fastcma.fmin(
        sphere, [0.4, -0.6, 0.2], 0.3, maxfevals=4000, ftarget=1e-12
    )
    assert max(abs(v) for v in xmin) < 1e-3


def test_naive_python_baseline_runs():
    def shifted_quad(x):
        return sum((v - 0.1) * (v - 0.1) for v in x)

    xmin, _es = fastcma.fmin(
        shifted_quad, [0.3, -0.2], 0.25, maxfevals=2000, ftarget=1e-10
    )
    assert shifted_quad(xmin) < 1e-6


if __name__ == "__main__":
    test_python_sphere()
