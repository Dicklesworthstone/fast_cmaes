import math


def sphere_vec(xs):
    return [sum(v * v for v in x) for x in xs]


def test_fmin_roundtrip_basic():
    from fastcma import fmin

    xmin, es = fmin(lambda x: sum(v * v for v in x), [0.5, -0.2, 0.8], sigma=0.3, maxfevals=5000)
    fbest = es.result[1]
    assert len(xmin) == 3
    assert fbest < 1e-6


def test_fmin_vec_roundtrip():
    from fastcma import fmin_vec

    x0 = [0.4, -0.1, 0.3]
    xmin, es = fmin_vec(sphere_vec, x0, sigma=0.25, maxfevals=4000)
    fbest = es.result[1]
    assert math.isfinite(fbest)
    assert fbest < 1e-6
    assert len(xmin) == len(x0)
