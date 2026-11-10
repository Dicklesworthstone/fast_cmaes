import math

import pytest


def sphere_vec(xs):
    return [sum(v * v for v in x) for x in xs]


def test_package_reexports_every_native_name():
    import fastcma
    from fastcma import fastcma as native

    public = {name for name in dir(native) if not name.startswith("_")}
    assert public <= set(fastcma.__all__)
    assert all(getattr(fastcma, name) is getattr(native, name) for name in public)


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


def test_stop_names_the_termination_reasons():
    from fastcma import fmin

    _xmin, es = fmin(lambda x: sum(v * v for v in x), [0.5, -0.2], sigma=0.3, maxfevals=60)
    reasons = es.stop()
    assert isinstance(reasons, dict)
    assert "maxfevals" in reasons
    assert set(reasons) <= {"maxfevals", "ftarget", "condition", "tolfun", "tolx"}


def test_fmin_vec_accepts_a_strided_numpy_result():
    np = pytest.importorskip("numpy")
    from fastcma import fmin_vec

    def strided(xs):
        vals = np.array(sphere_vec(xs))
        # A view with stride 2: same values, not contiguous.
        return np.repeat(vals, 2)[::2]

    _xmin, es = fmin_vec(strided, [0.4, -0.1, 0.3], sigma=0.25, maxfevals=4000)
    assert es.result[1] < 1e-6


def test_fmin_vec_final_evaluation_checks_its_length():
    from fastcma import fmin_vec

    def drops_single_rows(xs):
        # Populations always have several rows; only the final mean evaluation has one.
        return [] if len(xs) == 1 else sphere_vec(xs)

    with pytest.raises(ValueError, match="len\\(X\\) fitness values"):
        fmin_vec(drops_single_rows, [0.4, -0.1, 0.3], sigma=0.25, maxfevals=400)
