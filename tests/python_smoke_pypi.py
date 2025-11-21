def test_import_and_run_sphere():
    import fastcma

    def sphere(x):
        return sum(v * v for v in x)

    xmin, _es = fastcma.fmin(sphere, [0.4, -0.6, 0.2], 0.3, maxfevals=2000, ftarget=1e-8)
    assert max(abs(v) for v in xmin) < 1e-2


if __name__ == "__main__":
    test_import_and_run_sphere()
