"""Minimal demo using fastcma and python-decouple for settings."""
from decouple import AutoConfig
from fastcma import CMAES, ff


def main() -> None:
    config = AutoConfig()
    dim = config("DIM", cast=int, default=4)
    sigma = config("SIGMA", cast=float, default=0.5)

    x0 = [0.5] * dim
    es = CMAES(x0, sigma, None, None, None, covariance_mode="full")

    while not es.stop():
        X = es.ask()
        fvals = [ff.rosenbrock(x) for x in X]
        es.tell(X, fvals)
        es.disp(verb_modulo=20)

    xbest, fbest, *_ = es.result
    print("best x:", xbest)
    print("best f:", fbest)


if __name__ == "__main__":
    main()
