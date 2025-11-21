"""Compare Rust vs naive Python CMA-ES on a couple of functions."""

from pprint import pprint

from fastcma_baseline import benchmark_sphere, cma_es


def rastrigin(x):
    import math

    return 10 * len(x) + sum(v * v - 10 * math.cos(2 * math.pi * v) for v in x)


def main():
    print("Sphere baseline (Rust vs Python):")
    pprint(benchmark_sphere(dim=20, iters=120))

    print("\nNaive Python CMA-ES on Rastrigin (4D):")
    xbest, fbest, evals, elapsed = cma_es(rastrigin, [0.4] * 4, sigma=0.25, max_iter=400, seed=0)
    print({"fbest": fbest, "xbest": xbest, "evals": evals, "elapsed": elapsed})


if __name__ == "__main__":
    main()
