"""
Multi-objective optimization with the ZDT suite.

Unlike the rest of the package, `multiobjective` functions return a 2-element
objective array `[f1, f2]` (both minimized) rather than a single float, so
there is no single "best_cost" -- quality is judged by how well an approximate
Pareto front covers the true one. This example evaluates zdt1 directly, then
runs a minimal NSGA-II-style optimizer (non-dominated sorting + crowding-free
truncation, no crossover -- just enough to demonstrate convergence pressure)
through `MOBenchmarkRunner` and reports IGD/hypervolume against the true front.
"""

import numpy as np

from bananabench import MOBenchmarkRunner
from bananabench import multiobjective as mo
from bananabench.multiobjective import get_pareto_front, non_dominated_front


def evaluate_at_a_point():
    x = np.full(30, 0.5)
    f1, f2 = mo.zdt1(x)
    print("zdt1(0.5, ..., 0.5):")
    print(f"  f1 = {f1:.4f}, f2 = {f2:.4f}")

    front = get_pareto_front("zdt1", n_points=5)
    print("  sample of the true Pareto front (x1 swept, x2..xn = 0):")
    for f1, f2 in front:
        print(f"    f1={f1:.4f}  f2={f2:.4f}")


def simple_mo_optimizer(func, bounds, n_objectives, generations=30, pop_size=100, seed=0):
    """
    A bare-bones (mu+lambda) evolutionary loop: mutate the current population,
    keep the non-dominated union of parents and offspring, and fill back up to
    pop_size with dominated leftovers so population size stays constant.
    Missing crowding-distance diversity preservation, so it's a demonstration
    optimizer, not a competitive one.
    """
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    dim = len(bounds)

    X = lo + rng.random((pop_size, dim)) * (hi - lo)

    for _ in range(generations):
        step = (hi - lo) * 0.1
        offspring = np.clip(X + rng.normal(scale=step, size=X.shape), lo, hi)
        combined = np.vstack([X, offspring])
        F_combined = np.array([func(x) for x in combined])

        nondominated_mask = np.array(
            [
                not np.any(
                    np.all(F_combined <= F_combined[i], axis=1)
                    & np.any(F_combined < F_combined[i], axis=1)
                )
                for i in range(len(F_combined))
            ]
        )
        survivors = combined[nondominated_mask]
        if len(survivors) >= pop_size:
            X = survivors[:pop_size]
        else:
            fill = combined[~nondominated_mask][: pop_size - len(survivors)]
            X = np.vstack([survivors, fill])

    F = np.array([func(x) for x in X])
    return X, F


def run_benchmark():
    runner = MOBenchmarkRunner(simple_mo_optimizer, algorithm_name="SimpleEA", seed=1)
    results = runner.run_suite(functions=["zdt1", "zdt2", "zdt4"], generations=30, pop_size=100)

    # non_dominated_front + get_pareto_front are what MOBenchmarkRunner uses
    # internally to score each run -- shown here directly for zdt1.
    _, F = simple_mo_optimizer(mo.zdt1, mo.get_mo_bounds("zdt1"), 2, seed=1)
    front = non_dominated_front(F)
    print(f"\nzdt1: {len(front)} non-dominated solutions out of {len(F)} found by SimpleEA")
    return results


if __name__ == "__main__":
    evaluate_at_a_point()
    print()
    run_benchmark()
