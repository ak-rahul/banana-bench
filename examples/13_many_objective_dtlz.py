"""
Many-objective benchmarking with DTLZ.

DTLZ problems scale to any number of objectives M. This example benchmarks a
toy multi-objective random search on DTLZ2 and DTLZ7 with 3 objectives, scoring
it with IGD and exact hypervolume against the true Pareto front. It shows how
to pick M with `n_objectives`, give every run the same evaluation budget, and
build uniform reference points with Das–Dennis.
"""

import numpy as np

from bananabench import MOBenchmarkRunner
from bananabench import multiobjective as mo


def random_search_mo(func, bounds, n_objectives, n_points=2000, seed=0):
    """Samples random points; the runner keeps the non-dominated ones."""
    rng = np.random.default_rng(seed)
    low, high = np.array(bounds).T
    X = rng.uniform(low, high, size=(n_points, len(bounds)))
    return X, np.array([func(x) for x in X])


def near_front_search(func, bounds, n_objectives, n_points=500, seed=0):
    """Cheats with problem knowledge (distance variables at their optimum) to show
    what a near-perfect approximation scores."""
    rng = np.random.default_rng(seed)
    X = rng.random((n_points, len(bounds)))
    X[:, n_objectives - 1 :] = 0.5
    return X, np.array([func(x) for x in X])


def main():
    print("Das-Dennis reference points for M = 3, 12 partitions:", mo.das_dennis(3, 12).shape)
    print("True DTLZ2 front sample:", mo.get_pareto_front("dtlz2", n_objectives=3).shape)

    for algorithm, name in [(random_search_mo, "RandomSearch"), (near_front_search, "NearFront")]:
        runner = MOBenchmarkRunner(
            algorithm, algorithm_name=name, n_objectives=3, max_evals=2000, verbose=True
        )
        runner.run_suite(functions=["dtlz2"])

    # Five objectives: hypervolume stays exact for small fronts, Monte Carlo for large ones.
    front = mo.get_pareto_front("dtlz2", n_points=50, n_objectives=5)
    ref = np.full(5, 1.1)
    exact = mo.hypervolume(front, ref, method="exact")
    estimate = mo.hypervolume(front, ref, method="monte_carlo", n_samples=200_000, seed=0)
    print(
        f"\n5-objective DTLZ2 front ({len(front)} points): exact HV = {exact:.4f}, "
        f"Monte Carlo = {estimate:.4f}"
    )


if __name__ == "__main__":
    main()
