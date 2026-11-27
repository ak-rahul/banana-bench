"""
Fair benchmarking: evaluation budgets and result verification.

Two optimizers can only be compared fairly at equal cost. `BenchmarkRunner`'s
`max_evals` gives every run the same evaluation budget (here, 500 evaluations per
dimension), and the runner stops any algorithm that tries to exceed it. Algorithms
don't need to handle the budget themselves: the run is scored on the best point it
evaluated.

By default the runner also re-evaluates f(best_x) instead of trusting the reported
cost, so the "lying" optimizer below gets flagged with `cost_mismatch=True` and
scored on what it actually found.
"""

import numpy as np

from bananabench import BenchmarkRunner, compare_algorithms


def random_search(func, bounds, seed=0):
    """Samples uniformly forever; only the budget stops it."""
    rng = np.random.default_rng(seed)
    low, high = np.array(bounds).T
    while True:
        func(rng.uniform(low, high))


def local_search(func, bounds, step=0.5, seed=0):
    """(1+1) hill climber with a shrinking step: much better use of the same budget."""
    rng = np.random.default_rng(seed)
    low, high = np.array(bounds).T
    x = rng.uniform(low, high)
    fx = func(x)
    while True:
        candidate = np.clip(x + rng.normal(0, step * (high - low) / 10, x.size), low, high)
        fc = func(candidate)
        if fc < fx:
            x, fx = candidate, fc
        else:
            step *= 0.99


def lying_optimizer(func, bounds, **kwargs):
    """Reports a perfect score its point doesn't achieve."""
    x = np.array([b[1] for b in bounds], dtype=float)
    func(x)
    return x, 0.0


def main():
    functions = ["sphere", "rosenbrock", "ackley", "rastrigin"]
    dimensions = {name: 5 for name in functions}
    budget = lambda dim: 500 * dim  # noqa: E731 - a budget proportional to dimension

    runners = []
    for algorithm, name in [(random_search, "RandomSearch"), (local_search, "LocalSearch")]:
        runner = BenchmarkRunner(
            algorithm,
            algorithm_name=name,
            n_runs=3,
            seed=1,
            max_evals=budget,
            verbose=False,
            show_progress=False,
        )
        runner.run_suite(functions=functions, dimensions=dimensions)
        runners.append(runner)

    print("Both algorithms ran with exactly 500 * dim = 2500 evaluations per run:\n")
    print(compare_algorithms(runners))

    print("\nVerification catches a misreported result:")
    result = BenchmarkRunner(lying_optimizer, verbose=False).run_single("sphere", dim=5)
    print(f"  reported best_cost : {result['reported_cost']}")
    print(f"  verified best_cost : {result['best_cost']:.2f}")
    print(f"  cost_mismatch      : {result['cost_mismatch']}")


if __name__ == "__main__":
    main()
