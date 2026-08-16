"""
Benchmarking a custom optimizer with BenchmarkRunner.

Shows the full loop: write an optimizer with the signature
`algorithm(func, bounds, **kwargs) -> (best_x, best_cost)`, run it across
several functions with multiple repetitions, and inspect/save the results.
Needs only NumPy; enables progress bars and parallel execution automatically
if tqdm / joblib happen to be installed (BenchmarkRunner degrades gracefully
without them).
"""

import tempfile
from pathlib import Path

import numpy as np

from bananabench import BenchmarkRunner


def random_search(func, bounds, max_iter=2000, seed=None):
    """Uniform random search — a deliberately weak baseline optimizer."""
    rng = np.random.default_rng(seed)
    bounds_array = np.array(bounds)
    lower, upper = bounds_array[:, 0], bounds_array[:, 1]

    best_x, best_cost = None, np.inf
    for _ in range(max_iter):
        x = rng.uniform(lower, upper)
        cost = func(x)
        if cost < best_cost:
            best_x, best_cost = x, cost

    return best_x, best_cost


def main():
    runner = BenchmarkRunner(
        algorithm=random_search,
        algorithm_name="RandomSearch",
        n_runs=5,
        seed=42,  # each of the 5 runs still gets its own derived seed
        verbose=True,
    )

    functions = ["sphere", "ackley", "rastrigin", "rosenbrock", "griewank"]
    results = runner.run_suite(functions=functions, max_iter=2000)

    print("\nPer-function summary:")
    for r in results:
        print(f"  {r['function']:<12} error_mean={r['error_mean']:.4f}")

    stats = runner.get_summary_stats()
    print(f"\nOverall: {stats['n_successful']}/{stats['n_results']} successful runs, ")
    print(f"mean error {stats['error_mean']:.4f}")

    # save_results() needs a real path; use a temp directory rather than
    # littering the current working directory.
    output_dir = Path(tempfile.mkdtemp(prefix="bananabench_"))
    csv_path = output_dir / "results.csv"
    runner.save_results(csv_path, format="csv")
    print(f"\nFull per-run results saved to: {csv_path}")


if __name__ == "__main__":
    main()
