"""
Comparing optimizers across a whole problem set, the way the literature does.

One mean error per algorithm hides almost everything. This example records each
run's best-so-far trace (`record_trace=True`), then uses `bananabench.analysis`
for four standard views:

- expected running time (ERT) to a target precision (COCO's aRT);
- Dolan–Moré performance profiles, with ERT as the cost;
- Moré–Wild data profiles: fraction of problems solved per budget unit;
- a COCO-style runtime ECDF over 51 targets from 1e2 to 1e-8;

plus a Friedman / Wilcoxon–Holm statistical comparison (Demšar 2006). The
statistics need SciPy and the plots need Matplotlib:
pip install "banana-bench[stats,viz]".
"""

import numpy as np

from bananabench import BenchmarkRunner
from bananabench.analysis import (
    SCIPY_AVAILABLE,
    data_profile,
    ert_from_runners,
    histories_from_runners,
    performance_profile_from_runners,
    runtime_ecdf,
    statistical_comparison,
)


def random_search(func, bounds, seed=0):
    rng = np.random.default_rng(seed)
    low, high = np.array(bounds).T
    while True:
        func(rng.uniform(low, high))


def hill_climber(func, bounds, seed=0):
    rng = np.random.default_rng(seed)
    low, high = np.array(bounds).T
    x = rng.uniform(low, high)
    fx = func(x)
    sigma = 0.2
    while True:
        y = np.clip(x + rng.normal(0, sigma, x.size) * (high - low), low, high)
        fy = func(y)
        if fy <= fx:
            x, fx = y, fy
            sigma *= 1.5
        else:
            sigma *= 0.9


def coordinate_search(func, bounds, seed=0):
    """Compass search: tries +/- steps along each axis, halving the step on failure."""
    low, high = np.array(bounds).T
    x = np.random.default_rng(seed).uniform(low, high)
    fx = func(x)
    step = (high - low) / 4
    while True:
        improved = False
        for i in range(x.size):
            for sign in (1, -1):
                y = x.copy()
                y[i] = np.clip(y[i] + sign * step[i], low[i], high[i])
                fy = func(y)
                if fy < fx:
                    x, fx, improved = y, fy, True
                    break
        if not improved:
            step /= 2


def main():
    functions = ["sphere", "rosenbrock", "ackley", "rastrigin", "griewank", "levy"]
    dims = {name: 4 for name in functions}
    runners = []
    for algorithm, name in [
        (random_search, "RandomSearch"),
        (hill_climber, "HillClimber"),
        (coordinate_search, "CompassSearch"),
    ]:
        runner = BenchmarkRunner(
            algorithm,
            name,
            n_runs=3,
            seed=0,
            max_evals=lambda d: 1000 * d,
            record_trace=True,
            verbose=False,
            show_progress=False,
        )
        runner.run_suite(functions=functions, dimensions=dims)
        runners.append(runner)

    target = 1e-3
    print(f"Expected running time to f - f_opt <= {target:g} (inf = never reached):")
    for name, per_function in ert_from_runners(runners, target=target).items():
        cells = ", ".join(f"{fn}={v:.0f}" for fn, v in per_function.items())
        print(f"  {name:<14} {cells}")

    taus, perf = performance_profile_from_runners(runners, target=target)
    print("\nPerformance profile: fraction of functions where each solver is fastest (tau = 1)")
    for name, rho in perf.items():
        print(f"  {name:<14} {rho[0]:.2f}  (solved at all: {rho[-1]:.2f})")

    histories, f_opt, run_dims = histories_from_runners(runners)
    alphas, data = data_profile(histories, run_dims, tau=1e-3)
    print("\nData profile: fraction of problem instances solved within 100 simplex gradients")
    for name, d in data.items():
        print(f"  {name:<14} {d[np.searchsorted(alphas, 100, side='right') - 1]:.2f}")

    budgets, ecdf = runtime_ecdf(histories, f_opt, dims=run_dims)
    print("\nRuntime ECDF: fraction of (problem, target) pairs solved by the end")
    for name, curve in ecdf.items():
        print(f"  {name:<14} {curve[-1]:.2f}")

    if SCIPY_AVAILABLE:
        print()
        print(statistical_comparison(runners, metric="error")["report"])

    try:
        import matplotlib

        matplotlib.use("Agg")
        from bananabench import plot_data_profile, plot_performance_profile, plot_runtime_ecdf

        plot_performance_profile(taus, perf)
        plot_data_profile(alphas, data, tau=1e-3)
        plot_runtime_ecdf(budgets, ecdf)
        print("\nPlots created (call fig.savefig(...) or pass save_path=... to keep them).")
    except ImportError:
        print("\nInstall matplotlib to draw these as plots.")


if __name__ == "__main__":
    main()
