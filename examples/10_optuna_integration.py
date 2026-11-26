"""
Wiring a banana-bench function into Optuna.

Optuna (Akiba, Sano, Yanase, Ohta & Koyama, "Optuna: A Next-generation
Hyperparameter Optimization Framework," KDD 2019) is the dominant
hyperparameter-optimization framework in the ML ecosystem -- a materially
different, much larger audience than the metaheuristics researchers most
other benchmark-function packages target. Any `BENCHMARK_SUITE` function
plugs straight into Optuna's standard `objective(trial) -> float` interface:
one `trial.suggest_float` call per dimension, built from the function's own
`bounds`/`default_dim` metadata rather than hardcoding a search space by
hand. Requires Optuna (`pip install optuna`); this dependency lives only in
this example, not in `bananabench`'s own install requirements.
"""

import numpy as np

from bananabench import metadata
from bananabench.utils import normalize_bounds


def main():
    try:
        import optuna
    except ImportError:
        print("This example needs Optuna: pip install optuna")
        return

    # Quiet down Optuna's default per-trial logging so the summary below reads
    # cleanly; comment this out to watch every trial as it runs.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    function_name = "rastrigin"
    info = metadata.get_function_info(function_name)
    func = info["function"]
    # Override the function's own default_dim (10) down to 2: rastrigin is
    # highly multimodal, and a fixed trial budget on a black-box sampler like
    # TPE gets meaningfully closer to the known optimum in a low-dimensional
    # search space -- this keeps the demo's convergence readable rather than
    # implying anything about Optuna's general scalability to high dimensions.
    dim = 2
    bounds = normalize_bounds(info["bounds"], dim)

    def objective(trial: "optuna.Trial") -> float:
        x = np.array([trial.suggest_float(f"x{i}", lo, hi) for i, (lo, hi) in enumerate(bounds)])
        return func(x)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=200, show_progress_bar=False)

    best_x = np.array([study.best_params[f"x{i}"] for i in range(dim)])
    print(f"{function_name} via Optuna (TPE sampler, 200 trials):")
    print(f"  x*        = {best_x}")
    print(f"  f(x*)     = {study.best_value:.6f}")
    print(f"  known min = {info['known_minimum']}")


if __name__ == "__main__":
    main()
