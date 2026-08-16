"""
Visualization gallery: the plotting functions in one place.

Every function here returns a matplotlib Figure and takes an optional
save_path — this script just calls plt.show() and leaves saving to you
(via fig.savefig(...) or the save_path kwarg) rather than dumping PNGs into
your current directory as a side effect. Requires matplotlib
(`pip install banana-bench[viz]`).
"""

import numpy as np

try:
    import matplotlib.pyplot as plt

    from bananabench.visualization import (
        plot_algorithm_comparison,
        plot_convergence,
        plot_function_2d,
        plot_function_3d,
        plot_search_heatmap,
        plot_trajectory_2d,
    )
except ImportError:
    plt = None


def main():
    if plt is None:
        print("This example needs matplotlib: pip install 'banana-bench[viz]'")
        return

    # 1. Function landscape: 2D contour and 3D surface.
    plot_function_2d("ackley", show_optimum=True)
    plot_function_3d("rastrigin", elevation=30, azimuth=45)

    # 2. Optimization trajectory over the function's contour.
    trajectory = np.array([[5.0, 5.0], [3.0, 3.0], [1.0, 1.0], [0.3, 0.3], [0.0, 0.0]])
    plot_trajectory_2d("sphere", trajectory)

    # 3. Convergence history, with a reference line at the known minimum.
    history = [100, 50, 20, 10, 5, 2, 1, 0.1, 0.01]
    plot_convergence(history, function_name="sphere", known_minimum=0.0, log_scale=True)

    # 4. Search heatmap: where did the optimizer actually spend its time?
    rng = np.random.default_rng(0)
    visited_points = rng.uniform(-5, 5, size=(500, 2))
    plot_search_heatmap("rastrigin", visited_points, bins=30, cmap="hot")

    # 5. Comparing algorithms across functions on a chosen metric.
    results = {
        "RandomSearch": {
            "sphere": {"error": 0.05, "time": 0.8},
            "ackley": {"error": 0.3, "time": 0.9},
        },
        "L-BFGS-B": {
            "sphere": {"error": 1e-8, "time": 0.1},
            "ackley": {"error": 0.02, "time": 0.2},
        },
    }
    plot_algorithm_comparison(results, metric="error")

    print("Showing 6 figures - close each window to see the next.")
    plt.show()


if __name__ == "__main__":
    main()
