"""
Testing an optimizer for center bias (Kudela 2022).

Most classic benchmark functions have their optimum at the origin, the exact
center of a symmetric search box. An optimizer that drifts toward the center
looks excellent on them, and the advantage disappears on real problems, where
the optimum could be anywhere. Kudela's follow-up study (arXiv:2301.01984) found
47 of 90 published methods center-biased.

`analysis.center_bias_test` runs an optimizer on each function and again with
the optimum moved by 10% of the range. It compares the mean errors, and flags
the optimizer when the geometric mean of the shifted-to-unshifted ratios is
above 10. This example tests two optimizers that differ in a single line.
"""

import numpy as np

from bananabench.analysis import center_bias_test


def hill_climber(func, bounds):
    """A plain (1+1) hill climber with an adaptive step size."""
    low, high = np.array(bounds).T
    x = np.random.uniform(low, high)
    fx = func(x)
    sigma = 0.1
    while True:
        y = np.clip(x + np.random.normal(0, sigma, x.size) * (high - low), low, high)
        fy = func(y)
        if fy <= fx:
            x, fx, sigma = y, fy, sigma * 1.5
        else:
            sigma *= 0.9


def center_pulled_hill_climber(func, bounds):
    """The same hill climber, except each step is also pulled 5% toward the center.

    A pull like this is easy to introduce by accident (for example, averaging
    with a mirrored point) and it helps a lot on functions centered at 0.
    """
    low, high = np.array(bounds).T
    center = (low + high) / 2
    x = np.random.uniform(low, high)
    fx = func(x)
    sigma = 0.1
    while True:
        step = np.random.normal(0, sigma, x.size) * (high - low)
        y = np.clip(x + step - 0.05 * (x - center), low, high)
        fy = func(y)
        if fy <= fx:
            x, fx, sigma = y, fy, sigma * 1.5
        else:
            sigma *= 0.9


def main():
    for optimizer in (hill_climber, center_pulled_hill_climber):
        print(f"\n{optimizer.__name__}")
        result = center_bias_test(
            optimizer,
            functions=["sphere", "rastrigin", "ackley", "griewank", "schwefel2_26"],
            n_runs=3,
            max_evals=lambda d: 500 * d,
            seed=0,
        )
        print(result["report"])


if __name__ == "__main__":
    main()
