"""
Gradient-based optimization via finite-difference gradient estimation.

Every function in banana-bench is a black box (no analytic gradient), but
`gradients.GradientWrapper` estimates one via finite differences and exposes
it as a SciPy-compatible `jac`, so gradient-based methods like L-BFGS-B work
out of the box. Requires SciPy (`pip install banana-bench[dev]`, or just
`pip install scipy`); the script explains and exits cleanly if it's missing.
"""

import numpy as np

from bananabench import get_bounds, get_function_info
from bananabench.gradients import GradientWrapper, approximate_gradient


def main():
    try:
        from scipy.optimize import minimize
    except ImportError:
        print("This example needs SciPy: pip install scipy")
        return

    # approximate_gradient() alone, compared against sphere's known analytic
    # gradient (2x), to show what it's actually computing.
    x = np.array([1.0, -2.0, 3.0])
    grad = approximate_gradient(get_function_info("sphere")["function"], x, method="central")
    print(f"approximate_gradient(sphere, {x}) = {grad}")
    print(f"analytic 2*x                      = {2 * x}")

    # GradientWrapper bundles a function with .gradient()/.fun_and_grad(),
    # and counts evaluations — useful for comparing optimizer efficiency.
    function_name = "rosenbrock"
    info = get_function_info(function_name)
    wrapped = GradientWrapper(info["function"])

    bounds = get_bounds(function_name, dim=5)
    x0 = np.zeros(5)

    result = minimize(
        wrapped,
        x0,
        jac=wrapped.gradient,
        method="L-BFGS-B",
        bounds=bounds,
    )

    print(f"\nL-BFGS-B on {function_name} (5D), starting from zeros:")
    print(f"  best x     = {result.x}")
    print(f"  best value = {result.fun:.6e}")
    print(f"  known min  = {info['known_minimum']}")
    print(f"  evaluations: {wrapped.evaluations} calls, {wrapped.grad_evaluations} gradients")


if __name__ == "__main__":
    main()
