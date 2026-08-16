"""
Constrained optimization with the g_suite functions.

Unlike the rest of the package, `g_suite` functions return
`(objective, inequality_violations, equality_violations)` rather than a single
float — inequality entries must be <= 0 and equality entries == 0 at a
feasible point. This example wires g06 into `scipy.optimize.minimize`'s SLSQP
solver, which is the standard way to actually *use* a constrained problem
rather than just evaluate it at one point. Requires SciPy.
"""

import numpy as np

from bananabench import g_suite


def main():
    try:
        from scipy.optimize import minimize
    except ImportError:
        print("This example needs SciPy: pip install scipy")
        return

    # g06: minimize (x0-10)^3 + (x1-20)^3 subject to 2 inequality constraints.
    # Known optimum: f(x*) ~= -6961.814 at x* ~= (14.095, 0.84296).
    def objective(x):
        f, _, _ = g_suite.g06(x)
        return f

    def inequalities(x):
        # g_suite returns g(x) <= 0 for feasibility; SLSQP wants fun(x) >= 0.
        _, g, _ = g_suite.g06(x)
        return -g

    bounds = [(13.0, 100.0), (0.0, 100.0)]
    x0 = np.array([20.1, 5.84])  # a commonly used feasible-ish starting point

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "ineq", "fun": inequalities}],
    )

    f, g, _ = g_suite.g06(result.x)
    print("g06 constrained optimization (SLSQP):")
    print(f"  x*        = {result.x}")
    print(f"  f(x*)     = {f:.4f}   (known optimum: -6961.8139)")
    print(f"  g(x*)     = {g}   (feasible if both <= ~0)")

    # g11 shows the equality-constraint path: minimize x0^2 + (x1-1)^2 subject
    # to x1 = x0^2. Known optimum: f(x*) ~= 0.7499 at x* ~= (+-0.7071, 0.5).
    def g11_objective(x):
        f, _, _ = g_suite.g11(x)
        return f

    def g11_equality(x):
        _, _, h = g_suite.g11(x)
        return h

    result11 = minimize(
        g11_objective,
        x0=np.array([0.5, 0.5]),
        method="SLSQP",
        constraints=[{"type": "eq", "fun": g11_equality}],
    )
    f11, _, h11 = g_suite.g11(result11.x)
    print("\ng11 constrained optimization (SLSQP):")
    print(f"  x*        = {result11.x}")
    print(f"  f(x*)     = {f11:.4f}   (known optimum: 0.7499)")
    print(f"  h(x*)     = {h11}   (feasible if ~0)")


if __name__ == "__main__":
    main()
