"""
Classic engineering design optimization with the engineering suite.

Like `g_suite`, `engineering` functions return
`(objective, inequality_violations, equality_violations)` rather than a single
float -- inequality entries must be <= 0 at a feasible point (none of these
four problems has an equality constraint). This example wires spring_design
and welded_beam into `scipy.optimize.minimize`'s SLSQP solver. Requires SciPy.
"""

import numpy as np

from bananabench import engineering


def main():
    try:
        from scipy.optimize import minimize
    except ImportError:
        print("This example needs SciPy: pip install scipy")
        return

    # spring_design: minimize (N+2)*D*d^2 subject to 4 inequality constraints.
    # Known optimum: f(x*) ~= 0.012665 at x* ~= (0.05169, 0.35675, 11.287126).
    def spring_objective(x):
        f, _, _ = engineering.spring_design(x)
        return f

    def spring_inequalities(x):
        # engineering returns g(x) <= 0 for feasibility; SLSQP wants fun(x) >= 0.
        _, g, _ = engineering.spring_design(x)
        return -g

    bounds = [(0.05, 2.0), (0.25, 1.3), (2.0, 15.0)]
    x0 = np.array([0.05, 0.25, 10.0])  # a feasible-ish starting point

    result = minimize(
        spring_objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "ineq", "fun": spring_inequalities}],
    )

    f, g, _ = engineering.spring_design(result.x)
    print("spring_design constrained optimization (SLSQP):")
    print(f"  x*        = {result.x}")
    print(f"  f(x*)     = {f:.6f}   (known optimum: 0.012665)")
    print(f"  g(x*)     = {g}   (feasible if all <= ~0)")

    # welded_beam: minimize fabrication cost subject to 7 inequality
    # constraints. Known optimum: f(x*) ~= 1.724852 at
    # x* ~= (0.205730, 3.470489, 9.036624, 0.205730).
    def beam_objective(x):
        f, _, _ = engineering.welded_beam(x)
        return f

    def beam_inequalities(x):
        _, g, _ = engineering.welded_beam(x)
        return -g

    beam_bounds = [(0.1, 2.0), (0.1, 10.0), (0.1, 10.0), (0.1, 2.0)]
    beam_x0 = np.array([0.5, 5.0, 5.0, 0.5])

    beam_result = minimize(
        beam_objective,
        beam_x0,
        method="SLSQP",
        bounds=beam_bounds,
        constraints=[{"type": "ineq", "fun": beam_inequalities}],
    )

    f_beam, g_beam, _ = engineering.welded_beam(beam_result.x)
    print("\nwelded_beam constrained optimization (SLSQP):")
    print(f"  x*        = {beam_result.x}")
    print(f"  f(x*)     = {f_beam:.6f}   (known optimum: 1.724852)")
    print(f"  g(x*)     = {g_beam}   (feasible if all <= ~0)")


if __name__ == "__main__":
    main()
