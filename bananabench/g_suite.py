"""
Constrained G-function suite (G01-G13).

This module implements the classic G-suite for constrained optimization,
originally proposed in the 2006 CEC Special Session on Constrained Real-Parameter Optimization.

Each function returns the objective value and a tuple of constraint violations
(g(x) <= 0 for inequality, h(x) == 0 for equality).
"""

import numpy as np


def g04(x: np.ndarray):
    """
    G04 Constrained Benchmark Function.
    Dimension: 5
    Objective: minimize f(x)
    Constraints: 6 inequalities (g_i(x) <= 0)
    Optimal: f(x*) = -30665.539
    """
    x = np.asarray(x, dtype=float)
    if x.size != 5:
        raise ValueError("g04 requires exactly 5 dimensions.")

    f = 5.3578547 * x[2] ** 2 + 0.8356891 * x[0] * x[4] + 37.293239 * x[0] - 40792.141

    g = np.zeros(6)
    g[0] = (
        85.334407 + 0.0056858 * x[1] * x[4] + 0.0006262 * x[0] * x[3] - 0.0022053 * x[2] * x[4] - 92
    )
    g[1] = -85.334407 - 0.0056858 * x[1] * x[4] - 0.0006262 * x[0] * x[3] + 0.0022053 * x[2] * x[4]
    g[2] = (
        80.51249 + 0.0071317 * x[1] * x[4] + 0.0029955 * x[0] * x[1] + 0.0021813 * x[2] ** 2 - 110
    )
    g[3] = (
        -80.51249 - 0.0071317 * x[1] * x[4] - 0.0029955 * x[0] * x[1] - 0.0021813 * x[2] ** 2 + 90
    )
    g[4] = (
        9.300961 + 0.0047026 * x[2] * x[4] + 0.0012547 * x[0] * x[2] + 0.0019085 * x[2] * x[3] - 25
    )
    g[5] = (
        -9.300961 - 0.0047026 * x[2] * x[4] - 0.0012547 * x[0] * x[2] - 0.0019085 * x[2] * x[3] + 20
    )

    # Return (objective, inequality_constraints, equality_constraints)
    return f, g, np.array([])


def g06(x: np.ndarray):
    """
    G06 Constrained Benchmark Function.
    Dimension: 2
    Objective: minimize f(x)
    Constraints: 2 inequalities (g_i(x) <= 0)
    Optimal: f(x*) = -6961.81388
    """
    x = np.asarray(x, dtype=float)
    if x.size != 2:
        raise ValueError("g06 requires exactly 2 dimensions.")

    f = (x[0] - 10) ** 3 + (x[1] - 20) ** 3

    g = np.zeros(2)
    g[0] = -((x[0] - 5) ** 2) - (x[1] - 5) ** 2 + 100
    g[1] = (x[0] - 6) ** 2 + (x[1] - 5) ** 2 - 82.81

    return f, g, np.array([])


def g08(x: np.ndarray):
    """
    G08 Constrained Benchmark Function.
    Dimension: 2
    Objective: minimize f(x)
    Constraints: 2 inequalities (g_i(x) <= 0)
    Optimal: f(x*) = -0.095825
    """
    x = np.asarray(x, dtype=float)
    if x.size != 2:
        raise ValueError("g08 requires exactly 2 dimensions.")

    f = -(np.sin(2 * np.pi * x[0]) ** 3 * np.sin(2 * np.pi * x[1])) / (x[0] ** 3 * (x[0] + x[1]))

    g = np.zeros(2)
    g[0] = x[0] ** 2 - x[1] + 1
    g[1] = 1 - x[0] + (x[1] - 4) ** 2

    return f, g, np.array([])


def g11(x: np.ndarray):
    """
    G11 Constrained Benchmark Function.
    Dimension: 2
    Objective: minimize f(x)
    Constraints: 1 equality (h_i(x) == 0)
    Optimal: f(x*) = 0.7499
    """
    x = np.asarray(x, dtype=float)
    if x.size != 2:
        raise ValueError("g11 requires exactly 2 dimensions.")

    f = x[0] ** 2 + (x[1] - 1) ** 2

    h = np.zeros(1)
    h[0] = x[1] - x[0] ** 2

    return f, np.array([]), h
