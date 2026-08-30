"""
Constrained G-function suite.

This module implements the complete classic G-suite for constrained
optimization (g01-g24), originally proposed in the 2006 CEC Special Session
on Constrained Real-Parameter Optimization (Liang, Runarsson, Mezura-Montes,
Clerc, Suganthan, Coello Coello, Deb, 2006 -- see docs-bench/ for the
technical report). Every formula, coefficient, and known-optimum value below
is transcribed directly from that report and cross-checked against pagmo's
independent cec2006 implementation where the two overlap.

Each function returns the objective value and a tuple of constraint violations
(g(x) <= 0 for inequality, h(x) == 0 for equality).

g20 is a known exception: the technical report itself states that its listed
"best known solution" is slightly infeasible and that no fully feasible point
has ever been verified for this problem. Its known_minimum should be treated
as indicative, not a target a solver is expected to reach exactly.

``G_SUITE`` is a lightweight registry (function, dim, constraint counts, known
minimum) for tools that need to enumerate/introspect the suite (the CLI,
docs generator) without hand-parsing docstrings. It's deliberately smaller
than ``metadata.BENCHMARK_SUITE`` -- no ``bounds``/``optimal_point``/
``properties`` fields -- since those vary in shape per problem (some g-suite
problems have no box bounds at all) in ways that don't fit that schema
cleanly; see each function's own docstring for bounds and the optimal point.
``n_inequality``/``n_equality`` were cross-checked by calling every function
and measuring the actual returned array lengths, not just transcribed.
"""

from typing import Callable, Dict, List, Tuple, TypedDict

import numpy as np


def g01(x: np.ndarray):
    """
    G01 Constrained Benchmark Function.
    Dimension: 13
    Objective: minimize f(x)
    Constraints: 9 inequalities (g_i(x) <= 0)
    Optimal: x* = (1,1,1,1,1,1,1,1,1,3,3,3,1), f(x*) = -15
    """
    x = np.asarray(x, dtype=float)
    if x.size != 13:
        raise ValueError("g01 requires exactly 13 dimensions.")

    f = 5 * np.sum(x[:4]) - 5 * np.sum(x[:4] ** 2) - np.sum(x[4:13])

    g = np.zeros(9)
    g[0] = 2 * x[0] + 2 * x[1] + x[9] + x[10] - 10
    g[1] = 2 * x[0] + 2 * x[2] + x[9] + x[11] - 10
    g[2] = 2 * x[1] + 2 * x[2] + x[10] + x[11] - 10
    g[3] = -8 * x[0] + x[9]
    g[4] = -8 * x[1] + x[10]
    g[5] = -8 * x[2] + x[11]
    g[6] = -2 * x[3] - x[4] + x[9]
    g[7] = -2 * x[5] - x[6] + x[10]
    g[8] = -2 * x[7] - x[8] + x[11]

    return f, g, np.array([])


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


def g12(x: np.ndarray):
    """
    G12 Constrained Benchmark Function.
    Dimension: 3
    Objective: minimize f(x)
    Constraints: 1 inequality (g(x) <= 0), evaluated as the best of 729 disjoint
    spherical feasible regions centered on the integer grid {1,...,9}^3.
    Optimal: x* = (5, 5, 5), f(x*) = -1.0
    """
    x = np.asarray(x, dtype=float)
    if x.size != 3:
        raise ValueError("g12 requires exactly 3 dimensions.")

    f = -(100 - (x[0] - 5) ** 2 - (x[1] - 5) ** 2 - (x[2] - 5) ** 2) / 100

    grid = np.arange(1, 10, dtype=float)
    p, q, r = np.meshgrid(grid, grid, grid, indexing="ij")
    distances_sq = (x[0] - p) ** 2 + (x[1] - q) ** 2 + (x[2] - r) ** 2
    g = np.array([distances_sq.min() - 0.0625])

    return f, g, np.array([])


def g24(x: np.ndarray):
    """
    G24 Constrained Benchmark Function.
    Dimension: 2
    Objective: minimize f(x)
    Constraints: 2 inequalities (g_i(x) <= 0)
    Optimal: x* = (2.329520197, 3.178493496), f(x*) = -5.508013271
    """
    x = np.asarray(x, dtype=float)
    if x.size != 2:
        raise ValueError("g24 requires exactly 2 dimensions.")

    f = -x[0] - x[1]

    g = np.zeros(2)
    g[0] = -2 * x[0] ** 4 + 8 * x[0] ** 3 - 8 * x[0] ** 2 + x[1] - 2
    g[1] = -4 * x[0] ** 4 + 32 * x[0] ** 3 - 88 * x[0] ** 2 + 96 * x[0] + x[1] - 36

    return f, g, np.array([])


def g02(x: np.ndarray):
    """
    G02 Constrained Benchmark Function.
    Dimension: 20
    Objective: minimize f(x)
    Constraints: 2 inequalities (g_i(x) <= 0)
    Optimal: f(x*) = -0.803619104
    """
    x = np.asarray(x, dtype=float)
    if x.size != 20:
        raise ValueError("g02 requires exactly 20 dimensions.")

    n = x.size
    weights = np.arange(1, n + 1, dtype=float)
    numerator = np.sum(np.cos(x) ** 4) - 2 * np.prod(np.cos(x) ** 2)
    denominator = np.sqrt(np.sum(weights * x**2))
    f = -abs(numerator / denominator)

    g = np.zeros(2)
    g[0] = 0.75 - np.prod(x)
    g[1] = np.sum(x) - 7.5 * n

    return f, g, np.array([])


def g03(x: np.ndarray):
    """
    G03 Constrained Benchmark Function.
    Dimension: 10
    Objective: minimize f(x)
    Constraints: 1 equality (h(x) == 0)
    Optimal: f(x*) = -1.000500100
    """
    x = np.asarray(x, dtype=float)
    if x.size != 10:
        raise ValueError("g03 requires exactly 10 dimensions.")

    n = x.size
    f = -(np.sqrt(n) ** n) * np.prod(x)

    h = np.array([np.sum(x**2) - 1])

    return f, np.array([]), h


def g05(x: np.ndarray):
    """
    G05 Constrained Benchmark Function.
    Dimension: 4
    Objective: minimize f(x)
    Constraints: 2 inequalities + 3 equalities
    Optimal: f(x*) = 5126.496714007
    """
    x = np.asarray(x, dtype=float)
    if x.size != 4:
        raise ValueError("g05 requires exactly 4 dimensions.")

    f = 3 * x[0] + 1e-6 * x[0] ** 3 + 2 * x[1] + (2e-6 / 3) * x[1] ** 3

    g = np.zeros(2)
    g[0] = -x[3] + x[2] - 0.55
    g[1] = -x[2] + x[3] - 0.55

    h = np.zeros(3)
    h[0] = 1000 * np.sin(-x[2] - 0.25) + 1000 * np.sin(-x[3] - 0.25) + 894.8 - x[0]
    h[1] = 1000 * np.sin(x[2] - 0.25) + 1000 * np.sin(x[2] - x[3] - 0.25) + 894.8 - x[1]
    h[2] = 1000 * np.sin(x[3] - 0.25) + 1000 * np.sin(x[3] - x[2] - 0.25) + 1294.8

    return f, g, h


def g07(x: np.ndarray):
    """
    G07 Constrained Benchmark Function.
    Dimension: 10
    Objective: minimize f(x)
    Constraints: 8 inequalities
    Optimal: f(x*) = 24.306209068
    """
    x = np.asarray(x, dtype=float)
    if x.size != 10:
        raise ValueError("g07 requires exactly 10 dimensions.")

    f = (
        x[0] ** 2
        + x[1] ** 2
        + x[0] * x[1]
        - 14 * x[0]
        - 16 * x[1]
        + (x[2] - 10) ** 2
        + 4 * (x[3] - 5) ** 2
        + (x[4] - 3) ** 2
        + 2 * (x[5] - 1) ** 2
        + 5 * x[6] ** 2
        + 7 * (x[7] - 11) ** 2
        + 2 * (x[8] - 10) ** 2
        + (x[9] - 7) ** 2
        + 45
    )

    g = np.zeros(8)
    g[0] = -105 + 4 * x[0] + 5 * x[1] - 3 * x[6] + 9 * x[7]
    g[1] = 10 * x[0] - 8 * x[1] - 17 * x[6] + 2 * x[7]
    g[2] = -8 * x[0] + 2 * x[1] + 5 * x[8] - 2 * x[9] - 12
    g[3] = 3 * (x[0] - 2) ** 2 + 4 * (x[1] - 3) ** 2 + 2 * x[2] ** 2 - 7 * x[3] - 120
    g[4] = 5 * x[0] ** 2 + 8 * x[1] + (x[2] - 6) ** 2 - 2 * x[3] - 40
    g[5] = x[0] ** 2 + 2 * (x[1] - 2) ** 2 - 2 * x[0] * x[1] + 14 * x[4] - 6 * x[5]
    g[6] = 0.5 * (x[0] - 8) ** 2 + 2 * (x[1] - 4) ** 2 + 3 * x[4] ** 2 - x[5] - 30
    g[7] = -3 * x[0] + 6 * x[1] + 12 * (x[8] - 8) ** 2 - 7 * x[9]

    return f, g, np.array([])


def g09(x: np.ndarray):
    """
    G09 Constrained Benchmark Function.
    Dimension: 7
    Objective: minimize f(x)
    Constraints: 4 inequalities
    Optimal: f(x*) = 680.630057374
    """
    x = np.asarray(x, dtype=float)
    if x.size != 7:
        raise ValueError("g09 requires exactly 7 dimensions.")

    f = (
        (x[0] - 10) ** 2
        + 5 * (x[1] - 12) ** 2
        + x[2] ** 4
        + 3 * (x[3] - 11) ** 2
        + 10 * x[4] ** 6
        + 7 * x[5] ** 2
        + x[6] ** 4
        - 4 * x[5] * x[6]
        - 10 * x[5]
        - 8 * x[6]
    )

    g = np.zeros(4)
    g[0] = -127 + 2 * x[0] ** 2 + 3 * x[1] ** 4 + x[2] + 4 * x[3] ** 2 + 5 * x[4]
    g[1] = -282 + 7 * x[0] + 3 * x[1] + 10 * x[2] ** 2 + x[3] - x[4]
    g[2] = -196 + 23 * x[0] + x[1] ** 2 + 6 * x[5] ** 2 - 8 * x[6]
    g[3] = 4 * x[0] ** 2 + x[1] ** 2 - 3 * x[0] * x[1] + 2 * x[2] ** 2 + 5 * x[5] - 11 * x[6]

    return f, g, np.array([])


def g10(x: np.ndarray):
    """
    G10 Constrained Benchmark Function.
    Dimension: 8
    Objective: minimize f(x)
    Constraints: 6 inequalities
    Optimal: f(x*) = 7049.248020529
    """
    x = np.asarray(x, dtype=float)
    if x.size != 8:
        raise ValueError("g10 requires exactly 8 dimensions.")

    f = x[0] + x[1] + x[2]

    g = np.zeros(6)
    g[0] = -1 + 0.0025 * (x[3] + x[5])
    g[1] = -1 + 0.0025 * (x[4] + x[6] - x[3])
    g[2] = -1 + 0.01 * (x[7] - x[4])
    g[3] = -x[0] * x[5] + 833.33252 * x[3] + 100 * x[0] - 83333.333
    g[4] = -x[1] * x[6] + 1250 * x[4] + x[1] * x[3] - 1250 * x[3]
    g[5] = -x[2] * x[7] + 1250000 + x[2] * x[4] - 2500 * x[4]

    return f, g, np.array([])


def g13(x: np.ndarray):
    """
    G13 Constrained Benchmark Function.
    Dimension: 5
    Objective: minimize f(x)
    Constraints: 3 equalities
    Optimal: f(x*) = 0.053941514
    """
    x = np.asarray(x, dtype=float)
    if x.size != 5:
        raise ValueError("g13 requires exactly 5 dimensions.")

    f = np.exp(x[0] * x[1] * x[2] * x[3] * x[4])

    h = np.zeros(3)
    h[0] = np.sum(x**2) - 10
    h[1] = x[1] * x[2] - 5 * x[3] * x[4]
    h[2] = x[0] ** 3 + x[1] ** 3 + 1

    return f, np.array([]), h


def g14(x: np.ndarray):
    """
    G14 Constrained Benchmark Function.
    Dimension: 10
    Objective: minimize f(x)
    Constraints: 3 equalities
    Optimal: f(x*) = -47.764888459
    """
    x = np.asarray(x, dtype=float)
    if x.size != 10:
        raise ValueError("g14 requires exactly 10 dimensions.")

    c = np.array(
        [-6.089, -17.164, -34.054, -5.914, -24.721, -14.986, -24.1, -10.708, -26.662, -22.179]
    )
    f = np.sum(x * (c + np.log(x / np.sum(x))))

    h = np.zeros(3)
    h[0] = x[0] + 2 * x[1] + 2 * x[2] + x[5] + x[9] - 2
    h[1] = x[3] + 2 * x[4] + x[5] + x[6] - 1
    h[2] = x[2] + x[6] + x[7] + 2 * x[8] + x[9] - 1

    return f, np.array([]), h


def g15(x: np.ndarray):
    """
    G15 Constrained Benchmark Function.
    Dimension: 3
    Objective: minimize f(x)
    Constraints: 2 equalities
    Optimal: f(x*) = 961.715022290
    """
    x = np.asarray(x, dtype=float)
    if x.size != 3:
        raise ValueError("g15 requires exactly 3 dimensions.")

    f = 1000 - x[0] ** 2 - 2 * x[1] ** 2 - x[2] ** 2 - x[0] * x[1] - x[0] * x[2]

    h = np.zeros(2)
    h[0] = x[0] ** 2 + x[1] ** 2 + x[2] ** 2 - 25
    h[1] = 8 * x[0] + 14 * x[1] + 7 * x[2] - 56

    return f, np.array([]), h


def g16(x: np.ndarray):
    """
    G16 Constrained Benchmark Function.
    Dimension: 5
    Objective: minimize f(x)
    Constraints: 38 inequalities
    Optimal: f(x*) = -1.905155259

    Uses the intermediate-variable names (y1..y17, c1..c17) from the source
    report directly, 1-indexed with index 0 unused, so the code can be
    checked line-by-line against the paper.
    """
    x = np.asarray(x, dtype=float)
    if x.size != 5:
        raise ValueError("g16 requires exactly 5 dimensions.")

    x1, x2, x3, x4, x5 = x

    y = np.zeros(18)
    c = np.zeros(18)

    y[1] = x2 + x3 + 41.6
    c[1] = 0.024 * x4 - 4.62
    y[2] = 12.5 / c[1] + 12
    c[2] = 0.0003535 * x1**2 + 0.5311 * x1 + 0.08705 * y[2] * x1
    c[3] = 0.052 * x1 + 78 + 0.002377 * y[2] * x1
    y[3] = c[2] / c[3]
    y[4] = 19 * y[3]
    c[4] = 0.04782 * (x1 - y[3]) + (0.1956 * (x1 - y[3]) ** 2) / x2 + 0.6376 * y[4] + 1.594 * y[3]
    c[5] = 100 * x2
    c[6] = x1 - y[3] - y[4]
    c[7] = 0.950 - c[4] / c[5]
    y[5] = c[6] * c[7]
    y[6] = x1 - y[5] - y[4] - y[3]
    c[8] = (y[5] + y[4]) * 0.995
    y[7] = c[8] / y[1]
    y[8] = c[8] / 3798
    c[9] = y[7] - (0.0663 * y[7]) / y[8] - 0.3153
    y[9] = 96.82 / c[9] + 0.321 * y[1]
    y[10] = 1.29 * y[5] + 1.258 * y[4] + 2.29 * y[3] + 1.71 * y[6]
    y[11] = 1.71 * x1 - 0.452 * y[4] + 0.580 * y[3]
    c[10] = 12.3 / 752.3
    c[11] = 1.75 * y[2] * 0.995 * x1
    c[12] = 0.995 * y[10] + 1998
    y[12] = c[10] * x1 + c[11] / c[12]
    y[13] = c[12] - 1.75 * y[2]
    y[14] = 3623 + 64.4 * x2 + 58.4 * x3 + 146312 / (y[9] + x5)
    c[13] = 0.995 * y[10] + 60.8 * x2 + 48 * x4 - 0.1121 * y[14] - 5095
    y[15] = y[13] / c[13]
    y[16] = 148000 - 331000 * y[15] + 40 * y[13] - 61 * y[15] * y[13]
    c[14] = 2324 * y[10] - 28740000 * y[2]
    y[17] = 14130000 - 1328 * y[10] - 531 * y[11] + c[14] / c[12]
    c[15] = y[13] / y[15] - y[13] / 0.52
    c[16] = 1.104 - 0.72 * y[15]
    c[17] = y[9] + x5

    f = (
        0.000117 * y[14]
        + 0.1365
        + 0.00002358 * y[13]
        + 0.000001502 * y[16]
        + 0.0321 * y[12]
        + 0.004324 * y[5]
        + 0.0001 * (c[15] / c[16])
        + 37.48 * (y[2] / c[12])
        - 0.0000005843 * y[17]
    )

    g = np.zeros(38)
    g[0] = (0.28 / 0.72) * y[5] - y[4]
    g[1] = x3 - 1.5 * x2
    g[2] = 3496 * (y[2] / c[12]) - 21
    g[3] = 110.6 + y[1] - (62212 / c[17])
    g[4] = 213.1 - y[1]
    g[5] = y[1] - 405.23
    g[6] = 17.505 - y[2]
    g[7] = y[2] - 1053.6667
    g[8] = 11.275 - y[3]
    g[9] = y[3] - 35.03
    g[10] = 214.228 - y[4]
    g[11] = y[4] - 665.585
    g[12] = 7.458 - y[5]
    g[13] = y[5] - 584.463
    g[14] = 0.961 - y[6]
    g[15] = y[6] - 265.916
    g[16] = 1.612 - y[7]
    g[17] = y[7] - 7.046
    g[18] = 0.146 - y[8]
    g[19] = y[8] - 0.222
    g[20] = 107.99 - y[9]
    g[21] = y[9] - 273.366
    g[22] = 922.693 - y[10]
    g[23] = y[10] - 1286.105
    g[24] = 926.832 - y[11]
    g[25] = y[11] - 1444.046
    g[26] = 18.766 - y[12]
    g[27] = y[12] - 537.141
    g[28] = 1072.163 - y[13]
    g[29] = y[13] - 3247.039
    g[30] = 8961.448 - y[14]
    g[31] = y[14] - 26844.086
    g[32] = 0.063 - y[15]
    g[33] = y[15] - 0.386
    g[34] = 71084.33 - y[16]
    g[35] = y[16] - 140000
    g[36] = 2802713 - y[17]
    g[37] = y[17] - 12146108

    return f, g, np.array([])


def g17(x: np.ndarray):
    """
    G17 Constrained Benchmark Function.
    Dimension: 6
    Objective: minimize f(x)
    Constraints: 4 equalities
    Optimal: f(x*) = 8853.539674806
    """
    x = np.asarray(x, dtype=float)
    if x.size != 6:
        raise ValueError("g17 requires exactly 6 dimensions.")

    x1, x2, x3, x4, x5, x6 = x

    f1 = 30 * x1 if 0 <= x1 < 300 else 31 * x1
    if 0 <= x2 < 100:
        f2 = 28 * x2
    elif x2 < 200:
        f2 = 29 * x2
    else:
        f2 = 30 * x2
    f = f1 + f2

    h = np.zeros(4)
    h[0] = (
        -x1
        + 300
        - (x3 * x4 / 131.078) * np.cos(1.48477 - x6)
        + (0.90798 * x3**2 / 131.078) * np.cos(1.47588)
    )
    h[1] = (
        -x2
        - (x3 * x4 / 131.078) * np.cos(1.48477 + x6)
        + (0.90798 * x4**2 / 131.078) * np.cos(1.47588)
    )
    h[2] = (
        -x5
        - (x3 * x4 / 131.078) * np.sin(1.48477 + x6)
        + (0.90798 * x4**2 / 131.078) * np.sin(1.47588)
    )
    h[3] = (
        200
        - (x3 * x4 / 131.078) * np.sin(1.48477 - x6)
        + (0.90798 * x3**2 / 131.078) * np.sin(1.47588)
    )

    return f, np.array([]), h


def g18(x: np.ndarray):
    """
    G18 Constrained Benchmark Function.
    Dimension: 9
    Objective: minimize f(x)
    Constraints: 13 inequalities
    Optimal: f(x*) = -0.866025404
    """
    x = np.asarray(x, dtype=float)
    if x.size != 9:
        raise ValueError("g18 requires exactly 9 dimensions.")

    f = -0.5 * (x[0] * x[3] - x[1] * x[2] + x[2] * x[8] - x[4] * x[8] + x[4] * x[7] - x[5] * x[6])

    g = np.zeros(13)
    g[0] = x[2] ** 2 + x[3] ** 2 - 1
    g[1] = x[8] ** 2 - 1
    g[2] = x[4] ** 2 + x[5] ** 2 - 1
    g[3] = x[0] ** 2 + (x[1] - x[8]) ** 2 - 1
    g[4] = (x[0] - x[4]) ** 2 + (x[1] - x[5]) ** 2 - 1
    g[5] = (x[0] - x[6]) ** 2 + (x[1] - x[7]) ** 2 - 1
    g[6] = (x[2] - x[4]) ** 2 + (x[3] - x[5]) ** 2 - 1
    g[7] = (x[2] - x[6]) ** 2 + (x[3] - x[7]) ** 2 - 1
    g[8] = x[6] ** 2 + (x[7] - x[8]) ** 2 - 1
    g[9] = x[1] * x[2] - x[0] * x[3]
    g[10] = -x[2] * x[8]
    g[11] = x[4] * x[8]
    g[12] = x[5] * x[6] - x[4] * x[7]

    return f, g, np.array([])


def g19(x: np.ndarray):
    """
    G19 Constrained Benchmark Function.
    Dimension: 15
    Objective: minimize f(x)
    Constraints: 5 inequalities
    Optimal: f(x*) = 32.655592950
    """
    x = np.asarray(x, dtype=float)
    if x.size != 15:
        raise ValueError("g19 requires exactly 15 dimensions.")

    b = np.array([-40.0, -2.0, -0.25, -4.0, -4.0, -1.0, -40.0, -60.0, 5.0, 1.0])
    c = np.array(
        [
            [30, -20, -10, 32, -10],
            [-20, 39, -6, -31, 32],
            [-10, -6, 10, -6, -10],
            [32, -31, -6, 39, -20],
            [-10, 32, -10, -20, 30],
        ],
        dtype=float,
    )
    d = np.array([4.0, 8.0, 10.0, 6.0, 2.0])
    e = np.array([-15.0, -27.0, -36.0, -18.0, -12.0])
    a = np.array(
        [
            [-16, 2, 0, 1, 0],
            [0, -2, 0, 0.4, 2],
            [-3.5, 0, 2, 0, 0],
            [0, -2, 0, -4, -1],
            [0, -9, -2, 1, -2.8],
            [2, 0, -4, 0, 0],
            [-1, -1, -1, -1, -1],
            [-1, -2, -3, -2, -1],
            [1, 2, 3, 4, 5],
            [1, 1, 1, 1, 1],
        ],
        dtype=float,
    )

    x10, y = x[:10], x[10:]

    f = y @ c @ y + 2 * np.sum(d * y**3) - np.sum(b * x10)
    g = -2 * (y @ c) - 3 * d * y**2 - e + x10 @ a

    return f, g, np.array([])


def g20(x: np.ndarray):
    """
    G20 Constrained Benchmark Function.
    Dimension: 24
    Objective: minimize f(x)
    Constraints: 6 inequalities + 14 equalities
    Optimal: f(x*) ~= 0.204979400 -- per the source technical report, this
    "best known" point is itself slightly infeasible and no fully feasible
    solution has ever been verified for g20. Treat as indicative only.
    """
    x = np.asarray(x, dtype=float)
    if x.size != 24:
        raise ValueError("g20 requires exactly 24 dimensions.")

    a = np.array([0.0693, 0.0577, 0.05, 0.2, 0.26, 0.55, 0.06, 0.1, 0.12, 0.18, 0.1, 0.09] * 2)
    b = np.array(
        [44.094, 58.12, 58.12, 137.4, 120.9, 170.9, 62.501, 84.94, 133.425, 82.507, 46.07, 60.097]
        * 2
    )
    c = np.array([123.7, 31.7, 45.7, 14.7, 84.7, 27.7, 49.7, 7.1, 2.1, 17.7, 0.85, 0.64])
    d = np.array([31.244, 36.12, 34.784, 92.7, 82.7, 91.6, 56.708, 82.7, 80.8, 64.517, 49.4, 49.1])
    e = np.array([0.1, 0.3, 0.4, 0.3, 0.6, 0.3])

    f = np.sum(a * x)

    total = np.sum(x)
    g = np.zeros(6)
    g[0] = (x[0] + x[12]) / (total + e[0])
    g[1] = (x[1] + x[13]) / (total + e[1])
    g[2] = (x[2] + x[14]) / (total + e[2])
    g[3] = (x[6] + x[18]) / (total + e[3])
    g[4] = (x[7] + x[19]) / (total + e[4])
    g[5] = (x[8] + x[20]) / (total + e[5])

    s1 = np.sum(x[:12] / b[:12])
    s2 = np.sum(x[12:] / b[12:])
    h = np.zeros(14)
    h[:12] = x[12:] / (b[12:] * s2) - (c * x[:12]) / (40 * b[:12] * s1)
    h[12] = total - 1
    k = 0.7302 * 530 * (14.7 / 40)
    h[13] = np.sum(x[:12] / d) + k * np.sum(x[12:] / b[12:]) - 1.671

    return f, g, h


def g21(x: np.ndarray):
    """
    G21 Constrained Benchmark Function.
    Dimension: 7
    Objective: minimize f(x)
    Constraints: 1 inequality + 5 equalities
    Optimal: f(x*) = 193.724510070
    """
    x = np.asarray(x, dtype=float)
    if x.size != 7:
        raise ValueError("g21 requires exactly 7 dimensions.")

    f = x[0]

    g = np.array([-x[0] + 35 * x[1] ** 0.6 + 35 * x[2] ** 0.6])

    h = np.zeros(5)
    h[0] = (
        -300 * x[2] + 7500 * x[4] - 7500 * x[5] - 25 * x[3] * x[4] + 25 * x[3] * x[5] + x[2] * x[3]
    )
    h[1] = 100 * x[1] + 155.365 * x[3] + 2500 * x[6] - x[1] * x[3] - 25 * x[3] * x[6] - 15536.5
    h[2] = -x[4] + np.log(-x[3] + 900)
    h[3] = -x[5] + np.log(x[3] + 300)
    h[4] = -x[6] + np.log(-2 * x[3] + 700)

    return f, g, h


def g22(x: np.ndarray):
    """
    G22 Constrained Benchmark Function.
    Dimension: 22
    Objective: minimize f(x)
    Constraints: 1 inequality + 19 equalities
    Optimal: f(x*) = 236.430975504
    """
    x = np.asarray(x, dtype=float)
    if x.size != 22:
        raise ValueError("g22 requires exactly 22 dimensions.")

    (
        x1,
        x2,
        x3,
        x4,
        x5,
        x6,
        x7,
        x8,
        x9,
        x10,
        x11,
        x12,
        x13,
        x14,
        x15,
        x16,
        x17,
        x18,
        x19,
        x20,
        x21,
        x22,
    ) = x

    f = x1

    g = np.array([-x1 + x2**0.6 + x3**0.6 + x4**0.6])

    h = np.zeros(19)
    h[0] = x5 - 100000 * x8 + 1e7
    h[1] = x6 + 100000 * x8 - 100000 * x9
    h[2] = x7 + 100000 * x9 - 5e7
    h[3] = x5 + 100000 * x10 - 3.3e7
    h[4] = x6 + 100000 * x11 - 4.4e7
    h[5] = x7 + 100000 * x12 - 6.6e7
    h[6] = x5 - 120 * x2 * x13
    h[7] = x6 - 80 * x3 * x14
    h[8] = x7 - 40 * x4 * x15
    h[9] = x8 - x11 + x16
    h[10] = x9 - x12 + x17
    h[11] = -x18 + np.log(x10 - 100)
    h[12] = -x19 + np.log(-x8 + 300)
    h[13] = -x20 + np.log(x16)
    h[14] = -x21 + np.log(-x9 + 400)
    h[15] = -x22 + np.log(x17)
    h[16] = -x8 - x10 + x13 * x18 - x13 * x19 + 400
    h[17] = x8 - x9 - x11 + x14 * x20 - x14 * x21 + 400
    h[18] = x9 - x12 - 4.60517 * x15 + x15 * x22 + 100

    return f, g, h


def g23(x: np.ndarray):
    """
    G23 Constrained Benchmark Function.
    Dimension: 9
    Objective: minimize f(x)
    Constraints: 2 inequalities + 4 equalities
    Optimal: f(x*) = -400.055100000
    """
    x = np.asarray(x, dtype=float)
    if x.size != 9:
        raise ValueError("g23 requires exactly 9 dimensions.")

    f = -9 * x[4] - 15 * x[7] + 6 * x[0] + 16 * x[1] + 10 * (x[5] + x[6])

    g = np.zeros(2)
    g[0] = x[8] * x[2] + 0.02 * x[5] - 0.025 * x[4]
    g[1] = x[8] * x[3] + 0.02 * x[6] - 0.015 * x[7]

    h = np.zeros(4)
    h[0] = x[0] + x[1] - x[2] - x[3]
    h[1] = 0.03 * x[0] + 0.01 * x[1] - x[8] * (x[2] + x[3])
    h[2] = x[2] + x[5] - x[4]
    h[3] = x[3] + x[6] - x[7]

    return f, g, h


class _GEntry(TypedDict):
    function: Callable[[np.ndarray], Tuple[float, np.ndarray, np.ndarray]]
    dim: int
    n_inequality: int
    n_equality: int
    known_minimum: float


G_SUITE: Dict[str, _GEntry] = {
    "g01": {"function": g01, "dim": 13, "n_inequality": 9, "n_equality": 0, "known_minimum": -15.0},
    "g02": {
        "function": g02,
        "dim": 20,
        "n_inequality": 2,
        "n_equality": 0,
        "known_minimum": -0.803619104,
    },
    "g03": {
        "function": g03,
        "dim": 10,
        "n_inequality": 0,
        "n_equality": 1,
        "known_minimum": -1.000500100,
    },
    "g04": {
        "function": g04,
        "dim": 5,
        "n_inequality": 6,
        "n_equality": 0,
        "known_minimum": -30665.539,
    },
    "g05": {
        "function": g05,
        "dim": 4,
        "n_inequality": 2,
        "n_equality": 3,
        "known_minimum": 5126.496714007,
    },
    "g06": {
        "function": g06,
        "dim": 2,
        "n_inequality": 2,
        "n_equality": 0,
        "known_minimum": -6961.81388,
    },
    "g07": {
        "function": g07,
        "dim": 10,
        "n_inequality": 8,
        "n_equality": 0,
        "known_minimum": 24.306209068,
    },
    "g08": {
        "function": g08,
        "dim": 2,
        "n_inequality": 2,
        "n_equality": 0,
        "known_minimum": -0.095825,
    },
    "g09": {
        "function": g09,
        "dim": 7,
        "n_inequality": 4,
        "n_equality": 0,
        "known_minimum": 680.630057374,
    },
    "g10": {
        "function": g10,
        "dim": 8,
        "n_inequality": 6,
        "n_equality": 0,
        "known_minimum": 7049.248020529,
    },
    "g11": {"function": g11, "dim": 2, "n_inequality": 0, "n_equality": 1, "known_minimum": 0.7499},
    "g12": {"function": g12, "dim": 3, "n_inequality": 1, "n_equality": 0, "known_minimum": -1.0},
    "g13": {
        "function": g13,
        "dim": 5,
        "n_inequality": 0,
        "n_equality": 3,
        "known_minimum": 0.053941514,
    },
    "g14": {
        "function": g14,
        "dim": 10,
        "n_inequality": 0,
        "n_equality": 3,
        "known_minimum": -47.764888459,
    },
    "g15": {
        "function": g15,
        "dim": 3,
        "n_inequality": 0,
        "n_equality": 2,
        "known_minimum": 961.715022290,
    },
    "g16": {
        "function": g16,
        "dim": 5,
        "n_inequality": 38,
        "n_equality": 0,
        "known_minimum": -1.905155259,
    },
    "g17": {
        "function": g17,
        "dim": 6,
        "n_inequality": 0,
        "n_equality": 4,
        "known_minimum": 8853.539674806,
    },
    "g18": {
        "function": g18,
        "dim": 9,
        "n_inequality": 13,
        "n_equality": 0,
        "known_minimum": -0.866025404,
    },
    "g19": {
        "function": g19,
        "dim": 15,
        "n_inequality": 5,
        "n_equality": 0,
        "known_minimum": 32.655592950,
    },
    "g20": {
        "function": g20,
        "dim": 24,
        "n_inequality": 6,
        "n_equality": 14,
        "known_minimum": 0.204979400,
    },
    "g21": {
        "function": g21,
        "dim": 7,
        "n_inequality": 1,
        "n_equality": 5,
        "known_minimum": 193.724510070,
    },
    "g22": {
        "function": g22,
        "dim": 22,
        "n_inequality": 1,
        "n_equality": 19,
        "known_minimum": 236.430975504,
    },
    "g23": {
        "function": g23,
        "dim": 9,
        "n_inequality": 2,
        "n_equality": 4,
        "known_minimum": -400.055100000,
    },
    "g24": {
        "function": g24,
        "dim": 2,
        "n_inequality": 2,
        "n_equality": 0,
        "known_minimum": -5.508013271,
    },
}


def get_g_function_list() -> List[str]:
    """Return the names of every function in ``G_SUITE``."""
    return list(G_SUITE.keys())
