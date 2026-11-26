"""
Classic engineering design suite.

This module implements four widely-used mechanical/structural design
benchmarks for constrained optimization: tension/compression spring design
(Belegundu 1982), pressure vessel design (Kannan and Kramer 1994), welded
beam design (Ragsdell and Phillips 1976), and gear train design
(Sandgren 1990). Unlike the CEC2006 G-suite (one shared source report),
each problem here comes from its own original paper, so every formula was
cross-checked against at least two independent transcriptions before
landing -- see each function's docstring for its specific sources.

Following ``g_suite.py``'s convention, each function returns the objective
value and a tuple of constraint violations (g(x) <= 0 for inequality,
h(x) == 0 for equality); none of these four problems has an equality
constraint, so the second element of every return is always an empty array.

``gear_train`` is the one exception worth flagging: Sandgren's original
formulation treats its four variables as integer tooth counts with no
constraint beyond the box bounds. This module evaluates the formula at
whatever real-valued ``x`` is passed in (consistent with every other
function in the package taking a plain ``np.ndarray``) rather than
rounding internally -- callers wanting the true discrete benchmark should
round to the nearest integer themselves before evaluating.

``ENGINEERING_SUITE`` is a lightweight registry (function, dim, constraint
counts, known minimum), deliberately the same shape as ``g_suite.G_SUITE``
and for the same reason: bounds vary per-variable per-problem in ways that
don't fit one schema, so they're documented in each function's own
docstring rather than in the registry. ``get_engineering_function_list()``
is the accessor.
"""

from typing import Callable, Dict, List, Tuple, TypedDict

import numpy as np


def spring_design(x: np.ndarray):
    """
    Tension/Compression Spring Design.
    Dimension: 3 (x1 = wire diameter d, x2 = mean coil diameter D,
    x3 = number of active coils N)
    Objective: minimize f(x) = (x3 + 2) * x2 * x1^2 (spring weight)
    Constraints: 4 inequalities (g_i(x) <= 0) -- minimum deflection, shear
    stress, surge frequency, and outer diameter limit.
    Bounds: 0.05 <= x1 <= 2.0, 0.25 <= x2 <= 1.3, 2.0 <= x3 <= 15.0.
    Optimal: x* = (0.051690, 0.356750, 11.287126), f(x*) = 0.012665.

    Source: Belegundu, A. (1982), "A Study of Mathematical Programming
    Methods for Structural Optimization", PhD thesis, University of Iowa.
    Formula and optimum cross-checked against Yang & Deb (2010),
    "Engineering Optimisation by Cuckoo Search", Int. J. Mathematical
    Modelling and Numerical Optimisation, 1(4), 330-343 (which itself
    reproduces the value from Cagnina, Esquivel & Coello (2008)).
    """
    x = np.asarray(x, dtype=float)
    if x.size != 3:
        raise ValueError("spring_design requires exactly 3 dimensions.")
    d, dd, n = x

    f = (n + 2) * dd * d**2

    g = np.zeros(4)
    g[0] = 1 - (dd**3 * n) / (71785 * d**4)
    g[1] = (4 * dd**2 - d * dd) / (12566 * (d**3 * dd - d**4)) + 1 / (5108 * d**2) - 1
    g[2] = 1 - (140.45 * d) / (dd**2 * n)
    g[3] = (d + dd) / 1.5 - 1

    return float(f), g, np.array([])


def pressure_vessel(x: np.ndarray):
    """
    Pressure Vessel Design.
    Dimension: 4 (x1 = shell thickness Ts, x2 = head thickness Th,
    x3 = inner radius R, x4 = length of cylindrical section L)
    Objective: minimize f(x) = 0.6224*x1*x3*x4 + 1.7781*x2*x3^2
    + 3.1661*x1^2*x4 + 19.84*x1^2*x3 (fabrication cost)
    Constraints: 4 inequalities (g_i(x) <= 0).
    Bounds: 0.0625 <= x1, x2 <= 6.1875 (integer multiples of 0.0625 in the
    original mixed-integer formulation; evaluated here as continuous),
    10.0 <= x3, x4 <= 200.0.
    Optimal: x* = (0.8125, 0.4375, 42.0984455958549, 176.6365958424394),
    f(x*) = 6059.714335048436.

    Source: Kannan, B. K. and Kramer, S. N. (1994), "An Augmented Lagrange
    Multiplier Based Method for Mixed Integer Discrete Continuous
    Optimization and its Applications to Mechanical Design", J. Mech. Des.,
    116, 318-320. The optimum above is not merely the best found in
    literature but a proven global optimum: Yang, Huyck, Karamanoglu &
    Khan (2013), "True Global Optimality of the Pressure Vessel Design
    Problem", Int. J. Bio-Inspired Computation, 5(6), 329-335, derive it
    analytically (Lagrange multipliers) and show it improves on the
    previous best-known value of 6059.714 (Cagnina et al. 2008).
    """
    x = np.asarray(x, dtype=float)
    if x.size != 4:
        raise ValueError("pressure_vessel requires exactly 4 dimensions.")
    ts, th, r, ln = x

    f = 0.6224 * ts * r * ln + 1.7781 * th * r**2 + 3.1661 * ts**2 * ln + 19.84 * ts**2 * r

    g = np.zeros(4)
    g[0] = -ts + 0.0193 * r
    g[1] = -th + 0.00954 * r
    g[2] = -np.pi * r**2 * ln - (4 / 3) * np.pi * r**3 + 1296000
    g[3] = ln - 240

    return float(f), g, np.array([])


def welded_beam(x: np.ndarray):
    """
    Welded Beam Design.
    Dimension: 4 (x1 = weld width w, x2 = weld length L,
    x3 = bar thickness d, x4 = bar height h)
    Objective: minimize f(x) = 1.10471*x1^2*x2 + 0.04811*x3*x4*(14.0 + x2)
    (fabrication cost)
    Constraints: 7 inequalities (g_i(x) <= 0) -- shear stress, bending
    stress, geometry, buckling load, and deflection.
    Bounds: 0.1 <= x2, x3 <= 10.0, 0.1 <= x1, x4 <= 2.0.
    Optimal: x* = (0.205729639786079, 3.470488665627977, 9.036623910357633,
    0.205729639786079), f(x*) = 1.724852308597361.

    Source: Ragsdell, K. and Phillips, D. (1976), "Optimal Design of a
    Class of Welded Structures Using Geometric Programming", J. Eng. Ind.,
    98, 1021-1025. Formula, auxiliary stress/deflection equations, and
    optimum cross-checked against Yang & Deb (2010), "Engineering
    Optimisation by Cuckoo Search" (whose result matches Cagnina, Esquivel
    & Coello (2008) to 12 significant figures).
    """
    x = np.asarray(x, dtype=float)
    if x.size != 4:
        raise ValueError("welded_beam requires exactly 4 dimensions.")
    w, ln, d, h = x

    f = 1.10471 * w**2 * ln + 0.04811 * d * h * (14.0 + ln)

    sigma = 504000 / (h * d**2)
    q = 6000 * (14 + ln / 2)
    d_diag = 0.5 * np.sqrt(ln**2 + (w + d) ** 2)
    j = np.sqrt(2) * w * ln * (ln**2 / 6 + (w + d) ** 2 / 2)
    delta = 65856 / (30000 * h * d**3)
    beta = q * d_diag / j
    alpha = 6000 / (np.sqrt(2) * w * ln)
    tau = np.sqrt(alpha**2 + alpha * beta * ln / d_diag + beta**2)
    p = 0.61423e6 * d * h**3 / 6 * (1 - d * np.sqrt(30 / 48) / 28)

    g = np.zeros(7)
    g[0] = w - h
    g[1] = delta - 0.25
    g[2] = tau - 13600
    g[3] = sigma - 30000
    g[4] = 0.10471 * w**2 + 0.04811 * h * d * (14 + ln) - 5.0
    g[5] = 0.125 - w
    g[6] = 6000 - p

    return float(f), g, np.array([])


def gear_train(x: np.ndarray):
    """
    Gear Train Design.
    Dimension: 4 (x1, x2, x3, x4 = number of teeth on each of four gears)
    Objective: minimize f(x) = (1/6.931 - (x1*x2)/(x3*x4))^2, the squared
    error between the achieved and desired (1/6.931) gear ratio.
    Constraints: none beyond the box bounds -- unlike the other three
    problems in this module, this one has no inequality/equality
    constraints in its original formulation.
    Bounds: 12 <= x1, x2, x3, x4 <= 60 (integers in the original
    mixed-integer formulation; evaluated here as continuous -- see the
    module docstring).
    Optimal: x* = (19, 16, 43, 49) (or any permutation preserving the
    {x1,x2} vs {x3,x4} product grouping), f(x*) ~= 2.7e-12.

    Source: Sandgren, E. (1990), "Nonlinear Integer and Discrete
    Programming in Mechanical Design Optimization", J. Mech. Des., 112(2),
    223-229. Formula and optimum cross-checked numerically: 19*16=304,
    43*49=2107, and (304/2107 - 1/6.931)^2 ~= 2.7e-12, matching the widely
    reproduced literature value.
    """
    x = np.asarray(x, dtype=float)
    if x.size != 4:
        raise ValueError("gear_train requires exactly 4 dimensions.")
    x1, x2, x3, x4 = x

    f = (1 / 6.931 - (x1 * x2) / (x3 * x4)) ** 2

    return float(f), np.array([]), np.array([])


class _EngEntry(TypedDict):
    function: Callable[[np.ndarray], Tuple[float, np.ndarray, np.ndarray]]
    dim: int
    n_inequality: int
    n_equality: int
    known_minimum: float


ENGINEERING_SUITE: Dict[str, _EngEntry] = {
    "spring_design": {
        "function": spring_design,
        "dim": 3,
        "n_inequality": 4,
        "n_equality": 0,
        "known_minimum": 0.012665,
    },
    "pressure_vessel": {
        "function": pressure_vessel,
        "dim": 4,
        "n_inequality": 4,
        "n_equality": 0,
        "known_minimum": 6059.714335048436,
    },
    "welded_beam": {
        "function": welded_beam,
        "dim": 4,
        "n_inequality": 7,
        "n_equality": 0,
        "known_minimum": 1.724852308597361,
    },
    "gear_train": {
        "function": gear_train,
        "dim": 4,
        "n_inequality": 0,
        "n_equality": 0,
        "known_minimum": 2.7e-12,
    },
}


def get_engineering_function_list() -> List[str]:
    """Return the names of every function in ``ENGINEERING_SUITE``."""
    return list(ENGINEERING_SUITE.keys())
