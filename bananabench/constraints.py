"""
Constraint-handling utilities for the G-suite (``g_suite.py``) and the
engineering design suite (``engineering.py``).

Both suites return ``(objective, inequality_violations, equality_violations)``
following the same convention: inequality entries are raw ``g(x) <= 0``
values (feasible when ``<= 0``, violated by ``g(x)`` when positive) and
equality entries are raw ``h(x) == 0`` values (satisfied within a numerical
tolerance of zero, never exactly zero for a floating-point evaluation).

``BenchmarkRunner`` assumes a scalar cost, so it doesn't fit this tuple
shape -- ranking two constrained-optimization results needs a rule beyond
"lower objective wins," since an infeasible point with a great objective
value is not actually a good solution. This module implements the
feasibility rule nearly every paper reporting results on these suites
already relies on implicitly:

    Deb, K. (2000). "An efficient constraint handling method for genetic
    algorithms." Computer Methods in Applied Mechanics and Engineering,
    186(2-4), 311-338.

1. Any feasible solution is preferred over any infeasible one.
2. Between two feasible solutions, the one with the better (lower)
   objective value wins.
3. Between two infeasible solutions, the one with the smaller total
   constraint violation wins.
"""

from typing import Sequence, Tuple

import numpy as np

__all__ = ["constraint_violation", "is_feasible", "feasibility_compare"]

# The `(objective, inequality_violations, equality_violations)` shape every
# g_suite/engineering function returns.
ConstrainedResult = Tuple[float, Sequence[float], Sequence[float]]


def constraint_violation(
    inequality_violations: Sequence[float] = (),
    equality_violations: Sequence[float] = (),
    tol: float = 1e-6,
) -> float:
    """
    Total constraint violation magnitude for one evaluation, following
    Deb (2000)'s convention: the sum of each violated inequality's positive
    excess (``g_i(x)`` for every ``g_i(x) > tol``) plus each violated
    equality constraint's absolute deviation from zero (``abs(h_j(x))`` for
    every ``abs(h_j(x)) > tol``).

    A return value of ``0.0`` means every constraint is satisfied within
    tolerance -- i.e. the point is feasible (see `is_feasible`).

    Parameters
    ----------
    inequality_violations : sequence of float
        Raw ``g_i(x)`` values (feasible when ``<= 0``), as returned by any
        `g_suite`/`engineering` function.
    equality_violations : sequence of float
        Raw ``h_j(x)`` values (feasible when ``== 0``).
    tol : float
        Numerical tolerance below which a constraint counts as satisfied.
        Needed for equality constraints in particular, since a
        floating-point evaluation essentially never lands on exactly 0.0.
    """
    g = np.asarray(inequality_violations, dtype=float)
    h = np.asarray(equality_violations, dtype=float)

    ineq_total = float(np.sum(g[g > tol])) if g.size else 0.0
    eq_total = float(np.sum(np.abs(h)[np.abs(h) > tol])) if h.size else 0.0
    return ineq_total + eq_total


def is_feasible(
    inequality_violations: Sequence[float] = (),
    equality_violations: Sequence[float] = (),
    tol: float = 1e-6,
) -> bool:
    """
    True iff every inequality constraint is ``<= tol`` and every equality
    constraint's absolute value is ``<= tol``. See `constraint_violation`
    for the parameters' meaning.
    """
    g = np.asarray(inequality_violations, dtype=float)
    h = np.asarray(equality_violations, dtype=float)

    if g.size and np.any(g > tol):
        return False
    if h.size and np.any(np.abs(h) > tol):
        return False
    return True


def feasibility_compare(a: ConstrainedResult, b: ConstrainedResult, tol: float = 1e-6) -> int:
    """
    Compare two `(objective, inequality_violations, equality_violations)`
    results -- the return shape of every `g_suite`/`engineering` function --
    using Deb (2000)'s feasibility rule.

    Returns
    -------
    int
        -1 if `a` is preferred over `b`, 1 if `b` is preferred over `a`,
        0 if they tie exactly (same feasibility status and, depending on
        that status, the same objective or the same total violation).
    """
    obj_a, ineq_a, eq_a = a
    obj_b, ineq_b, eq_b = b

    feasible_a = is_feasible(ineq_a, eq_a, tol)
    feasible_b = is_feasible(ineq_b, eq_b, tol)

    if feasible_a and not feasible_b:
        return -1
    if feasible_b and not feasible_a:
        return 1

    if feasible_a and feasible_b:
        if obj_a < obj_b:
            return -1
        if obj_b < obj_a:
            return 1
        return 0

    # Both infeasible: smaller total constraint violation wins.
    viol_a = constraint_violation(ineq_a, eq_a, tol)
    viol_b = constraint_violation(ineq_b, eq_b, tol)
    if viol_a < viol_b:
        return -1
    if viol_b < viol_a:
        return 1
    return 0
