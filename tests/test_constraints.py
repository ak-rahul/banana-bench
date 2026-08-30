"""
Tests for bananabench.constraints: Deb (2000)'s feasibility rule for ranking
(objective, inequality_violations, equality_violations) results from the
g_suite/engineering suites.
"""

import numpy as np
import pytest

from bananabench import engineering, g_suite
from bananabench.constraints import constraint_violation, feasibility_compare, is_feasible


class TestConstraintViolation:
    def test_all_satisfied_is_zero(self):
        assert constraint_violation([-1.0, -0.5], [0.0]) == 0.0

    def test_no_constraints_is_zero(self):
        assert constraint_violation([], []) == 0.0
        assert constraint_violation() == 0.0

    def test_sums_only_the_excess_above_tolerance(self):
        # g = [-1 (ok), 2 (violated by 2), 0.5 (violated by 0.5)]
        assert constraint_violation([-1.0, 2.0, 0.5]) == pytest.approx(2.5)

    def test_equality_uses_absolute_deviation(self):
        assert constraint_violation([], [0.3, -0.2]) == pytest.approx(0.5)

    def test_within_tolerance_does_not_count(self):
        assert constraint_violation([1e-9], [1e-9], tol=1e-6) == 0.0

    def test_mixed_inequality_and_equality(self):
        assert constraint_violation([1.0, -5.0], [2.0]) == pytest.approx(3.0)


class TestIsFeasible:
    def test_all_satisfied(self):
        assert is_feasible([-1.0, -0.001], [0.0])

    def test_violated_inequality(self):
        assert not is_feasible([0.5], [])

    def test_violated_equality(self):
        assert not is_feasible([], [0.1])

    def test_no_constraints_is_feasible(self):
        assert is_feasible([], [])
        assert is_feasible()

    def test_boundary_within_tolerance_is_feasible(self):
        assert is_feasible([1e-9], [1e-9], tol=1e-6)


class TestFeasibilityCompare:
    def test_feasible_beats_infeasible_regardless_of_objective(self):
        feasible_but_worse = (100.0, [-1.0], [])
        infeasible_but_better_objective = (-100.0, [5.0], [])
        assert feasibility_compare(feasible_but_worse, infeasible_but_better_objective) == -1
        assert feasibility_compare(infeasible_but_better_objective, feasible_but_worse) == 1

    def test_between_two_feasible_lower_objective_wins(self):
        a = (1.0, [-1.0], [])
        b = (2.0, [-1.0], [])
        assert feasibility_compare(a, b) == -1
        assert feasibility_compare(b, a) == 1

    def test_between_two_infeasible_smaller_violation_wins(self):
        a = (0.0, [1.0], [])  # violation 1.0
        b = (0.0, [5.0], [])  # violation 5.0
        assert feasibility_compare(a, b) == -1
        assert feasibility_compare(b, a) == 1

    def test_exact_tie_returns_zero(self):
        a = (3.0, [-1.0], [])
        b = (3.0, [-2.0], [])
        assert feasibility_compare(a, b) == 0

    def test_exact_tie_both_infeasible_same_violation(self):
        a = (0.0, [2.0], [])
        b = (0.0, [2.0], [])
        assert feasibility_compare(a, b) == 0


class TestAgainstRealSuiteResults:
    """Sanity-check the rule against actual g_suite/engineering evaluations,
    not just hand-built tuples."""

    def test_g01_known_optimum_is_feasible(self):
        x_star = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 1], dtype=float)
        result = g_suite.g01(x_star)
        assert is_feasible(result[1], result[2])

    def test_known_optimum_beats_an_infeasible_point(self):
        x_star = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 1], dtype=float)
        optimum = g_suite.g01(x_star)
        infeasible = g_suite.g01(np.full(13, 10.0))
        assert not is_feasible(infeasible[1], infeasible[2])
        assert feasibility_compare(optimum, infeasible) == -1

    def test_spring_design_known_optimum_is_feasible(self):
        # Matches the tolerance test_engineering.py's own feasibility check
        # for this same documented optimum uses -- it's a literature value,
        # not an exact algebraic root, so it lands fractionally outside the
        # default 1e-6 tolerance.
        x_star = np.array([0.051690, 0.356750, 11.287126])
        result = engineering.spring_design(x_star)
        assert is_feasible(result[1], result[2], tol=1e-3)
