"""
Tests for bananabench.multiobjective: the ZDT suite (zdt1-zdt4, zdt6) and
its supporting Pareto-front / quality-indicator utilities.
"""

import numpy as np
import pytest

from bananabench import multiobjective as mo

FUNCTION_NAMES = list(mo.ZDT_SUITE.keys())


class TestRegistry:
    def test_zdt5_deliberately_excluded(self):
        assert "zdt5" not in mo.ZDT_SUITE

    def test_get_mo_function_list(self):
        assert set(mo.get_mo_function_list()) == set(FUNCTION_NAMES)

    @pytest.mark.parametrize("name", FUNCTION_NAMES)
    def test_registry_entry_shape(self, name):
        info = mo.ZDT_SUITE[name]
        assert callable(info["function"])
        assert info["n_objectives"] == 2
        assert info["default_dim"] >= 2
        assert callable(info["bounds"])
        assert isinstance(info["properties"], list) and "multi-objective" in info["properties"]

    @pytest.mark.parametrize("name", FUNCTION_NAMES)
    def test_get_mo_bounds_matches_default_dim(self, name):
        info = mo.ZDT_SUITE[name]
        bounds = mo.get_mo_bounds(name)
        assert len(bounds) == info["default_dim"]
        assert all(lo < hi for lo, hi in bounds)

    def test_zdt4_bounds_are_per_dimension(self):
        bounds = mo.get_mo_bounds("zdt4", dim=5)
        assert bounds[0] == (0.0, 1.0)
        assert bounds[1:] == [(-5.0, 5.0)] * 4


class TestZDTFunctions:
    @pytest.mark.parametrize("name", FUNCTION_NAMES)
    def test_dimension_validation(self, name):
        func = mo.ZDT_SUITE[name]["function"]
        with pytest.raises(ValueError):
            func(np.ones(1))

    @pytest.mark.parametrize("name", FUNCTION_NAMES)
    def test_return_shape(self, name):
        info = mo.ZDT_SUITE[name]
        func = info["function"]
        dim = info["default_dim"]
        x = np.zeros(dim)
        result = func(x)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)

    def test_zdt1_at_origin(self):
        f1, f2 = mo.zdt1(np.zeros(30))
        assert f1 == pytest.approx(0.0)
        assert f2 == pytest.approx(1.0)

    def test_zdt1_at_minimal_g_manifold(self):
        # x1 = 1, x[1:] = 0 -> g = 1, f2 = 1 - sqrt(1/1) = 0
        x = np.zeros(30)
        x[0] = 1.0
        f1, f2 = mo.zdt1(x)
        assert f1 == pytest.approx(1.0)
        assert f2 == pytest.approx(0.0, abs=1e-9)

    def test_zdt2_non_convex_front_shape(self):
        x = np.zeros(30)
        x[0] = 0.5
        f1, f2 = mo.zdt2(x)
        assert f2 == pytest.approx(1 - f1**2)

    def test_zdt4_minimal_g_within_bounds(self):
        # g's minimizer x[1:] = 0 lies inside zdt4's [-5, 5] bounds, so g = 1 there.
        x = np.zeros(10)
        x[0] = 0.5
        f1, f2 = mo.zdt4(x)
        assert f2 == pytest.approx(1 - np.sqrt(0.5))

    def test_zdt6_f1_not_identity(self):
        # Unlike zdt1/2/3/4, zdt6's f1 is not simply x[0].
        x = np.zeros(10)
        x[0] = 0.5
        f1, _ = mo.zdt6(x)
        assert f1 != pytest.approx(0.5)


class TestNonDominatedFront:
    def test_filters_dominated_points(self):
        points = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        front = mo.non_dominated_front(points)
        assert len(front) == 3
        assert not any(np.allclose(p, [1.0, 1.0]) for p in front)

    def test_all_points_nondominated(self):
        points = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        front = mo.non_dominated_front(points)
        assert len(front) == 3


class TestGetParetoFront:
    @pytest.mark.parametrize("name", FUNCTION_NAMES)
    def test_front_is_nondominated_and_sorted(self, name):
        front = mo.get_pareto_front(name, n_points=100)
        assert front.ndim == 2 and front.shape[1] == 2
        assert len(front) > 1
        assert np.all(np.diff(front[:, 0]) >= 0)
        assert np.array_equal(front, mo.non_dominated_front(front))

    def test_zdt1_front_matches_analytic_formula(self):
        front = mo.get_pareto_front("zdt1", n_points=50)
        assert np.allclose(front[:, 1], 1 - np.sqrt(front[:, 0]), atol=1e-6)

    def test_zdt3_front_is_disconnected(self):
        front = mo.get_pareto_front("zdt3", n_points=300)
        # A disconnected front has at least one gap between consecutive f1 samples
        # noticeably larger than the uniform sampling step.
        step = 1.0 / 300
        assert np.any(np.diff(front[:, 0]) > 5 * step)

    def test_zdt6_front_does_not_start_at_zero(self):
        # zdt6's f1 isn't x1 itself and isn't monotonic on [0, 1], so the true
        # front's minimum f1 is strictly greater than 0 (~0.2808 in the literature).
        front = mo.get_pareto_front("zdt6", n_points=300)
        assert front[0, 0] > 0.2


class TestIGD:
    def test_zero_when_front_equals_reference(self):
        front = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        assert mo.igd(front, front) == pytest.approx(0.0)

    def test_positive_when_fronts_differ(self):
        reference = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        front = np.array([[0.0, 2.0]])
        assert mo.igd(front, reference) > 0.0

    def test_empty_front_is_infinite(self):
        reference = np.array([[0.0, 1.0]])
        assert mo.igd(np.empty((0, 2)), reference) == float("inf")


class TestHypervolume:
    def test_known_value(self):
        front = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        ref = np.array([1.1, 1.1])
        assert mo.hypervolume(front, ref) == pytest.approx(0.46, abs=1e-9)

    def test_zero_when_nothing_dominates_reference(self):
        front = np.array([[2.0, 2.0]])
        ref = np.array([1.1, 1.1])
        assert mo.hypervolume(front, ref) == 0.0

    def test_raises_for_more_than_two_objectives(self):
        front = np.array([[0.0, 0.0, 0.0]])
        ref = np.array([1.0, 1.0, 1.0])
        with pytest.raises(NotImplementedError):
            mo.hypervolume(front, ref)

    def test_larger_front_has_larger_hypervolume(self):
        ref = np.array([1.1, 1.1])
        small = np.array([[0.5, 0.5]])
        large = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        assert mo.hypervolume(large, ref) > mo.hypervolume(small, ref)
