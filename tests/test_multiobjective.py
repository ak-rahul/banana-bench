"""
Tests for bananabench.multiobjective: the ZDT suite (zdt1-zdt4, zdt6), the DTLZ
suite (dtlz1-dtlz7), and the Pareto-front / quality-indicator utilities.
"""

import numpy as np
import pytest

from bananabench import multiobjective as mo

FUNCTION_NAMES = list(mo.ZDT_SUITE.keys())


class TestRegistry:
    def test_zdt5_deliberately_excluded(self):
        assert "zdt5" not in mo.ZDT_SUITE

    def test_get_mo_function_list(self):
        assert set(mo.get_mo_function_list()) == set(FUNCTION_NAMES) | set(mo.DTLZ_SUITE)

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

    def test_single_point_is_its_box_volume(self):
        for m in (2, 3, 4, 5):
            assert mo.hypervolume(np.full((1, m), 0.5), np.ones(m)) == pytest.approx(0.5**m)

    def test_larger_front_has_larger_hypervolume(self):
        ref = np.array([1.1, 1.1])
        small = np.array([[0.5, 0.5]])
        large = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        assert mo.hypervolume(large, ref) > mo.hypervolume(small, ref)


# Reference values from pymoo 0.6.2 (anyoptimization/pymoo, problems/many/dtlz.py), an
# implementation independent of this one, at x = linspace(0.05, 0.95, n) with M = 3 and
# the paper's default k. Every function also matched pymoo to <= 2.4e-15 relative error
# at 200 random points for M = 2, 3, 4 when this suite was added; Platypus's DTLZ1-4/7
# formulas agree as well.
_PYMOO_REFERENCE = {
    "dtlz1": [3.1737500000000005, 12.695, 301.50624999999997],
    "dtlz2": [1.579743014590091, 0.33185707550796356, 0.12704213496262826],
    "dtlz3": [996.4816421137755, 209.33118899408927, 80.13655011986134],
    "dtlz4": [1.6192148760330576, 2.529042163412256e-88, 2.006433617517688e-130],
    "dtlz5": [1.36401821590179, 0.8632331188033654, 0.12704213496262826],
    "dtlz6": [10.003763694347276, 2.690329500867642, 0.8152872388471067],
    "dtlz7": [0.05, 0.09285714285714286, 20.420304891979775],
}
DTLZ_NAMES = list(mo.DTLZ_SUITE)


class TestDTLZFunctions:
    @pytest.mark.parametrize("name", DTLZ_NAMES)
    def test_matches_independent_implementation(self, name):
        n = mo.get_mo_default_dim(name, 3)
        x = np.linspace(0.05, 0.95, n)
        np.testing.assert_allclose(
            mo.DTLZ_SUITE[name]["function"](x, n_objectives=3),
            _PYMOO_REFERENCE[name],
            rtol=1e-12,
            atol=1e-300,
        )

    @pytest.mark.parametrize("name", DTLZ_NAMES)
    @pytest.mark.parametrize("m", [2, 3, 5])
    def test_pareto_optimal_points_lie_on_the_front(self, name, m):
        info = mo.DTLZ_SUITE[name]
        n = mo.get_mo_default_dim(name, m)
        rng = np.random.default_rng(0)
        X = rng.random((25, n))
        X[:, m - 1 :] = info["optimal_distance_value"]
        F = np.array([mo.get_mo_function(name, m)(x) for x in X])
        assert F.shape == (25, m)
        if name == "dtlz1":
            np.testing.assert_allclose(F.sum(axis=1), 0.5, atol=1e-12)
        elif name == "dtlz7":
            expected = 2 * m - np.sum(F[:, :-1] * (1 + np.sin(3 * np.pi * F[:, :-1])), axis=1)
            np.testing.assert_allclose(F[:, -1], expected, atol=1e-12)
        else:
            np.testing.assert_allclose((F**2).sum(axis=1), 1.0, atol=1e-12)

    @pytest.mark.parametrize("name", DTLZ_NAMES)
    def test_rejects_too_few_variables(self, name):
        with pytest.raises(ValueError):
            mo.DTLZ_SUITE[name]["function"](np.full(2, 0.5), n_objectives=3)

    def test_rejects_single_objective(self):
        with pytest.raises(ValueError):
            mo.dtlz2(np.full(5, 0.5), n_objectives=1)

    @pytest.mark.parametrize("name,k", [("dtlz1", 5), ("dtlz2", 10), ("dtlz6", 10), ("dtlz7", 20)])
    def test_default_dimension_follows_the_paper(self, name, k):
        assert mo.DTLZ_SUITE[name]["k"] == k
        assert mo.get_mo_default_dim(name) == 3 + k - 1
        assert mo.get_mo_default_dim(name, 5) == 5 + k - 1

    def test_n_objectives_validation(self):
        assert mo.get_mo_n_objectives("dtlz2") == 3
        assert mo.get_mo_n_objectives("dtlz2", 6) == 6
        assert mo.get_mo_n_objectives("zdt1") == 2
        with pytest.raises(ValueError):
            mo.get_mo_n_objectives("zdt1", 3)


class TestDasDennis:
    @pytest.mark.parametrize("m,h", [(2, 4), (3, 12), (5, 3)])
    def test_count_and_simplex(self, m, h):
        import math

        w = mo.das_dennis(m, h)
        assert w.shape == (math.comb(h + m - 1, m - 1), m)
        np.testing.assert_allclose(w.sum(axis=1), 1.0)
        assert np.all(w >= 0)
        np.testing.assert_allclose(w * h, np.round(w * h))
        assert len(np.unique(w, axis=0)) == len(w)

    def test_two_objectives(self):
        np.testing.assert_allclose(mo.das_dennis(2, 2), [[0, 1], [0.5, 0.5], [1, 0]])


class TestDTLZParetoFront:
    @pytest.mark.parametrize("name", DTLZ_NAMES)
    def test_front_is_non_dominated(self, name):
        front = mo.get_pareto_front(name, n_points=60)
        assert front.shape[1] == 3
        assert len(mo.non_dominated_front(front)) == len(front)

    def test_dtlz1_front_is_the_half_simplex(self):
        np.testing.assert_allclose(mo.get_pareto_front("dtlz1").sum(axis=1), 0.5)

    @pytest.mark.parametrize("name", ["dtlz2", "dtlz3", "dtlz4", "dtlz5", "dtlz6"])
    def test_spherical_fronts(self, name):
        front = mo.get_pareto_front(name, n_objectives=4)
        np.testing.assert_allclose(np.linalg.norm(front, axis=1), 1.0)

    def test_dtlz5_front_is_a_curve(self):
        # At g = 0, theta_2 = theta_3 = pi/4, so f1 == f2 for M = 4.
        front = mo.get_pareto_front("dtlz5", n_points=50, n_objectives=4)
        np.testing.assert_allclose(front[:, 0], front[:, 1])

    def test_dtlz7_front_is_disconnected(self):
        # 2^(M-1) = 4 regions: f1 splits into separate intervals with a wide gap.
        front = mo.get_pareto_front("dtlz7", n_points=400)
        assert np.max(np.diff(np.sort(front[:, 0]))) > 0.2


class TestHypervolumeManyObjectives:
    def test_exact_matches_independent_implementation(self):
        # Values from pymoo 0.6.2's independent HV implementation on the same random fronts.
        rng = np.random.default_rng(7)
        expected = {
            (2, 50): 1.1508653170683796,
            (3, 60): 1.1007641050647117,
            (3, 150): 1.1341773790931484,
            (4, 40): 1.0800631845122428,
            (5, 25): 0.6313341166384926,
        }
        for (m, n), value in expected.items():
            points = rng.random((n, m))
            result = mo.hypervolume(points, np.full(m, 1.1), method="exact")
            assert result == pytest.approx(value, rel=1e-12)

    def test_dominated_and_duplicate_points_do_not_change_it(self):
        points = np.array([[0.2, 0.5, 0.4], [0.5, 0.2, 0.3], [0.4, 0.4, 0.1]])
        ref = np.ones(3)
        base = mo.hypervolume(points, ref)
        noisy = np.vstack([points, points[0], [0.9, 0.9, 0.9], [0.6, 0.6, 0.5]])
        assert mo.hypervolume(noisy, ref) == pytest.approx(base)

    def test_sphere_front_converges_to_analytic_value(self):
        # HV of the positive-orthant unit sphere w.r.t. ref 1.1^3 is 1.331 - pi/6.
        limit = 1.1**3 - np.pi / 6
        coarse = mo.hypervolume(mo.get_pareto_front("dtlz2", n_points=100), np.full(3, 1.1))
        fine = mo.hypervolume(mo.get_pareto_front("dtlz2", n_points=1000), np.full(3, 1.1))
        assert coarse < fine < limit
        assert limit - fine < 0.03

    def test_monte_carlo_agrees_with_exact(self):
        rng = np.random.default_rng(3)
        points = rng.random((40, 4))
        ref = np.full(4, 1.1)
        exact = mo.hypervolume(points, ref, method="exact")
        estimate = mo.hypervolume(points, ref, method="monte_carlo", n_samples=200_000, seed=0)
        assert estimate == pytest.approx(exact, rel=0.01)

    def test_monte_carlo_is_reproducible_with_seed(self):
        points = np.random.default_rng(1).random((30, 4))
        ref = np.full(4, 1.1)
        first = mo.hypervolume(points, ref, method="monte_carlo", seed=5)
        assert first == mo.hypervolume(points, ref, method="monte_carlo", seed=5)

    def test_rejects_bad_input(self):
        with pytest.raises(ValueError):
            mo.hypervolume(np.zeros((2, 3)), np.ones(2))
        with pytest.raises(ValueError):
            mo.hypervolume(np.zeros((2, 2)), np.ones(2), method="magic")
