"""
Tests for bananabench.g_suite: the constrained G-suite (g04, g06, g08, g11).

Each function returns (objective, inequality_violations, equality_violations)
rather than a single float; correctness is checked against the documented
optimal objective values and feasibility of the documented optimal points.
"""

import numpy as np
import pytest

from bananabench import g_suite


class TestG04:
    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            g_suite.g04(np.ones(4))

    def test_return_shape(self):
        f, g, h = g_suite.g04(np.ones(5))
        assert isinstance(f, float)
        assert isinstance(g, np.ndarray) and g.shape == (6,)
        assert isinstance(h, np.ndarray) and h.shape == (0,)

    def test_known_optimum_objective_and_feasibility(self):
        x = np.array([78.0, 33.0, 29.995, 45.0, 36.776])
        f, g, h = g_suite.g04(x)
        assert f == pytest.approx(-30665.539, abs=1.0)
        assert np.all(g <= 1e-3)


class TestG06:
    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            g_suite.g06(np.ones(3))

    def test_return_shape(self):
        f, g, h = g_suite.g06(np.ones(2))
        assert isinstance(f, float)
        assert g.shape == (2,)
        assert h.shape == (0,)

    def test_known_optimum_objective_and_feasibility(self):
        x = np.array([14.095, 0.84296])
        f, g, h = g_suite.g06(x)
        assert f == pytest.approx(-6961.81388, abs=1.0)
        assert np.all(g <= 1e-2)


class TestG08:
    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            g_suite.g08(np.ones(3))

    def test_known_optimum_objective_and_feasibility(self):
        x = np.array([1.2279713, 4.2453733])
        f, g, h = g_suite.g08(x)
        assert f == pytest.approx(-0.095825, abs=1e-3)
        assert np.all(g <= 1e-6)


class TestG11:
    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            g_suite.g11(np.ones(3))

    def test_known_optimum_objective_and_feasibility(self):
        x = np.array([-0.70711, 0.5])
        f, g, h = g_suite.g11(x)
        assert f == pytest.approx(0.7499, abs=1e-3)
        assert np.all(np.abs(h) <= 1e-4)


@pytest.mark.parametrize(
    "func,dim",
    [(g_suite.g04, 5), (g_suite.g06, 2), (g_suite.g08, 2), (g_suite.g11, 2)],
)
def test_return_types_are_consistent_across_functions(func, dim):
    f, g, h = func(np.ones(dim))
    assert isinstance(f, float)
    assert isinstance(g, np.ndarray)
    assert isinstance(h, np.ndarray)


def test_g_suite_module_exposed_on_package():
    from bananabench import g_suite as pkg_g_suite

    assert pkg_g_suite.g04 is g_suite.g04
