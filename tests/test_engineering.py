"""
Tests for bananabench.engineering: the classic engineering design suite
(spring design, pressure vessel, welded beam, gear train).

Each function returns (objective, inequality_violations, equality_violations)
rather than a single float, following g_suite.py's convention; correctness is
checked against the documented optimal objective values and feasibility of
the documented optimal points -- see engineering.py's docstrings for sources.
"""

import numpy as np
import pytest

from bananabench import engineering


class TestSpringDesign:
    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            engineering.spring_design(np.ones(2))

    def test_return_shape(self):
        f, g, h = engineering.spring_design(np.array([0.05, 0.25, 2.0]))
        assert isinstance(f, float)
        assert isinstance(g, np.ndarray) and g.shape == (4,)
        assert isinstance(h, np.ndarray) and h.shape == (0,)

    def test_known_optimum_objective_and_feasibility(self):
        x = np.array([0.051690, 0.356750, 11.287126])
        f, g, h = engineering.spring_design(x)
        assert f == pytest.approx(0.012665, abs=1e-4)
        assert np.all(g <= 1e-3)


class TestPressureVessel:
    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            engineering.pressure_vessel(np.ones(3))

    def test_return_shape(self):
        f, g, h = engineering.pressure_vessel(np.array([1.0, 1.0, 20.0, 50.0]))
        assert isinstance(f, float)
        assert isinstance(g, np.ndarray) and g.shape == (4,)
        assert isinstance(h, np.ndarray) and h.shape == (0,)

    def test_known_optimum_objective_and_feasibility(self):
        x = np.array([0.8125, 0.4375, 42.0984455958549, 176.6365958424394])
        f, g, h = engineering.pressure_vessel(x)
        assert f == pytest.approx(6059.714335048436, abs=1e-4)
        assert np.all(g <= 1e-6)


class TestWeldedBeam:
    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            engineering.welded_beam(np.ones(3))

    def test_return_shape(self):
        f, g, h = engineering.welded_beam(np.array([0.5, 5.0, 5.0, 0.5]))
        assert isinstance(f, float)
        assert isinstance(g, np.ndarray) and g.shape == (7,)
        assert isinstance(h, np.ndarray) and h.shape == (0,)

    def test_known_optimum_objective_and_feasibility(self):
        x = np.array([0.205729639786079, 3.470488665627977, 9.036623910357633, 0.205729639786079])
        f, g, h = engineering.welded_beam(x)
        assert f == pytest.approx(1.724852308597361, abs=1e-6)
        # The literature point is an optimizer's converged solution, not an
        # exact algebraic root, so allow a small tolerance rather than
        # requiring g <= 0 exactly.
        assert np.all(g <= 0.05)


class TestGearTrain:
    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            engineering.gear_train(np.ones(3))

    def test_return_shape(self):
        f, g, h = engineering.gear_train(np.array([20.0, 20.0, 20.0, 20.0]))
        assert isinstance(f, float)
        assert isinstance(g, np.ndarray) and g.shape == (0,)
        assert isinstance(h, np.ndarray) and h.shape == (0,)

    def test_known_optimum_objective(self):
        x = np.array([19.0, 16.0, 43.0, 49.0])
        f, g, h = engineering.gear_train(x)
        assert f == pytest.approx(2.7e-12, abs=1e-13)


@pytest.mark.parametrize(
    "func,x",
    [
        (engineering.spring_design, np.array([1.0, 2.0, 3.0])),
        (engineering.pressure_vessel, np.ones(4)),
        (engineering.welded_beam, np.ones(4)),
        (engineering.gear_train, np.ones(4)),
    ],
)
def test_return_types_are_consistent_across_functions(func, x):
    f, g, h = func(x)
    assert isinstance(f, float)
    assert isinstance(g, np.ndarray)
    assert isinstance(h, np.ndarray)


def test_engineering_module_exposed_on_package():
    from bananabench import engineering as pkg_engineering

    assert pkg_engineering.spring_design is engineering.spring_design


class TestEngineeringSuiteRegistry:
    def test_covers_all_four_problems(self):
        assert set(engineering.ENGINEERING_SUITE.keys()) == {
            "spring_design",
            "pressure_vessel",
            "welded_beam",
            "gear_train",
        }

    def test_get_engineering_function_list_matches_registry(self):
        assert set(engineering.get_engineering_function_list()) == set(
            engineering.ENGINEERING_SUITE.keys()
        )

    @pytest.mark.parametrize("name", list(engineering.ENGINEERING_SUITE.keys()))
    def test_entry_function_matches_module_function(self, name):
        assert engineering.ENGINEERING_SUITE[name]["function"] is getattr(engineering, name)

    @pytest.mark.parametrize(
        "name,x",
        [
            ("spring_design", np.array([1.0, 2.0, 3.0])),
            ("pressure_vessel", np.ones(4)),
            ("welded_beam", np.ones(4)),
            ("gear_train", np.ones(4)),
        ],
    )
    def test_constraint_counts_match_actual_returned_arrays(self, name, x):
        # Cross-check the registry's n_inequality/n_equality against a live
        # call, not just the transcribed docstring, so the two can't
        # silently drift (same discipline as g_suite.G_SUITE's own test).
        entry = engineering.ENGINEERING_SUITE[name]
        f, g, h = entry["function"](x)
        assert len(g) == entry["n_inequality"]
        assert len(h) == entry["n_equality"]
