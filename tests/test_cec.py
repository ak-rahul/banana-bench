"""
Tests for bananabench.cec: CECFunction and CECCompositionFunction, the
shift/rotate/bias/compose framework used for CEC 2017/2020-style suites.
"""

import numpy as np
import pytest

from bananabench import cec, sphere
from bananabench.cec import CECCompositionFunction, CECFunction


class TestCECFunction:
    def test_shift_and_bias(self):
        shift = np.array([1.0, 2.0])
        f = CECFunction(sphere, shift_vector=shift, bias=5.0)
        assert np.isclose(f(shift), 5.0)
        assert np.isclose(f(np.zeros(2)), sphere(-shift) + 5.0)

    def test_rotation(self):
        rotation = np.array([[0, -1], [1, 0]])
        f = CECFunction(sphere, rotation_matrix=rotation)
        x = np.array([3.0, 4.0])
        assert np.isclose(f(x), sphere(rotation @ x))

    def test_shift_and_rotation_applied_in_order(self):
        shift = np.array([1.0, 0.0])
        rotation = np.array([[0, -1], [1, 0]])
        f = CECFunction(sphere, shift_vector=shift, rotation_matrix=rotation)
        x = np.array([2.0, 3.0])
        expected = sphere(rotation @ (x - shift))
        assert np.isclose(f(x), expected)

    def test_no_shift_or_rotation_is_plain_bias(self):
        f = CECFunction(sphere, bias=10.0)
        x = np.array([1.0, 1.0])
        assert np.isclose(f(x), sphere(x) + 10.0)

    def test_does_not_mutate_input(self):
        shift = np.array([1.0, 2.0])
        f = CECFunction(sphere, shift_vector=shift)
        x = np.array([5.0, 5.0])
        x_copy = x.copy()
        f(x)
        assert np.array_equal(x, x_copy)


class TestCECCompositionFunction:
    def test_finite_result(self):
        f1 = CECFunction(sphere, shift_vector=np.zeros(2))
        f2 = CECFunction(sphere, shift_vector=np.array([100.0, 100.0]))
        comp = CECCompositionFunction([f1, f2], sigma=np.array([1.0, 1.0]), biases=np.zeros(2))
        assert np.isfinite(comp(np.array([1e-6, 0.0])))

    def test_exact_match_to_one_components_optimum(self):
        f1 = CECFunction(sphere, shift_vector=np.zeros(2))
        f2 = CECFunction(sphere, shift_vector=np.array([100.0, 100.0]))
        comp = CECCompositionFunction([f1, f2], sigma=np.array([1.0, 1.0]), biases=np.zeros(2))
        # Exactly at f1's optimum: dist_sq == 0 gives f1 an overwhelming weight.
        result = comp(np.zeros(2))
        assert np.isclose(result, f1(np.zeros(2)), atol=1e-6)

    def test_biases_shift_composed_result(self):
        f1 = CECFunction(sphere, shift_vector=np.zeros(2))
        comp_no_bias = CECCompositionFunction([f1], sigma=np.array([1.0]), biases=np.array([0.0]))
        comp_with_bias = CECCompositionFunction(
            [f1], sigma=np.array([1.0]), biases=np.array([50.0])
        )
        x = np.array([1.0, 1.0])
        assert np.isclose(comp_with_bias(x) - comp_no_bias(x), 50.0)


def test_cec_module_exposed_on_package():
    assert cec.CECFunction is CECFunction
    assert cec.CECCompositionFunction is CECCompositionFunction
