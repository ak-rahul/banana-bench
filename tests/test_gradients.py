"""
Tests for bananabench.gradients: finite-difference gradient estimation
(approximate_gradient) and the GradientWrapper convenience class.
"""

import numpy as np
import pytest

from bananabench import gradients, sphere
from bananabench.gradients import GradientWrapper, approximate_gradient


class TestApproximateGradient:
    def test_matches_analytic_sphere_gradient(self):
        x = np.array([1.0, -2.0, 3.0])
        grad = approximate_gradient(sphere, x, method="central")
        np.testing.assert_allclose(grad, 2 * x, atol=1e-5)

    def test_forward_and_backward_agree_with_central(self):
        x = np.array([0.5, 0.5])
        central = approximate_gradient(sphere, x, method="central")
        forward = approximate_gradient(sphere, x, method="forward")
        backward = approximate_gradient(sphere, x, method="backward")
        np.testing.assert_allclose(central, forward, atol=1e-4)
        np.testing.assert_allclose(central, backward, atol=1e-4)

    def test_zero_gradient_at_the_minimum(self):
        grad = approximate_gradient(sphere, np.zeros(4), method="central")
        np.testing.assert_allclose(grad, np.zeros(4), atol=1e-6)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            approximate_gradient(sphere, np.zeros(2), method="unknown")

    def test_does_not_mutate_input(self):
        x = np.array([1.0, 2.0])
        x_copy = x.copy()
        approximate_gradient(sphere, x, method="central")
        assert np.array_equal(x, x_copy)


class TestGradientWrapper:
    def test_call_returns_function_value_and_counts_evaluations(self):
        wrapper = GradientWrapper(sphere)
        x = np.array([1.0, 2.0])
        assert np.isclose(wrapper(x), sphere(x))
        assert wrapper.evaluations == 1
        wrapper(x)
        assert wrapper.evaluations == 2

    def test_gradient_matches_analytic_and_counts_evaluations(self):
        wrapper = GradientWrapper(sphere)
        x = np.array([1.0, 2.0])
        grad = wrapper.gradient(x)
        np.testing.assert_allclose(grad, 2 * x, atol=1e-5)
        assert wrapper.grad_evaluations == 1

    def test_fun_and_grad_matches_separate_calls(self):
        wrapper = GradientWrapper(sphere)
        x = np.array([1.0, 2.0])
        fx, fgrad = wrapper.fun_and_grad(x)
        assert np.isclose(fx, sphere(x))
        np.testing.assert_allclose(fgrad, 2 * x, atol=1e-5)


def test_gradients_module_exposed_on_package():
    assert gradients.approximate_gradient is approximate_gradient
    assert gradients.GradientWrapper is GradientWrapper
