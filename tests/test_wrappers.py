"""
Tests for bananabench.wrappers: BenchmarkWrapper, NoisyFunction,
ShiftedFunction, RotatedFunction.
"""

import numpy as np
import pytest

from bananabench import NoisyFunction, RotatedFunction, ShiftedFunction, sphere
from bananabench.wrappers import BenchmarkWrapper


class TestBenchmarkWrapper:
    def test_call_delegates_to_wrapped_function(self):
        wrapper = BenchmarkWrapper(sphere)
        x = np.array([1.0, 2.0, 3.0])
        assert wrapper(x) == sphere(x)

    def test_preserves_name_and_docstring(self):
        wrapper = BenchmarkWrapper(sphere)
        assert wrapper.__name__ == sphere.__name__
        assert wrapper.__doc__ == sphere.__doc__


class TestNoisyFunction:
    def test_noise_makes_repeated_calls_differ(self):
        noisy_sphere = NoisyFunction(sphere, scale=1.0)
        x = np.zeros(5)
        assert noisy_sphere(x) != noisy_sphere(x)

    def test_seeded_noise_is_reproducible(self):
        x = np.zeros(5)
        noisy1 = NoisyFunction(sphere, scale=1.0, seed=42)
        noisy2 = NoisyFunction(sphere, scale=1.0, seed=42)
        assert noisy1(x) == noisy2(x)

    def test_uniform_noise_type(self):
        noisy_sphere = NoisyFunction(sphere, noise_type="uniform", scale=0.5, seed=0)
        x = np.zeros(3)
        value = noisy_sphere(x)
        assert abs(value - sphere(x)) <= 0.5

    def test_unknown_noise_type_raises(self):
        noisy_sphere = NoisyFunction(sphere, noise_type="not_a_real_type", seed=0)
        with pytest.raises(ValueError):
            noisy_sphere(np.zeros(3))


class TestShiftedFunction:
    def test_optimum_moves_to_shift_vector(self):
        shift = np.array([1.0, 1.0])
        shifted_sphere = ShiftedFunction(sphere, shift=shift)
        assert np.isclose(shifted_sphere(shift), 0.0)

    def test_value_at_original_optimum(self):
        shift = np.array([1.0, 1.0])
        shifted_sphere = ShiftedFunction(sphere, shift=shift)
        # sphere(0 - shift) = sphere(-1, -1) = 1 + 1 = 2
        assert np.isclose(shifted_sphere(np.zeros(2)), 2.0)


class TestRotatedFunction:
    def test_rotation_matches_manually_rotated_input(self):
        matrix = np.array([[0, -1], [1, 0]])  # 90-degree rotation
        rotated_sphere = RotatedFunction(sphere, matrix=matrix)
        x = np.array([1.0, 0.0])
        assert np.isclose(rotated_sphere(x), sphere(matrix @ x))

    def test_identity_matrix_is_a_no_op(self):
        identity = np.eye(3)
        rotated_sphere = RotatedFunction(sphere, matrix=identity)
        x = np.array([1.0, -2.0, 3.0])
        assert np.isclose(rotated_sphere(x), sphere(x))
