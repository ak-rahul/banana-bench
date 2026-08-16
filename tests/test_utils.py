"""
Tests for bananabench.utils: bounds normalization, point generation,
coordinate transforms, and distance-to-optimum helpers.
"""

import numpy as np
import pytest

from bananabench.utils import (
    calculate_distance_to_optimum,
    check_bounds,
    clip_to_bounds,
    generate_grid_points,
    generate_random_point,
    get_bounds_center,
    get_bounds_range,
    normalize_bounds,
    scale_from_unit,
    scale_to_unit,
)


class TestNormalizeBounds:
    def test_single_bound_in_list_replicated(self):
        bounds = normalize_bounds([(-5, 5)], 3)
        assert bounds == [(-5, 5), (-5, 5), (-5, 5)]

    def test_already_normalized_passes_through(self):
        bounds = normalize_bounds([(-5, 5), (-10, 10)], 2)
        assert bounds == [(-5, 5), (-10, 10)]

    def test_plain_tuple_replicated(self):
        bounds = normalize_bounds((-5, 5), 4)
        assert bounds == [(-5, 5)] * 4

    def test_unnormalizable_input_raises(self):
        with pytest.raises(ValueError):
            normalize_bounds([], 3)


class TestGenerateRandomPoint:
    @pytest.mark.parametrize("method", ["uniform", "normal", "center_biased"])
    def test_point_within_bounds(self, method):
        bounds = [(-5, 5), (-10, 10)]
        point = generate_random_point(bounds, method=method)
        assert len(point) == 2
        assert -5 <= point[0] <= 5
        assert -10 <= point[1] <= 10

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            generate_random_point([(-5, 5)], method="not_a_real_method")


class TestCheckBounds:
    def test_point_inside_bounds(self):
        assert check_bounds(np.array([0, 0]), [(-5, 5), (-5, 5)])

    def test_point_outside_bounds(self):
        assert not check_bounds(np.array([10, 0]), [(-5, 5), (-5, 5)])

    def test_point_on_boundary_within_tolerance(self):
        assert check_bounds(np.array([5.0, -5.0]), [(-5, 5), (-5, 5)])

    def test_point_just_outside_default_tolerance(self):
        assert not check_bounds(np.array([5.1, 0.0]), [(-5, 5), (-5, 5)])


class TestScaling:
    def test_scale_to_unit_center(self):
        unit_point = scale_to_unit(np.array([0, 0]), [(-10, 10), (-5, 5)])
        assert np.allclose(unit_point, [0.5, 0.5])

    def test_scale_from_unit_center(self):
        point = scale_from_unit(np.array([0.5, 0.5]), [(-10, 10), (-5, 5)])
        assert np.allclose(point, [0, 0])

    def test_round_trip_is_identity(self):
        bounds = [(-10, 10), (-5, 5), (0, 100)]
        original = np.array([3.0, -2.0, 42.0])
        round_tripped = scale_from_unit(scale_to_unit(original, bounds), bounds)
        assert np.allclose(round_tripped, original)


class TestClipToBounds:
    def test_clips_out_of_range_values(self):
        clipped = clip_to_bounds(np.array([10, -10]), [(-5, 5), (-5, 5)])
        assert np.allclose(clipped, [5, -5])

    def test_leaves_in_range_values_unchanged(self):
        clipped = clip_to_bounds(np.array([1, -1]), [(-5, 5), (-5, 5)])
        assert np.allclose(clipped, [1, -1])


class TestBoundsGeometry:
    def test_get_bounds_range(self):
        assert np.allclose(get_bounds_range([(-5, 5), (-10, 10)]), [10, 20])

    def test_get_bounds_center(self):
        assert np.allclose(get_bounds_center([(-5, 5), (-10, 10)]), [0, 0])

    def test_get_bounds_center_asymmetric(self):
        assert np.allclose(get_bounds_center([(0, 10)]), [5])


class TestGenerateGridPoints:
    def test_grid_shape(self):
        grid = generate_grid_points([(-1, 1), (-1, 1)], points_per_dim=3)
        assert grid.shape == (9, 2)

    def test_grid_covers_bounds_extremes(self):
        grid = generate_grid_points([(-1, 1)], points_per_dim=3)
        assert np.isclose(grid.min(), -1)
        assert np.isclose(grid.max(), 1)

    def test_one_dimensional_grid(self):
        grid = generate_grid_points([(0, 10)], points_per_dim=5)
        assert grid.shape == (5, 1)


class TestCalculateDistanceToOptimum:
    def test_single_optimum(self):
        distance = calculate_distance_to_optimum(np.array([1, 1]), np.array([0, 0]))
        assert np.isclose(distance, np.sqrt(2))

    def test_zero_distance_at_optimum(self):
        distance = calculate_distance_to_optimum(np.array([3.0, 3.0]), np.array([3.0, 3.0]))
        assert distance == 0.0

    def test_multiple_optima_returns_nearest(self):
        point = np.array([0.0, 0.0])
        optima = [np.array([10.0, 10.0]), np.array([1.0, 0.0])]
        distance = calculate_distance_to_optimum(point, optima)
        assert np.isclose(distance, 1.0)
