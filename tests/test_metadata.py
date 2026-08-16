"""
Tests for bananabench.metadata: the BENCHMARK_SUITE registry and its
accessor functions (get_function_info, get_all_functions, get_bounds,
get_function_list).

Mathematical correctness of known_minimum/optimal_point is covered by
test_functions.py; this file only tests the registry's own API contract.
"""

import numpy as np
import pytest

from bananabench.metadata import (
    BENCHMARK_SUITE,
    get_all_functions,
    get_bounds,
    get_function_info,
    get_function_list,
)

REQUIRED_FIELDS = [
    "function",
    "properties",
    "bounds",
    "default_dim",
    "known_minimum",
    "optimal_point",
]


class TestBenchmarkSuiteCompleteness:
    def test_has_at_least_55_functions(self):
        # Don't hardcode the exact count — adding new functions should not break this test.
        assert len(BENCHMARK_SUITE) >= 55

    @pytest.mark.parametrize("name", sorted(BENCHMARK_SUITE))
    def test_entry_has_required_fields(self, name):
        meta = BENCHMARK_SUITE[name]
        for field in REQUIRED_FIELDS:
            assert field in meta, f"{name} is missing required field {field!r}"

    @pytest.mark.parametrize("name", sorted(BENCHMARK_SUITE))
    def test_entry_function_is_callable(self, name):
        assert callable(BENCHMARK_SUITE[name]["function"])

    @pytest.mark.parametrize("name", sorted(BENCHMARK_SUITE))
    def test_default_dim_is_positive_int(self, name):
        assert isinstance(BENCHMARK_SUITE[name]["default_dim"], int)
        assert BENCHMARK_SUITE[name]["default_dim"] > 0

    @pytest.mark.parametrize("name", sorted(BENCHMARK_SUITE))
    def test_bounds_are_well_formed(self, name):
        bounds = BENCHMARK_SUITE[name]["bounds"]
        assert len(bounds) >= 1
        for low, high in bounds:
            assert low < high

    @pytest.mark.parametrize("name", sorted(BENCHMARK_SUITE))
    def test_properties_is_nonempty_list_of_strings(self, name):
        properties = BENCHMARK_SUITE[name]["properties"]
        assert isinstance(properties, list)
        assert len(properties) > 0
        assert all(isinstance(p, str) for p in properties)


class TestGetFunctionInfo:
    def test_returns_registry_entry(self):
        info = get_function_info("ackley")
        assert info is BENCHMARK_SUITE["ackley"]

    def test_unknown_function_raises_value_error(self):
        with pytest.raises(ValueError, match="not found in benchmark suite"):
            get_function_info("not_a_real_function")

    def test_error_message_lists_available_functions(self):
        with pytest.raises(ValueError, match="sphere"):
            get_function_info("not_a_real_function")


class TestGetAllFunctions:
    def test_returns_sorted_list(self):
        functions = get_all_functions()
        assert functions == sorted(functions)

    def test_matches_registry_keys(self):
        assert set(get_all_functions()) == set(BENCHMARK_SUITE)

    def test_contains_well_known_functions(self):
        functions = get_all_functions()
        for name in ("ackley", "sphere", "rosenbrock"):
            assert name in functions


class TestGetBounds:
    def test_default_dimension_replicates_single_bound(self):
        bounds = get_bounds("ackley")
        assert len(bounds) == BENCHMARK_SUITE["ackley"]["default_dim"]
        assert all(b == (-30, 30) for b in bounds)

    def test_custom_dimension_replicates_single_bound(self):
        bounds = get_bounds("sphere", dim=5)
        assert len(bounds) == 5
        assert all(b == (-100, 100) for b in bounds)

    def test_per_dimension_bounds_returned_as_is(self):
        bounds = get_bounds("branin")
        assert bounds == [(-5, 10), (0, 15)]

    def test_per_dimension_bounds_ignore_dim_override(self):
        # branin's bounds are inherently 2D; a dim override can't stretch them.
        assert get_bounds("branin", dim=5) == get_bounds("branin")


class TestGetFunctionList:
    def test_returns_string_listing_every_function(self):
        listing = get_function_list()
        assert isinstance(listing, str)
        for name in get_all_functions():
            assert name in listing


class TestMetadataAgreesWithFunctionsModule:
    """The registry and bananabench.functions must reference the same callables."""

    @pytest.mark.parametrize("name", sorted(BENCHMARK_SUITE))
    def test_registry_function_is_from_functions_module(self, name):
        from bananabench import functions as _f

        registered = BENCHMARK_SUITE[name]["function"]
        assert registered is getattr(_f, name)
