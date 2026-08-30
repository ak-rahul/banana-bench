"""
Tests for bananabench.functions: correctness of each benchmark function at
its documented global minimum, general numerical properties, and dimension
validation.

Global-minimum test cases are driven directly by ``BENCHMARK_SUITE`` (see
bananabench.metadata) instead of a hand-duplicated list, so every function
registered there is exercised automatically and the suite can't silently
drift out of sync as functions are added or their metadata changes.

References
----------
[1] Adorio, E. P. (2005). MVF - Multivariate Test Functions Library in C
    for Unconstrained Global Optimization.
"""

import numpy as np
import pytest

from bananabench import BENCHMARK_SUITE, fraudenstein_roth, freudenstein_roth, maxmod, schwefel2_21
from bananabench.metadata import get_all_functions, get_bounds

# Functions whose true minimum is achieved at more than one point. Each is
# tested separately against every one of its listed optima.
_MULTI_MINIMA_POINTS = {
    "branin": [(-np.pi, 12.275), (np.pi, 2.275), (9.42478, 2.475)],
    "camel6": [(0.08983, -0.7126), (-0.08983, 0.7126)],
    "trecanni": [(0.0, 0.0), (-2.0, 0.0)],
}

# Functions with no single known optimal_point recorded (approximate/complex
# optima from the literature) — covered only by the weaker sanity checks
# below, not an exact known-minimum assertion.
_NO_KNOWN_POINT = {"michalewicz", "langerman", "stretched_v", "watson", "xor", "lennard_jones"}

# holzman1: at its documented optimum (50, 25, 1.5) the shipped implementation
# returns ~13.74, not 0, and unconstrained search inside its own documented
# box finds far lower (and even NaN-producing) values. This looks like a real
# bug in the function body, not a metadata typo — left for a follow-up fix
# rather than guessing at a corrected formula here. xfail(strict=True) so this
# starts failing loudly (telling us to update/remove the marker) if the
# implementation is ever corrected.
_KNOWN_BROKEN = {"holzman1"}

_SINGLE_POINT_FUNCTIONS = sorted(
    set(BENCHMARK_SUITE) - set(_MULTI_MINIMA_POINTS) - _NO_KNOWN_POINT - _KNOWN_BROKEN
)

# A handful of known_minimum values are literature-rounded approximations
# rather than exact closed forms; give those a looser tolerance.
_LOOSE_TOLERANCE = {
    "mccormick": 1e-3,
    "eggholder": 1e-3,
    "gear": 1e-6,
    "schwefel2_26": 1e-3,
}


def _optimal_point_array(info):
    """Broadcast a single-element optimal_point to default_dim, as bounds are."""
    point = np.asarray(info["optimal_point"], dtype=float)
    if point.size == 1 and info["default_dim"] > 1:
        return np.full(info["default_dim"], point[0])
    return point


def _sample_within_bounds(name, dim=None):
    """A deterministic, in-domain point — avoids degenerate inputs like the all-zero
    vector, which sits outside gear's [12, 60] domain and makes every particle in
    lennard_jones coincide (both produce spurious divide-by-zero/NaN)."""
    bounds = get_bounds(name, dim=dim)
    low = np.array([b[0] for b in bounds])
    high = np.array([b[1] for b in bounds])
    rng = np.random.default_rng(0)
    return rng.uniform(low, high)


class TestGlobalMinima:
    """Every function in BENCHMARK_SUITE reaches its claimed minimum at its claimed point."""

    @pytest.mark.parametrize("name", _SINGLE_POINT_FUNCTIONS)
    def test_known_minimum_at_optimal_point(self, name):
        info = BENCHMARK_SUITE[name]
        x = _optimal_point_array(info)
        tol = _LOOSE_TOLERANCE.get(name, 1e-4)
        assert info["function"](x) == pytest.approx(info["known_minimum"], abs=tol)

    @pytest.mark.parametrize(
        "name,point",
        [(name, point) for name, points in _MULTI_MINIMA_POINTS.items() for point in points],
    )
    def test_known_minimum_at_each_multiple_optimum(self, name, point):
        info = BENCHMARK_SUITE[name]
        result = info["function"](np.array(point))
        assert result == pytest.approx(info["known_minimum"], abs=1e-3)

    @pytest.mark.xfail(
        reason="holzman1 does not reach known_minimum=0 at its documented optimum (50,25,1.5); "
        "likely an implementation bug, see module docstring",
        strict=True,
    )
    def test_holzman1_known_minimum(self):
        info = BENCHMARK_SUITE["holzman1"]
        x = _optimal_point_array(info)
        assert info["function"](x) == pytest.approx(info["known_minimum"], abs=1e-4)

    def test_freudenstein_roth_minimum(self):
        """Freudenstein-Roth: f(5, 4) = 0. Not in BENCHMARK_SUITE, tested directly."""
        x = np.array([5.0, 4.0])
        assert abs(fraudenstein_roth(x)) < 1e-8
        assert abs(freudenstein_roth(x)) < 1e-8

    def test_freudenstein_roth_spelling_alias(self):
        """freudenstein_roth is the corrected spelling of fraudenstein_roth; same function."""
        assert freudenstein_roth is fraudenstein_roth


class TestApproximateOptima:
    """Functions whose known_minimum comes from a complex/approximate literature optimum.

    No single exact optimal_point is recorded for these, so we only sanity-check that
    the function is finite/callable at its default dimension and that nothing in a
    modest random sample undercuts the claimed known_minimum (which would indicate the
    claimed value is wrong, not just approximate).
    """

    @pytest.mark.parametrize("name", sorted(_NO_KNOWN_POINT))
    def test_finite_at_default_dimension(self, name):
        x = _sample_within_bounds(name)
        assert np.isfinite(BENCHMARK_SUITE[name]["function"](x))

    @pytest.mark.parametrize("name", sorted(_NO_KNOWN_POINT))
    def test_random_samples_do_not_undercut_known_minimum(self, name):
        info = BENCHMARK_SUITE[name]
        bounds = get_bounds(name)
        rng = np.random.default_rng(0)
        low = np.array([b[0] for b in bounds])
        high = np.array([b[1] for b in bounds])
        for _ in range(200):
            x = rng.uniform(low, high)
            value = info["function"](x)
            assert value >= info["known_minimum"] - 1e-6, (
                f"{name}({x}) = {value} undercuts known_minimum={info['known_minimum']}"
            )


class TestFunctionProperties:
    """General numerical properties every registered function should have."""

    @pytest.mark.parametrize("name", get_all_functions())
    def test_returns_finite_float_at_default_dimension(self, name):
        x = _sample_within_bounds(name)
        result = BENCHMARK_SUITE[name]["function"](x)
        assert isinstance(result, float)
        assert np.isfinite(result)

    @pytest.mark.parametrize("name", get_all_functions())
    def test_accepts_plain_list_input(self, name):
        """Every function coerces via np.asarray, so plain lists (not just ndarrays) must work."""
        x = list(_sample_within_bounds(name))
        result = BENCHMARK_SUITE[name]["function"](x)
        assert np.isfinite(result)

    @pytest.mark.parametrize(
        "name", [n for n in get_all_functions() if "scalable" in BENCHMARK_SUITE[n]["properties"]]
    )
    def test_scalable_functions_accept_other_dimensions(self, name):
        if name == "langerman":
            pytest.skip("langerman has a hardcoded 10-dimensional coefficient table")
        if name == "powell":
            pytest.skip("powell requires a dimension divisible by 4")
        for dim in (2, 6, 10):  # even dims only: rosenbrock_ext1/ext2 require pairs
            result = BENCHMARK_SUITE[name]["function"](_sample_within_bounds(name, dim=dim))
            assert np.isfinite(result)

    @pytest.mark.parametrize(
        "name",
        ["sphere", "sphere2", "sum_squares", "hyperellipsoid", "schwefel1_2", "schwefel2_21"],
    )
    def test_convex_unimodal_functions_are_nonnegative(self, name):
        info = BENCHMARK_SUITE[name]
        rng = np.random.default_rng(0)
        x = rng.uniform(-10, 10, info["default_dim"])
        assert info["function"](x) >= 0


class TestDimensionValidation:
    """Fixed-dimension functions must reject input of the wrong size."""

    @pytest.mark.parametrize(
        "name,expected_dim",
        [
            ("beale", 2),
            ("bohachevsky1", 2),
            ("bohachevsky2", 2),
            ("booth", 2),
            ("box_betts", 3),
            ("branin", 2),
            ("camel3", 2),
            ("camel6", 2),
            ("chichinadze", 2),
            ("colville", 4),
            ("easom", 2),
            ("gear", 4),
            ("goldstein_price", 2),
            ("himmelblau", 2),
            ("hosaki", 2),
            ("kowalik", 4),
            ("leon", 2),
            ("matyas", 2),
            ("mccormick", 2),
            ("rastrigin2", 2),
            ("schaffer1", 2),
            ("schaffer2", 2),
            ("trecanni", 2),
            ("trefethen4", 2),
            ("watson", 6),
            ("xor", 9),
            ("zettl", 2),
            ("zimmerman", 2),
            ("hansen", 2),
            ("hartman3", 3),
            ("hartman6", 6),
            ("neumaier_powersum", 4),
            ("paviani", 10),
            ("plateau", 5),
            ("powell", 4),
            ("shekel2", 2),
            ("shekel4_5", 4),
            ("shekel4_7", 4),
            ("shekel4_10", 4),
        ],
    )
    def test_wrong_dimension_raises_value_error(self, name, expected_dim):
        func = BENCHMARK_SUITE[name]["function"]
        wrong_size = expected_dim + 1
        with pytest.raises(ValueError):
            func(np.zeros(wrong_size))


class TestMaxmodDeprecation:
    """maxmod is deprecated in favor of the identical schwefel2_21."""

    def test_maxmod_warns(self):
        with pytest.deprecated_call():
            maxmod(np.zeros(5))

    def test_maxmod_matches_schwefel2_21(self):
        x = np.array([1.0, -2.0, 3.0, -4.0])
        with pytest.deprecated_call():
            maxmod_value = maxmod(x)
        assert maxmod_value == schwefel2_21(x)
