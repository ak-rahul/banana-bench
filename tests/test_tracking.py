"""
Tests for bananabench.tracking: TrackedFunction and BudgetExhausted.
"""

import pickle

import numpy as np
import pytest

from bananabench import NoisyFunction, sphere
from bananabench.tracking import BudgetExhausted, TrackedFunction


class TestCounting:
    def test_counts_every_call(self):
        f = TrackedFunction(sphere)
        for _ in range(7):
            f(np.zeros(3))
        assert f.n_evals == 7

    def test_returns_exact_function_value(self):
        f = TrackedFunction(sphere)
        x = np.array([1.0, -2.0, 3.0])
        assert f(x) == sphere(x)

    def test_reset_clears_state(self):
        f = TrackedFunction(sphere, max_evals=5)
        f(np.ones(2))
        f.reset()
        assert f.n_evals == 0
        assert f.best_f == float("inf")
        assert f.best_x is None
        assert f.first_f is None
        assert f.trace.shape == (0, 2)
        assert f.budget_left == 5


class TestBudget:
    def test_exactly_max_evals_allowed(self):
        f = TrackedFunction(sphere, max_evals=3)
        for _ in range(3):
            f(np.ones(2))
        assert f.budget_left == 0
        with pytest.raises(BudgetExhausted):
            f(np.ones(2))

    def test_exceeding_call_is_not_evaluated(self):
        calls = []

        def counted(x):
            calls.append(1)
            return float(np.sum(x))

        f = TrackedFunction(counted, max_evals=2)
        f(np.ones(1))
        f(np.ones(1))
        with pytest.raises(BudgetExhausted) as excinfo:
            f(np.ones(1))
        assert len(calls) == 2
        assert f.n_evals == 2
        assert excinfo.value.max_evals == 2

    def test_unlimited_by_default(self):
        f = TrackedFunction(sphere)
        assert f.budget_left is None
        for _ in range(1000):
            f(np.ones(2))
        assert f.n_evals == 1000

    @pytest.mark.parametrize("bad", [0, -5])
    def test_rejects_non_positive_budget(self, bad):
        with pytest.raises(ValueError):
            TrackedFunction(sphere, max_evals=bad)


class TestBestSoFar:
    def test_tracks_best_value_and_point(self):
        f = TrackedFunction(sphere)
        f(np.array([3.0, 0.0]))
        f(np.array([1.0, 0.0]))
        f(np.array([2.0, 0.0]))
        assert f.best_f == 1.0
        np.testing.assert_array_equal(f.best_x, [1.0, 0.0])
        assert f.first_f == 9.0

    def test_best_x_is_a_copy(self):
        """An optimizer that mutates its array in place must not change best_x."""
        f = TrackedFunction(sphere)
        x = np.array([1.0, 1.0])
        f(x)
        x[:] = 100.0
        np.testing.assert_array_equal(f.best_x, [1.0, 1.0])

    def test_nan_and_inf_count_but_never_improve(self):
        values = iter([float("nan"), 5.0, float("inf"), float("-inf"), 4.0])
        f = TrackedFunction(lambda x: next(values))
        for _ in range(5):
            f(np.zeros(1))
        assert f.n_evals == 5
        assert f.best_f == 4.0

    def test_trace_records_first_eval_and_strict_improvements(self):
        values = iter([10.0, 12.0, 8.0, 8.0, 3.0, 5.0])
        f = TrackedFunction(lambda x: next(values))
        for _ in range(6):
            f(np.zeros(1))
        np.testing.assert_array_equal(f.trace, [[1, 10.0], [3, 8.0], [5, 3.0]])

    def test_trace_best_values_are_monotone(self):
        rng = np.random.default_rng(0)
        f = TrackedFunction(sphere)
        for _ in range(500):
            f(rng.uniform(-5, 5, 3))
        assert np.all(np.diff(f.trace[:, 1]) < 0)
        assert np.all(np.diff(f.trace[:, 0]) > 0)

    def test_trace_starts_with_non_finite_first_eval(self):
        values = iter([float("nan"), 2.0])
        f = TrackedFunction(lambda x: next(values))
        f(np.zeros(1))
        f(np.zeros(1))
        assert f.trace.shape == (2, 2)
        assert np.isnan(f.trace[0, 1])
        assert f.trace[1].tolist() == [2.0, 2.0]

    def test_record_trace_false(self):
        f = TrackedFunction(sphere, record_trace=False)
        f(np.ones(2))
        assert f.trace.shape == (0, 2)
        assert f.best_f == 2.0


class TestInterop:
    def test_pickle_round_trip(self):
        """Needed for joblib's process-based parallel backend."""
        f = TrackedFunction(sphere, max_evals=10)
        f(np.ones(2))
        clone = pickle.loads(pickle.dumps(f))
        assert clone.n_evals == 1
        assert clone.best_f == 2.0
        assert clone.max_evals == 10

    def test_composes_with_wrappers(self):
        """Wrapping order matters: the tracker records the noisy values it sees."""
        noisy = NoisyFunction(sphere, scale=0.5, seed=0)
        reference = NoisyFunction(sphere, scale=0.5, seed=0)
        f = TrackedFunction(noisy)
        x = np.ones(2)
        assert f(x) == reference(x)
        assert f.n_evals == 1

    def test_repr(self):
        assert "sphere" in repr(TrackedFunction(sphere, max_evals=3))
