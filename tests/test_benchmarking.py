"""
Tests for bananabench.benchmarking: BenchmarkRunner and quick_benchmark.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from bananabench import multiobjective as mo
from bananabench.benchmarking import BenchmarkRunner, MOBenchmarkRunner, quick_benchmark


def simple_optimizer(func, bounds, max_iter=50):
    """Random search: enough runs to be deterministic-ish, cheap enough for tests."""
    bounds_array = np.array(bounds)
    best_x = None
    best_cost = float("inf")

    for _ in range(max_iter):
        x = bounds_array[:, 0] + np.random.rand(len(bounds)) * (
            bounds_array[:, 1] - bounds_array[:, 0]
        )
        cost = func(x)
        if cost < best_cost:
            best_cost = cost
            best_x = x

    return best_x, best_cost


def zero_optimizer(func, bounds, **kwargs):
    """Deterministic optimizer that always returns the origin, for reproducibility tests."""
    return np.zeros(len(bounds)), func(np.zeros(len(bounds)))


class TestBenchmarkRunnerInit:
    def test_stores_configuration(self):
        runner = BenchmarkRunner(simple_optimizer, algorithm_name="TestAlgo", n_runs=3)
        assert runner.algorithm_name == "TestAlgo"
        assert runner.n_runs == 3

    def test_default_algorithm_name(self):
        runner = BenchmarkRunner(simple_optimizer)
        assert runner.algorithm_name == "UnnamedAlgorithm"


class TestRunSingle:
    def test_returns_result_dict(self):
        runner = BenchmarkRunner(simple_optimizer, algorithm_name="TestAlgo", verbose=False)
        result = runner.run_single("sphere", dim=2, max_iter=50)
        assert result["function"] == "sphere"
        assert result["dimension"] == 2
        assert result["success"] is True

    def test_uses_default_dim_when_not_specified(self):
        runner = BenchmarkRunner(simple_optimizer, verbose=False)
        result = runner.run_single("sphere", max_iter=50)
        assert result["dimension"] == 10  # sphere's default_dim

    def test_algorithm_exception_is_captured_not_raised(self):
        def broken_optimizer(func, bounds, **kwargs):
            raise RuntimeError("boom")

        runner = BenchmarkRunner(broken_optimizer, verbose=False)
        result = runner.run_single("sphere")
        assert result["status"] == "exception"
        assert result["success"] is False

    def test_nan_result_marked_invalid(self):
        def nan_optimizer(func, bounds, **kwargs):
            return np.zeros(len(bounds)), float("nan")

        runner = BenchmarkRunner(nan_optimizer, verbose=False)
        result = runner.run_single("sphere")
        assert result["status"] == "invalid_result"
        assert result["success"] is False


class TestRunSuite:
    def test_runs_every_requested_function(self):
        runner = BenchmarkRunner(
            simple_optimizer, algorithm_name="TestAlgo", n_runs=2, verbose=False
        )
        results = runner.run_suite(functions=["sphere", "ackley"], max_iter=50)
        assert len(results) == 2
        assert all("function" in r for r in results)

    def test_reproducible_with_seed(self):
        """A fixed seed on the runner must produce identical results across independent runs."""
        runner1 = BenchmarkRunner(zero_optimizer, n_runs=3, seed=123, verbose=False)
        results1 = runner1.run_suite(functions=["sphere"])

        runner2 = BenchmarkRunner(zero_optimizer, n_runs=3, seed=123, verbose=False)
        results2 = runner2.run_suite(functions=["sphere"])

        assert results1[0]["error_mean"] == results2[0]["error_mean"]

    def test_progress_bars_do_not_change_results(self):
        results_with = BenchmarkRunner(
            zero_optimizer, n_runs=2, seed=1, show_progress=True, verbose=False
        ).run_suite(functions=["sphere", "ackley"])
        results_without = BenchmarkRunner(
            zero_optimizer, n_runs=2, seed=1, show_progress=False, verbose=False
        ).run_suite(functions=["sphere", "ackley"])
        assert [r["error_mean"] for r in results_with] == [r["error_mean"] for r in results_without]


class TestParallelExecution:
    def test_parallel_matches_serial_result_shape(self):
        runner_serial = BenchmarkRunner(zero_optimizer, n_runs=2, n_jobs=1, verbose=False)
        results_serial = runner_serial.run_suite(functions=["sphere"])

        runner_parallel = BenchmarkRunner(zero_optimizer, n_runs=2, n_jobs=2, verbose=False)
        results_parallel = runner_parallel.run_suite(functions=["sphere"])

        assert len(results_serial) == len(results_parallel) == 1
        assert results_parallel[0]["n_runs"] == 2
        assert results_parallel[0]["success_rate"] == 1.0


class TestSaveResults:
    def test_save_csv(self):
        runner = BenchmarkRunner(simple_optimizer, verbose=False)
        runner.run_suite(functions=["sphere"], max_iter=50)

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            filepath = Path(f.name)
        try:
            runner.save_results(filepath, format="csv")
            assert filepath.exists()
        finally:
            filepath.unlink()

    def test_save_json(self):
        runner = BenchmarkRunner(simple_optimizer, verbose=False)
        runner.run_suite(functions=["sphere"], max_iter=50)

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            filepath = Path(f.name)
        try:
            runner.save_results(filepath, format="json")
            assert filepath.exists()
        finally:
            filepath.unlink()

    def test_unknown_format_raises(self):
        runner = BenchmarkRunner(simple_optimizer, verbose=False)
        runner.run_suite(functions=["sphere"], max_iter=50)
        with pytest.raises(ValueError):
            runner.save_results("out.xyz", format="xyz")


class TestSummaryStats:
    def test_reports_success_counts_and_error_stats(self):
        runner = BenchmarkRunner(simple_optimizer, verbose=False)
        runner.run_suite(functions=["sphere", "ackley"], max_iter=50)
        stats = runner.get_summary_stats()
        assert "n_results" in stats
        assert "n_successful" in stats
        assert "error_mean" in stats
        assert stats["n_successful"] <= stats["n_results"]

    def test_empty_results_reports_zero_successful(self):
        runner = BenchmarkRunner(simple_optimizer, verbose=False)
        stats = runner.get_summary_stats()
        assert stats == {"n_results": 0, "n_successful": 0}


def test_quick_benchmark():
    results = quick_benchmark(
        simple_optimizer, function_names=["sphere", "ackley"], n_runs=2, max_iter=50
    )
    assert len(results) == 2


def test_quick_benchmark_default_function_subset():
    results = quick_benchmark(simple_optimizer, n_runs=1, max_iter=20, show_progress=False)
    assert len(results) == 10  # documented default subset size


def random_search_mo(func, bounds, n_objectives, n_points=30, seed=0):
    """Cheap, deterministic-given-seed multi-objective 'optimizer' for tests."""
    rng = np.random.default_rng(seed)
    bounds_array = np.array(bounds)
    lo, hi = bounds_array[:, 0], bounds_array[:, 1]
    X = lo + rng.random((n_points, len(bounds))) * (hi - lo)
    F = np.array([func(x) for x in X])
    return X, F


class TestMOBenchmarkRunnerInit:
    def test_stores_configuration(self):
        runner = MOBenchmarkRunner(random_search_mo, algorithm_name="TestMOAlgo", seed=1)
        assert runner.algorithm_name == "TestMOAlgo"
        assert runner.seed == 1
        assert runner.results == []

    def test_default_algorithm_name(self):
        runner = MOBenchmarkRunner(random_search_mo)
        assert runner.algorithm_name == "UnnamedMOAlgorithm"


class TestMORunSingle:
    def test_returns_result_dict_with_igd_and_hypervolume(self):
        runner = MOBenchmarkRunner(random_search_mo, verbose=False, seed=1)
        result = runner.run_single("zdt1", dim=5, n_points=20)
        assert result["function"] == "zdt1"
        assert result["dimension"] == 5
        assert result["n_solutions"] == 20
        assert result["igd"] >= 0.0
        assert result["hypervolume"] >= 0.0

    def test_uses_default_dim_when_not_specified(self):
        runner = MOBenchmarkRunner(random_search_mo, verbose=False, seed=1)
        result = runner.run_single("zdt1", n_points=10)
        assert result["dimension"] == 30  # zdt1's default_dim

    def test_custom_reference_point(self):
        runner = MOBenchmarkRunner(random_search_mo, verbose=False, seed=1)
        result = runner.run_single("zdt1", dim=5, n_points=20, reference_point=np.array([2.0, 2.0]))
        assert result["hypervolume"] >= 0.0


class TestMORunSuite:
    def test_runs_every_zdt_function_by_default(self):
        runner = MOBenchmarkRunner(random_search_mo, verbose=False, seed=1)
        results = runner.run_suite(n_points=10)
        assert {r["function"] for r in results} == set(mo.get_mo_function_list())

    def test_runs_selected_functions_and_stores_results(self):
        runner = MOBenchmarkRunner(random_search_mo, verbose=False, seed=1)
        results = runner.run_suite(functions=["zdt1", "zdt2"], n_points=10)
        assert len(results) == 2
        assert len(runner.results) == 2
