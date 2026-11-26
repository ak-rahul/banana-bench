"""
Tests for bananabench.benchmarking: BenchmarkRunner and quick_benchmark.
"""

import csv
import tempfile
from pathlib import Path

import numpy as np
import pytest

from bananabench import multiobjective as mo
from bananabench.benchmarking import (
    BenchmarkRunner,
    MOBenchmarkRunner,
    compare_algorithms,
    quick_benchmark,
)


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


def far_optimizer(func, bounds, **kwargs):
    """Deterministic optimizer that returns each bound's upper edge -- a worse result than
    zero_optimizer for every function in the suite, for comparison-ranking tests."""
    x = np.array([b[1] for b in bounds])
    return x, func(x)


def failing_optimizer(func, bounds, **kwargs):
    """Optimizer that never succeeds, for zero-successful-runs tests."""
    return None, None


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


def endless_optimizer(func, bounds, **kwargs):
    """Random search with no stopping rule: only a budget can stop it."""
    rng = np.random.default_rng(0)
    low = np.array([b[0] for b in bounds])
    high = np.array([b[1] for b in bounds])
    while True:
        func(rng.uniform(low, high))


def lying_optimizer(func, bounds, **kwargs):
    """Reports a perfect score its point doesn't achieve."""
    x = np.array([b[1] for b in bounds], dtype=float)
    func(x)
    return x, 0.0


class TestEvaluationBudget:
    def test_budget_stops_an_endless_algorithm(self):
        runner = BenchmarkRunner(endless_optimizer, max_evals=250, verbose=False)
        result = runner.run_single("sphere", dim=3)
        assert result["status"] == "budget_exhausted"
        assert result["success"] is True
        assert result["n_evals"] == 250
        assert result["max_evals"] == 250

    def test_budget_exhausted_run_is_scored_on_best_so_far(self):
        runner = BenchmarkRunner(endless_optimizer, max_evals=500, verbose=False)
        result = runner.run_single("sphere", dim=2)
        assert np.isfinite(result["best_cost"])
        assert result["cost_mismatch"] is False
        assert result["error"] == pytest.approx(result["best_cost"])

    def test_budget_as_function_of_dimension(self):
        runner = BenchmarkRunner(endless_optimizer, max_evals=lambda d: 100 * d, verbose=False)
        assert runner.run_single("sphere", dim=2)["n_evals"] == 200
        assert runner.run_single("sphere", dim=5)["n_evals"] == 500

    def test_invalid_budget_raises(self):
        runner = BenchmarkRunner(zero_optimizer, max_evals=lambda d: 0, verbose=False)
        with pytest.raises(ValueError):
            runner.run_single("sphere", dim=2)

    def test_evaluations_counted_without_budget(self):
        runner = BenchmarkRunner(simple_optimizer, verbose=False)
        result = runner.run_single("sphere", dim=2, max_iter=40)
        assert result["n_evals"] == 40
        assert result["max_evals"] is None

    def test_failed_runs_still_report_evaluations(self):
        def crashes_after_some_work(func, bounds, **kwargs):
            for _ in range(5):
                func(np.zeros(len(bounds)))
            raise RuntimeError("boom")

        result = BenchmarkRunner(crashes_after_some_work, verbose=False).run_single("sphere", dim=2)
        assert result["status"] == "exception"
        assert result["n_evals"] == 5

    def test_suite_aggregates_evaluations(self):
        runner = BenchmarkRunner(
            endless_optimizer, n_runs=3, max_evals=120, verbose=False, show_progress=False
        )
        (agg,) = runner.run_suite(functions=["sphere"], dimensions={"sphere": 2})
        assert agg["n_evals_mean"] == 120
        assert agg["n_budget_exhausted"] == 3

    def test_parallel_matches_serial_under_budget(self):
        kwargs = dict(n_runs=2, seed=7, max_evals=200, verbose=False, show_progress=False)
        serial = BenchmarkRunner(endless_optimizer, n_jobs=1, **kwargs).run_suite(
            functions=["sphere"]
        )
        parallel = BenchmarkRunner(endless_optimizer, n_jobs=2, **kwargs).run_suite(
            functions=["sphere"]
        )
        assert serial[0]["error_mean"] == parallel[0]["error_mean"]
        assert serial[0]["n_evals_mean"] == parallel[0]["n_evals_mean"] == 200


class TestVerification:
    def test_lying_algorithm_is_scored_on_the_truth(self):
        result = BenchmarkRunner(lying_optimizer, verbose=False).run_single("sphere", dim=2)
        assert result["reported_cost"] == 0.0
        assert result["cost_mismatch"] is True
        # sphere's upper bound corner is far from zero; the error must reflect that.
        assert result["best_cost"] > 1.0
        assert result["error"] == pytest.approx(result["best_cost"])

    def test_honest_algorithm_has_no_mismatch(self):
        result = BenchmarkRunner(zero_optimizer, verbose=False).run_single("sphere", dim=2)
        assert result["cost_mismatch"] is False
        assert result["best_cost"] == result["reported_cost"]

    def test_verify_false_scores_the_reported_cost(self):
        runner = BenchmarkRunner(lying_optimizer, verify=False, verbose=False)
        result = runner.run_single("sphere", dim=2)
        assert result["best_cost"] == 0.0
        assert result["cost_mismatch"] is None

    def test_verification_is_not_counted_against_the_budget(self):
        runner = BenchmarkRunner(endless_optimizer, max_evals=50, verbose=False)
        assert runner.run_single("sphere", dim=2)["n_evals"] == 50

    def test_summary_warns_about_mismatches(self, capsys):
        BenchmarkRunner(lying_optimizer, verbose=True, show_progress=False).run_suite(
            functions=["sphere"]
        )
        assert "does not match f(best_x)" in capsys.readouterr().out

    def test_compare_algorithms_shows_evaluations(self):
        a = BenchmarkRunner(endless_optimizer, algorithm_name="A", max_evals=100, verbose=False)
        b = BenchmarkRunner(zero_optimizer, algorithm_name="B", verbose=False)
        for runner in (a, b):
            runner.run_suite(functions=["sphere"])
        table = compare_algorithms([a, b])
        assert "Evals Mean" in table
        assert "100" in table


class TestTraces:
    def test_trace_recorded_when_requested(self):
        runner = BenchmarkRunner(endless_optimizer, max_evals=300, record_trace=True, verbose=False)
        result = runner.run_single("sphere", dim=2)
        trace = np.array(result["trace"])
        assert trace.shape[1] == 2
        assert trace[0, 0] == 1
        assert result["first_cost"] == trace[0, 1]
        assert trace[-1, 1] == pytest.approx(result["best_cost"])

    def test_no_trace_by_default(self):
        result = BenchmarkRunner(zero_optimizer, verbose=False).run_single("sphere", dim=2)
        assert "trace" not in result

    def test_json_keeps_traces_and_csv_drops_them(self, tmp_path):
        import json

        runner = BenchmarkRunner(
            endless_optimizer, max_evals=50, record_trace=True, verbose=False, show_progress=False
        )
        runner.run_suite(functions=["sphere"])
        runner.save_results(tmp_path / "r.json", format="json")
        runner.save_results(tmp_path / "r.csv", format="csv")
        assert "trace" in json.loads((tmp_path / "r.json").read_text())[0]
        with open(tmp_path / "r.csv", newline="") as f:
            assert "trace" not in next(csv.DictReader(f))


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

    def test_verbose_table_shows_tiny_known_minimum(self, capsys):
        """gear's known minimum (~2.7e-12) used to print as 0.000000 in the verbose table."""

        def lower_edge_optimizer(func, bounds, **kwargs):
            # gear is undefined at the origin; its lower bound edge (12, 12, 12, 12) is in-domain.
            x = np.array([b[0] for b in bounds], dtype=float)
            return x, func(x)

        BenchmarkRunner(lower_edge_optimizer, verbose=True, show_progress=False).run_suite(
            functions=["gear"]
        )
        gear_rows = [
            line for line in capsys.readouterr().out.splitlines() if line.startswith("gear ")
        ]
        assert gear_rows, "gear row missing from verbose output"
        assert "e-12" in gear_rows[0]


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

    def test_save_csv_with_mixed_success_and_failure_results(self):
        # A run mixing successes and failures produces result dicts with different
        # key sets (run_single's exception branch returns fewer keys than a
        # successful run), so fieldnames must be the union across every result --
        # csv.DictWriter raises ValueError if any row has a key outside fieldnames.
        runner = BenchmarkRunner(failing_optimizer, verbose=False, show_progress=False)
        runner.run_suite(functions=["sphere"])

        good_runner = BenchmarkRunner(zero_optimizer, verbose=False, show_progress=False)
        good_runner.run_suite(functions=["beale"])
        runner.results.extend(good_runner.results)

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            filepath = Path(f.name)
        try:
            runner.save_results(filepath, format="csv")  # must not raise
            with open(filepath, newline="") as saved:
                rows = list(csv.DictReader(saved))
            assert len(rows) == 2
            assert {"sphere", "beale"} == {row["function"] for row in rows}
        finally:
            filepath.unlink()


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


class TestCompareAlgorithms:
    def test_rejects_fewer_than_two_runners(self):
        runner = BenchmarkRunner(zero_optimizer, algorithm_name="Solo", verbose=False)
        runner.run_suite(functions=["sphere"])
        with pytest.raises(ValueError):
            compare_algorithms([runner])

    def test_ranks_lower_error_first(self):
        good = BenchmarkRunner(zero_optimizer, algorithm_name="Good", verbose=False)
        good.run_suite(functions=["sphere", "ackley"])

        bad = BenchmarkRunner(far_optimizer, algorithm_name="Bad", verbose=False)
        bad.run_suite(functions=["sphere", "ackley"])

        table = compare_algorithms([bad, good])  # deliberately passed worse-first
        # Good's row must appear before Bad's row in the rendered table.
        assert table.index("Good") < table.index("Bad")

    def test_zero_successful_runs_sorts_last_and_does_not_crash(self):
        good = BenchmarkRunner(zero_optimizer, algorithm_name="Good", verbose=False)
        good.run_suite(functions=["sphere"])

        failing = BenchmarkRunner(failing_optimizer, algorithm_name="Failing", verbose=False)
        failing.run_suite(functions=["sphere"])

        table = compare_algorithms([failing, good])
        assert table.index("Good") < table.index("Failing")
        assert "no successful runs" in table

    def test_output_mentions_every_algorithm_name(self):
        good = BenchmarkRunner(zero_optimizer, algorithm_name="AlphaAlgo", verbose=False)
        good.run_suite(functions=["sphere"])
        bad = BenchmarkRunner(far_optimizer, algorithm_name="BetaAlgo", verbose=False)
        bad.run_suite(functions=["sphere"])

        table = compare_algorithms([good, bad])
        assert "AlphaAlgo" in table
        assert "BetaAlgo" in table


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

    def test_algorithm_exception_returns_failure_dict_instead_of_raising(self):
        def failing_mo(func, bounds, n_objectives, **kwargs):
            raise RuntimeError("mo boom")

        runner = MOBenchmarkRunner(failing_mo, verbose=False, seed=1)
        result = runner.run_single("zdt1", dim=5)
        assert result["function"] == "zdt1"
        assert result["success"] is False
        assert result["status"] == "exception"
        assert "mo boom" in result["error_message"]

    def test_successful_result_is_marked_success(self):
        runner = MOBenchmarkRunner(random_search_mo, verbose=False, seed=1)
        result = runner.run_single("zdt1", dim=5, n_points=20)
        assert result["success"] is True
        assert result["status"] == "success"


class TestMORunSuite:
    def test_runs_every_mo_function_by_default(self):
        runner = MOBenchmarkRunner(random_search_mo, verbose=False, seed=1)
        results = runner.run_suite(n_points=10)
        assert {r["function"] for r in results} == set(mo.get_mo_function_list())

    def test_runs_selected_functions_and_stores_results(self):
        runner = MOBenchmarkRunner(random_search_mo, verbose=False, seed=1)
        results = runner.run_suite(functions=["zdt1", "zdt2"], n_points=10)
        assert len(results) == 2
        assert len(runner.results) == 2

    def test_one_failing_function_does_not_abort_the_rest_of_the_suite(self):
        # A single bad algorithm call must not take down the whole batch --
        # run_single isolates exceptions the same way BenchmarkRunner.run_single
        # does, so run_suite (which has no try/except of its own around the
        # run_single call) still returns a result for every requested function.
        def flaky_mo(func, bounds, n_objectives, **kwargs):
            if len(bounds) == 30:  # zdt1's default_dim -- fail only this one
                raise RuntimeError("mo boom")
            return random_search_mo(func, bounds, n_objectives, **kwargs)

        runner = MOBenchmarkRunner(flaky_mo, verbose=False, seed=1)
        results = runner.run_suite(functions=["zdt1", "zdt4"], n_points=10)  # zdt4 default_dim=10
        assert len(results) == 2
        by_name = {r["function"]: r for r in results}
        assert by_name["zdt1"]["success"] is False
        assert by_name["zdt4"]["success"] is True


class TestMODTLZ:
    def test_dtlz_runs_with_three_objectives_by_default(self):
        result = MOBenchmarkRunner(random_search_mo, verbose=False, seed=1).run_single(
            "dtlz2", n_points=30
        )
        assert result["success"] is True
        assert result["n_objectives"] == 3
        assert result["dimension"] == 12  # M + k - 1 with k = 10
        assert result["igd"] > 0.0
        assert result["hypervolume"] >= 0.0

    def test_n_objectives_override(self):
        runner = MOBenchmarkRunner(random_search_mo, verbose=False, seed=1, n_objectives=5)
        result = runner.run_single("dtlz1", n_points=20)
        assert result["n_objectives"] == 5
        assert result["dimension"] == 9  # 5 + 5 - 1

    def test_pareto_optimal_points_score_near_zero_igd(self):
        def optimal_dtlz2(func, bounds, n_objectives, **kwargs):
            rng = np.random.default_rng(0)
            X = rng.random((400, len(bounds)))
            X[:, n_objectives - 1 :] = 0.5
            return X, np.array([func(x) for x in X])

        result = MOBenchmarkRunner(optimal_dtlz2, verbose=False).run_single("dtlz2")
        assert result["igd"] < 0.1

    def test_zdt_rejects_more_objectives(self):
        runner = MOBenchmarkRunner(random_search_mo, verbose=False, n_objectives=3)
        with pytest.raises(ValueError):
            runner.run_single("zdt1", dim=5)


class TestMOBudget:
    def test_counts_evaluations(self):
        result = MOBenchmarkRunner(random_search_mo, verbose=False, seed=1).run_single(
            "zdt1", dim=5, n_points=25
        )
        assert result["n_evals"] == 25
        assert result["max_evals"] is None

    def test_budget_exhausted_run_is_scored_on_archive(self):
        runner = MOBenchmarkRunner(random_search_mo, verbose=False, seed=1, max_evals=15)
        result = runner.run_single("zdt1", dim=5, n_points=100)
        assert result["status"] == "budget_exhausted"
        assert result["success"] is True
        assert result["n_evals"] == 15
        assert result["n_solutions"] == 15
        assert np.isfinite(result["igd"])

    def test_budget_as_function_of_dimension(self):
        runner = MOBenchmarkRunner(random_search_mo, verbose=False, max_evals=lambda d: 2 * d)
        assert runner.run_single("zdt1", dim=5, n_points=100)["n_evals"] == 10
