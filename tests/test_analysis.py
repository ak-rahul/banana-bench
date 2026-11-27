"""
Tests for bananabench.analysis. Expected values are worked by hand from each
method's defining paper (Dolan & Moré 2002; Moré & Wild 2009; COCO; Demšar 2006).
"""

import numpy as np
import pytest

from bananabench import analysis
from bananabench.analysis import (
    COCO_TARGETS,
    _holm,
    data_profile,
    ert_from_runners,
    evaluations_to_target,
    expected_running_time,
    histories_from_runners,
    performance_profile,
    performance_profile_from_runners,
    runtime_ecdf,
    statistical_comparison,
)
from bananabench.benchmarking import BenchmarkRunner


def _random_search(func, bounds, seed=0, **kwargs):
    rng = np.random.default_rng(seed)
    low = np.array([b[0] for b in bounds])
    high = np.array([b[1] for b in bounds])
    while True:
        func(rng.uniform(low, high))


def _shrinking_search(func, bounds, **kwargs):
    """Converges quickly on sphere: samples ever closer to the origin."""
    dim = len(bounds)
    rng = np.random.default_rng(1)
    scale = 1.0
    while True:
        func(rng.normal(0, scale, dim))
        scale *= 0.97


class TestPerformanceProfile:
    def test_hand_computed_profile(self):
        costs = {
            "A": {"p1": 1.0, "p2": 4.0, "p3": np.inf},
            "B": {"p1": 2.0, "p2": 2.0, "p3": 3.0},
        }
        # ratios  p1: A=1, B=2   p2: A=2, B=1   p3: A=inf, B=1
        taus, rho = performance_profile(costs)
        np.testing.assert_array_equal(taus, [1.0, 2.0])
        np.testing.assert_allclose(rho["A"], [1 / 3, 2 / 3])
        np.testing.assert_allclose(rho["B"], [2 / 3, 1.0])

    def test_missing_problem_counts_as_failure(self):
        taus, rho = performance_profile({"A": {"p1": 1.0}, "B": {"p1": 1.0, "p2": 5.0}})
        assert rho["A"][-1] == 0.5
        assert rho["B"][-1] == 1.0

    def test_zero_best_cost(self):
        _, rho = performance_profile({"A": {"p": 0.0}, "B": {"p": 5.0}})
        assert rho["A"][0] == 1.0
        assert rho["B"][-1] == 0.0

    def test_custom_taus(self):
        taus, rho = performance_profile({"A": {"p": 1.0}, "B": {"p": 3.0}}, taus=[1, 2, 4])
        np.testing.assert_array_equal(rho["B"], [0, 0, 1])

    def test_rejects_negative_costs(self):
        with pytest.raises(ValueError):
            performance_profile({"A": {"p": -1.0}})

    def test_profile_is_monotone(self):
        rng = np.random.default_rng(0)
        costs = {s: {p: float(rng.uniform(1, 100)) for p in range(30)} for s in "ABC"}
        _, rho = performance_profile(costs)
        for curve in rho.values():
            assert np.all(np.diff(curve) >= 0)
            assert curve[-1] == 1.0


class TestTraceMetrics:
    TRACE = [[1, 10.0], [3, 8.0], [5, 3.0]]

    @pytest.mark.parametrize("target,expected", [(10.0, 1.0), (8.0, 3.0), (3.0, 5.0)])
    def test_evaluations_to_target(self, target, expected):
        assert evaluations_to_target(self.TRACE, target) == expected

    def test_unreached_target_is_inf(self):
        assert evaluations_to_target(self.TRACE, 2.0) == np.inf

    def test_expected_running_time(self):
        # (10 + 30 successful) + (100 unsuccessful) over 2 successes
        assert expected_running_time([10, np.inf, 30], [10, 100, 30]) == 70.0

    def test_ert_all_successful_is_mean_runtime(self):
        assert expected_running_time([10, 20], [50, 50]) == 15.0

    def test_ert_no_success_is_inf(self):
        assert expected_running_time([np.inf, np.inf], [100, 100]) == np.inf

    def test_ert_length_mismatch(self):
        with pytest.raises(ValueError):
            expected_running_time([1, 2], [3])


class TestDataProfile:
    def test_hand_computed_profile(self):
        # One 1-D problem, so a simplex gradient is n + 1 = 2 evaluations.
        histories = {
            "A": {"p": (10.0, [[1, 10.0], [4, 1.0]])},
            "B": {"p": (10.0, [[1, 10.0], [2, 5.0]])},
        }
        # f_L = 1 (best of all solvers); target = f_L + tau (f0 - f_L) = 1 + 0.1 * 9 = 1.9
        alphas, d = data_profile(histories, dims={"p": 1}, tau=0.1)
        np.testing.assert_array_equal(alphas, [0.0, 2.0])
        np.testing.assert_array_equal(d["A"], [0.0, 1.0])
        np.testing.assert_array_equal(d["B"], [0.0, 0.0])

    def test_explicit_f_lowest(self):
        histories = {"A": {"p": (10.0, [[1, 10.0], [4, 1.0]])}}
        # With f_L = 0, the target is 0 + 0.1 * 10 = 1.0: still reached at eval 4.
        _, d = data_profile(histories, dims={"p": 1}, tau=0.1, f_lowest={"p": 0.0})
        assert d["A"][-1] == 1.0
        # With tau = 0.01 the target is 0.1: never reached.
        _, d = data_profile(histories, dims={"p": 1}, tau=0.01, f_lowest={"p": 0.0})
        assert d["A"][-1] == 0.0

    def test_rejects_bad_tau(self):
        with pytest.raises(ValueError):
            data_profile({"A": {}}, dims={}, tau=1.5)


class TestRuntimeECDF:
    def test_hand_computed_ecdf(self):
        histories = {
            "A": {"p": [[1, 5.0], [3, 0.5], [10, 0.05]]},
            "B": {"p": [[1, 0.5]]},
        }
        budgets, ecdf = runtime_ecdf(histories, f_opt={"p": 0.0}, targets=[1.0, 0.1])
        np.testing.assert_array_equal(budgets, [1.0, 3.0, 10.0])
        np.testing.assert_allclose(ecdf["A"], [0.0, 0.5, 1.0])
        np.testing.assert_allclose(ecdf["B"], [0.5, 0.5, 0.5])

    def test_dimension_scaling(self):
        histories = {"A": {"p": [[1, 5.0], [10, 0.0]]}}
        budgets, _ = runtime_ecdf(histories, f_opt={"p": 0.0}, targets=[0.5], dims={"p": 5})
        np.testing.assert_array_equal(budgets, [2.0])

    def test_accepts_data_profile_history_format(self):
        histories = {"A": {"p": (5.0, [[1, 5.0], [2, 0.0]])}}
        _, ecdf = runtime_ecdf(histories, f_opt={"p": 0.0}, targets=[1.0])
        assert ecdf["A"][-1] == 1.0

    def test_coco_targets(self):
        assert COCO_TARGETS.size == 51
        assert COCO_TARGETS[0] == pytest.approx(100.0)
        assert COCO_TARGETS[-1] == pytest.approx(1e-8)
        np.testing.assert_allclose(COCO_TARGETS[:-1] / COCO_TARGETS[1:], 10**0.2)


class TestRunnerAdapters:
    @pytest.fixture
    def runners(self):
        fast = BenchmarkRunner(
            _shrinking_search,
            "Fast",
            n_runs=2,
            seed=0,
            max_evals=400,
            record_trace=True,
            verbose=False,
            show_progress=False,
        )
        slow = BenchmarkRunner(
            _random_search,
            "Slow",
            n_runs=2,
            seed=0,
            max_evals=400,
            record_trace=True,
            verbose=False,
            show_progress=False,
        )
        for runner in (fast, slow):
            runner.run_suite(functions=["sphere"], dimensions={"sphere": 2})
        return [fast, slow]

    def test_histories_from_runners(self, runners):
        histories, f_opt, dims = histories_from_runners(runners)
        assert set(histories) == {"Fast", "Slow"}
        assert set(histories["Fast"]) == {"sphere#1", "sphere#2"}
        assert f_opt["sphere#1"] == 0.0
        assert dims["sphere#1"] == 2

    def test_untraced_runner_raises(self):
        runner = BenchmarkRunner(_random_search, "NoTrace", max_evals=10, verbose=False)
        runner.run_suite(functions=["sphere"])
        with pytest.raises(ValueError, match="record_trace"):
            histories_from_runners([runner])

    def test_ert_ranks_the_faster_algorithm_first(self, runners):
        ert = ert_from_runners(runners, target=1e-2)
        assert ert["Fast"]["sphere"] < ert["Slow"]["sphere"]

    def test_performance_profile_from_runners(self, runners):
        _, rho = performance_profile_from_runners(runners, target=1e-2)
        assert rho["Fast"][0] == 1.0

    def test_end_to_end_data_profile_and_ecdf(self, runners):
        histories, f_opt, dims = histories_from_runners(runners)
        _, d = data_profile(histories, dims)
        assert set(d) == {"Fast", "Slow"}
        _, ecdf = runtime_ecdf(histories, f_opt, dims=dims)
        assert ecdf["Fast"][-1] >= ecdf["Slow"][-1]


def _fake_runner(name, errors_by_function):
    runner = BenchmarkRunner(lambda f, b: (None, None), algorithm_name=name, verbose=False)
    runner.results = [
        {"function": fn, "success": True, "error": err, "run": 1}
        for fn, err in errors_by_function.items()
    ]
    return runner


class TestStatisticalComparison:
    FUNCTIONS = [f"f{i}" for i in range(8)]

    def _three(self):
        return [
            _fake_runner("A", {fn: 1.0 + i for i, fn in enumerate(self.FUNCTIONS)}),
            _fake_runner("B", {fn: 2.0 + i for i, fn in enumerate(self.FUNCTIONS)}),
            _fake_runner("C", {fn: 3.0 + i for i, fn in enumerate(self.FUNCTIONS)}),
        ]

    def test_holm_correction(self):
        np.testing.assert_allclose(_holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_average_ranks_and_friedman(self):
        pytest.importorskip("scipy")
        result = statistical_comparison(self._three())
        assert result["average_ranks"] == {"A": 1.0, "B": 2.0, "C": 3.0}
        # Perfect agreement across 8 functions: chi2_F = 12*8/(3*4) * (1+4+9 - 3*16/4) = 16
        assert result["friedman"]["statistic"] == pytest.approx(16.0)
        assert result["friedman"]["p_value"] < 0.001
        # CD = 2.343 * sqrt(3 * 4 / (6 * 8)) for k = 3, N = 8 (Demšar 2006, Table 5)
        assert result["critical_difference"] == pytest.approx(2.343 * np.sqrt(12 / 48))

    def test_pairwise_wilcoxon_with_holm(self):
        pytest.importorskip("scipy")
        result = statistical_comparison(self._three())
        a_vs_c = next(p for p in result["pairwise"] if {p["a"], p["b"]} == {"A", "C"})
        # Exact two-sided Wilcoxon p for 8 same-sign differences is 2/256.
        assert a_vs_c["p_value"] == pytest.approx(2 / 256)
        assert a_vs_c["significant"] is True
        assert a_vs_c["better"] == "A"
        assert "Average rank" in result["report"]

    def test_two_algorithms_skip_friedman(self):
        pytest.importorskip("scipy")
        result = statistical_comparison(self._three()[:2])
        assert result["friedman"] is None
        assert len(result["pairwise"]) == 1

    def test_identical_algorithms_not_significant(self):
        pytest.importorskip("scipy")
        errors = {fn: 1.0 for fn in self.FUNCTIONS}
        result = statistical_comparison([_fake_runner("A", errors), _fake_runner("B", errors)])
        assert result["pairwise"][0]["significant"] is False

    def test_failed_function_ranks_worst(self):
        pytest.importorskip("scipy")
        a = _fake_runner("A", {fn: 1.0 for fn in self.FUNCTIONS})
        b = _fake_runner("B", {fn: 1.0 for fn in self.FUNCTIONS})
        b.results[0]["success"] = False
        result = statistical_comparison([a, b])
        assert result["average_ranks"]["B"] > result["average_ranks"]["A"]

    def test_requires_two_runners_with_distinct_names(self):
        pytest.importorskip("scipy")
        with pytest.raises(ValueError):
            statistical_comparison(self._three()[:1])
        with pytest.raises(ValueError):
            statistical_comparison([_fake_runner("A", {"f": 1}), _fake_runner("A", {"f": 2})])

    def test_scipy_unavailable_raises(self, monkeypatch):
        monkeypatch.setattr(analysis, "SCIPY_AVAILABLE", False)
        with pytest.raises(ImportError, match="stats"):
            statistical_comparison(self._three())


def test_importing_bananabench_does_not_import_scipy():
    # scipy.stats takes ~1.5 s to import; loading it eagerly once doubled the import time
    # of the whole package (and of every CLI call). Run in a fresh interpreter, since this
    # session has already imported scipy.
    import subprocess
    import sys

    code = "import sys, bananabench; print('scipy' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
