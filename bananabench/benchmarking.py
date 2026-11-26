"""
Benchmarking utilities for optimization algorithms.

This module provides tools for systematically testing optimization algorithms
across multiple benchmark functions, tracking performance metrics, and
generating comprehensive reports.

License: MIT
"""

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from .metadata import BENCHMARK_SUITE, get_all_functions, get_function_info
from .multiobjective import (
    get_mo_bounds,
    get_mo_default_dim,
    get_mo_function,
    get_mo_function_list,
    get_mo_n_objectives,
    get_pareto_front,
    hypervolume,
    igd,
    non_dominated_front,
)
from .tracking import BudgetExhausted, TrackedFunction
from .utils import _format_value, calculate_distance_to_optimum, normalize_bounds

# A budget is either a fixed evaluation count or a function of the problem dimension,
# e.g. ``lambda dim: 10_000 * dim`` (budgets proportional to dimension are the usual
# convention in COCO and in Moré & Wild's data profiles).
Budget = Union[int, Callable[[int], int]]

# Tolerances for BenchmarkRunner's verification of a reported best_cost against a
# fresh evaluation of f(best_x). Deliberately tight: the algorithm sees the exact same
# deterministic function, so a correct report matches to rounding error.
_VERIFY_RTOL = 1e-9
_VERIFY_ATOL = 1e-12


def _resolve_budget(max_evals: Optional[Budget], dim: int) -> Optional[int]:
    """Turn a ``max_evals`` setting into a concrete evaluation count for ``dim``."""
    if max_evals is None:
        return None
    budget = int(max_evals(dim) if callable(max_evals) else max_evals)
    if budget < 1:
        raise ValueError(
            f"max_evals must resolve to a positive integer, got {budget} for dim={dim}."
        )
    return budget


class BenchmarkRunner:
    """
    Systematic benchmarking tool for optimization algorithms.

    This class provides a standardized interface for testing optimization
    algorithms across multiple benchmark functions with automatic result
    collection, statistical analysis, and report generation.

    Parameters
    ----------
    algorithm : callable
        Optimization algorithm with signature: func, bounds, **kwargs -> (best_x, best_cost)
    algorithm_name : str, optional
        Name of the algorithm for reporting
    n_runs : int, default=1
        Number of independent runs per function
    seed : int, optional
        Random seed for reproducibility
    verbose : bool, default=True
        Whether to print progress information
    show_progress : bool, default=True
        Whether to show progress bars (requires tqdm)
    n_jobs : int, default=1
        Number of parallel workers for multiple runs (requires joblib)
    max_evals : int or callable, optional
        Evaluation budget per run: a fixed count, or a function of the problem
        dimension such as ``lambda dim: 10_000 * dim``. The algorithm receives a
        :class:`~bananabench.tracking.TrackedFunction` that raises
        :class:`~bananabench.tracking.BudgetExhausted` instead of exceeding the
        budget. The runner catches it and scores the best point evaluated so far,
        so algorithms don't need to handle the budget themselves. ``None``
        (default) means unlimited.
    record_trace : bool, default=False
        Store each run's best-so-far trace (``trace``, ``first_cost``) for
        :mod:`bananabench.analysis` (data profiles, runtime ECDFs, ERT).
    verify : bool, default=True
        Re-evaluate the benchmark function at the reported ``best_x`` (not
        counted against the budget) and score that value instead of the reported
        ``best_cost``. A run whose report disagrees gets ``cost_mismatch=True``.
        Pass ``False`` to score reported costs as-is.

    Attributes
    ----------
    results : list
        List of result dictionaries, one per run. Every run records
        ``n_evals`` (evaluations the algorithm actually performed) and
        ``max_evals``.

    Examples
    --------
    >>> def my_optimizer(func, bounds, max_iter=1000):
    ...     # Your optimization code
    ...     return best_x, best_cost
    >>>
    >>> runner = BenchmarkRunner(my_optimizer, algorithm_name='MyAlgo', n_runs=10)
    >>> results = runner.run_suite(functions=['sphere', 'ackley'])
    >>> runner.save_results('results.csv')
    """

    def __init__(
        self,
        algorithm: Callable,
        algorithm_name: Optional[str] = None,
        n_runs: int = 1,
        seed: Optional[int] = None,
        verbose: bool = True,
        show_progress: bool = True,
        n_jobs: int = 1,
        max_evals: Optional[Budget] = None,
        record_trace: bool = False,
        verify: bool = True,
    ):
        self.algorithm = algorithm
        self.algorithm_name = algorithm_name or "UnnamedAlgorithm"
        self.n_runs = n_runs
        self.seed = seed
        self.verbose = verbose
        self.show_progress = show_progress and TQDM_AVAILABLE  # Only if tqdm available
        self.n_jobs = n_jobs
        self.max_evals = max_evals
        self.record_trace = record_trace
        self.verify = verify
        self.results = []

        self.rng = np.random.default_rng(seed)
        if seed is not None:
            # Also seed global numpy RNG for algorithms that use np.random directly.
            # Prefer passing seed explicitly to your algorithm when possible.
            np.random.seed(seed)

    def run_single(
        self,
        function_name: str,
        dim: Optional[int] = None,
        run_seed: Optional[int] = None,
        **algorithm_kwargs,
    ) -> Dict[str, Any]:
        """
        Run algorithm on a single function.

        Parameters
        ----------
        function_name : str
            Name of the benchmark function
        dim : int, optional
            Dimension to use (if None, uses default from metadata)
        **algorithm_kwargs
            Additional arguments passed to the algorithm

        Returns
        -------
        dict
            Result dictionary with metrics
        """
        # Get function info
        info = get_function_info(function_name)
        func = info["function"]
        dim = dim or info["default_dim"]
        known_min = info["known_minimum"]
        optimal_point = info["optimal_point"]

        # Normalize bounds
        try:
            bounds = normalize_bounds(info["bounds"], dim)
        except Exception as e:
            return {
                "function": function_name,
                "dimension": dim,
                "status": "bounds_error",
                "error_message": str(e),
                "success": False,
            }

        budget = _resolve_budget(self.max_evals, dim)
        tracked = TrackedFunction(func, max_evals=budget, record_trace=self.record_trace)
        base = {"function": function_name, "dimension": dim, "algorithm": self.algorithm_name}

        def accounting() -> Dict[str, Any]:
            extra: Dict[str, Any] = {"n_evals": tracked.n_evals, "max_evals": budget}
            if self.record_trace:
                extra["first_cost"] = tracked.first_f
                extra["trace"] = tracked.trace.tolist()
            return extra

        def failure(status: str, elapsed: float, **fields: Any) -> Dict[str, Any]:
            return {
                **base,
                "status": status,
                **fields,
                "time": float(elapsed),
                "success": False,
                **accounting(),
            }

        # Run algorithm
        if run_seed is not None:
            np.random.seed(run_seed)
        start_time = time.time()
        budget_exhausted = False
        try:
            try:
                best_x, best_cost = self.algorithm(tracked, bounds, **algorithm_kwargs)
            except BudgetExhausted:
                # Not a failure: the run is scored on the best point it evaluated.
                budget_exhausted = True
                best_x = tracked.best_x
                best_cost = tracked.best_f if tracked.best_x is not None else None
        except Exception as e:
            return failure("exception", time.time() - start_time, error_message=str(e))
        elapsed = time.time() - start_time

        # Validate the reported result
        if best_x is None or best_cost is None:
            return failure("failed", elapsed)
        reported_cost = float(best_cost)
        if not np.isfinite(reported_cost):
            return failure("invalid_result", elapsed, reported_cost=reported_cost)

        # Verify the report against a fresh evaluation of the raw function
        # (not counted against the budget).
        cost_mismatch: Optional[bool] = None
        scored_cost = reported_cost
        if self.verify:
            try:
                verified_cost = float(func(np.asarray(best_x, dtype=float)))
            except Exception as e:
                return failure(
                    "exception",
                    elapsed,
                    error_message=f"evaluating the reported best_x failed: {e}",
                )
            if not np.isfinite(verified_cost):
                return failure(
                    "invalid_result", elapsed, reported_cost=reported_cost, cost_mismatch=True
                )
            cost_mismatch = not bool(
                np.isclose(reported_cost, verified_cost, rtol=_VERIFY_RTOL, atol=_VERIFY_ATOL)
            )
            scored_cost = verified_cost

        # Calculate metrics
        distance = (
            calculate_distance_to_optimum(best_x, optimal_point)
            if optimal_point is not None
            else None
        )
        result = {
            **base,
            "best_cost": float(scored_cost),
            "reported_cost": reported_cost,
            "cost_mismatch": cost_mismatch,
            "known_minimum": float(known_min),
            "error": float(abs(scored_cost - known_min)),
            "distance_to_optimum": float(distance) if distance is not None else None,
            "time": float(elapsed),
            "status": "budget_exhausted" if budget_exhausted else "success",
            "success": True,
            **accounting(),
        }
        return result

    def run_suite(
        self,
        functions: Optional[List[str]] = None,
        dimensions: Optional[Dict[str, int]] = None,
        **algorithm_kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Run algorithm across multiple functions.

        Parameters
        ----------
        functions : list of str, optional
            List of function names to test (if None, uses all functions)
        dimensions : dict, optional
            Custom dimensions per function {function_name: dim}
        **algorithm_kwargs
            Additional arguments passed to the algorithm

        Returns
        -------
        list of dict
            List of result dictionaries
        """
        if functions is None:
            functions = get_all_functions()

        dimensions = dimensions or {}

        if self.verbose:
            print("=" * 102)
            print(f"BENCHMARK: {self.algorithm_name}")
            print("=" * 102)
            budget_note = "" if self.max_evals is None else ", evaluation budget enforced"
            print(f"Testing {len(functions)} functions with {self.n_runs} run(s) each{budget_note}")
            print("=" * 102)
            print(
                f"{'Function':<25} | {'Dim':>3} | {'Best':>12} | {'Known':>12} | "
                f"{'Error':>12} | {'Evals':>9} | {'Time':>7}"
            )
            print("-" * 102)

        all_results = []

        if self.show_progress:
            func_iterator = tqdm(functions, desc="Benchmarking", unit="func")
        else:
            func_iterator = functions

        for func_name in func_iterator:
            dim = dimensions.get(func_name, None)

            # Update progress bar description
            if self.show_progress:
                func_iterator.set_description(f"Testing {func_name:<15}")

            # Multiple runs

            # PARALLEL OR SERIAL EXECUTION
            if self.n_jobs != 1 and self.n_runs > 1:
                # Parallel execution using joblib
                try:
                    from joblib import Parallel, delayed

                    # Generate distinct seeds for each run
                    seeds: List[Optional[int]] = [None] * self.n_runs
                    if hasattr(self, "rng") and self.rng is not None:
                        seeds = [int(s) for s in self.rng.integers(0, 2**31 - 1, size=self.n_runs)]

                    # Disable internal tqdm for parallel runs to avoid mess
                    parallel_runs = Parallel(n_jobs=self.n_jobs)(
                        delayed(self.run_single)(
                            func_name, dim, run_seed=seeds[i], **algorithm_kwargs
                        )
                        for i in range(self.n_runs)
                    )

                    # Add run numbers post-hoc
                    run_results = []
                    for i, result in enumerate(parallel_runs):
                        result["run"] = i + 1
                        run_results.append(result)

                except ImportError:
                    if self.verbose:
                        print("Warning: joblib not installed. Falling back to serial execution.")
                    # Fallback to serial
                    run_results = self._run_serial(func_name, dim, **algorithm_kwargs)
            else:
                # Serial execution
                run_results = self._run_serial(func_name, dim, **algorithm_kwargs)

            # Aggregate stats across runs
            successful_runs = [r for r in run_results if r["success"]]

            if successful_runs:
                errors = [r["error"] for r in successful_runs]
                times = [r["time"] for r in successful_runs]

                # Best run
                best_run = min(successful_runs, key=lambda x: x["error"])

                # Aggregate result
                agg_result = {
                    "function": func_name,
                    "dimension": best_run["dimension"],
                    "algorithm": self.algorithm_name,
                    "n_runs": self.n_runs,
                    "n_successful": len(successful_runs),
                    "success_rate": len(successful_runs) / self.n_runs,
                    "best_cost": best_run["best_cost"],
                    "known_minimum": best_run["known_minimum"],
                    "error_mean": float(np.mean(errors)),
                    "error_std": float(np.std(errors)),
                    "error_min": float(np.min(errors)),
                    "error_max": float(np.max(errors)),
                    "time_mean": float(np.mean(times)),
                    "time_std": float(np.std(times)),
                    "n_evals_mean": float(np.mean([r["n_evals"] for r in successful_runs])),
                    "n_budget_exhausted": sum(
                        1 for r in successful_runs if r["status"] == "budget_exhausted"
                    ),
                    "n_cost_mismatch": sum(1 for r in successful_runs if r.get("cost_mismatch")),
                }

                if self.verbose:
                    # ASCII only: this prints to whatever stream the caller has, and a
                    # non-ASCII marker crashes with UnicodeEncodeError on legacy
                    # codepages (e.g. Windows cp1252 when stdout is redirected).
                    marker = "*" if agg_result["error_mean"] < 1.0 else " "
                    print(
                        f"{func_name:<25} | {agg_result['dimension']:3d} | "
                        f"{_format_value(agg_result['best_cost'], 12, 6)} | "
                        f"{_format_value(agg_result['known_minimum'], 12, 6)} | "
                        f"{_format_value(agg_result['error_mean'], 12, 6)} | "
                        f"{agg_result['n_evals_mean']:>9.0f} | "
                        f"{agg_result['time_mean']:6.2f}s {marker}"
                    )
            else:
                agg_result = {
                    "function": func_name,
                    "algorithm": self.algorithm_name,
                    "n_runs": self.n_runs,
                    "n_successful": 0,
                    "success_rate": 0.0,
                    "status": "all_runs_failed",
                }

                if self.verbose:
                    print(
                        f"{func_name:<25} | {'?':>3} | {'FAILED':>12} | "
                        f"{'N/A':>12} | {'N/A':>12} | {'N/A':>9} | {'N/A':>7}"
                    )

            all_results.append(agg_result)
            self.results.extend(run_results)  # Store individual run results

        if self.verbose:
            self.print_summary(all_results)

        return all_results

    def _run_serial(self, func_name, dim, **algorithm_kwargs):
        """Helper for serial execution of runs."""
        run_results = []

        run_iterator = range(self.n_runs)

        # Generate distinct seeds for each run
        seeds: List[Optional[int]] = [None] * self.n_runs
        if hasattr(self, "rng") and self.rng is not None:
            seeds = [int(s) for s in self.rng.integers(0, 2**31 - 1, size=self.n_runs)]

        for run in run_iterator:
            result = self.run_single(func_name, dim, run_seed=seeds[run], **algorithm_kwargs)
            result["run"] = run + 1
            run_results.append(result)
        return run_results

    def print_summary(self, results: List[Dict[str, Any]]):
        """Print summary statistics."""
        print("=" * 102)
        print("SUMMARY")
        print("=" * 102)

        successful = [r for r in results if r.get("n_successful", 0) > 0]

        if successful:
            total_functions = len(results)
            n_successful = len(successful)
            errors = [r["error_mean"] for r in successful]
            times = [r["time_mean"] for r in successful]

            # Convergence metrics
            converged_1 = sum(1 for e in errors if e < 1.0)
            converged_01 = sum(1 for e in errors if e < 0.1)
            converged_001 = sum(1 for e in errors if e < 0.01)

            print(f"Total functions tested: {total_functions}")
            print(
                f"Successful runs: {n_successful}/{total_functions} "
                f"({n_successful / total_functions * 100:.1f}%)"
            )
            print(
                f"Converged (error < 1.0): {converged_1}/{n_successful} "
                f"({converged_1 / n_successful * 100:.1f}%)"
            )
            print(
                f"Converged (error < 0.1): {converged_01}/{n_successful} "
                f"({converged_01 / n_successful * 100:.1f}%)"
            )
            print(
                f"Converged (error < 0.01): {converged_001}/{n_successful} "
                f"({converged_001 / n_successful * 100:.1f}%)"
            )

            print(f"\nError statistics:")
            print(f"  Mean: {np.mean(errors):.6f}")
            print(f"  Median: {np.median(errors):.6f}")
            print(f"  Std: {np.std(errors):.6f}")
            print(f"  Min: {np.min(errors):.6f}")
            print(f"  Max: {np.max(errors):.6f}")

            print(f"\nTime statistics:")
            print(f"  Total: {np.sum(times):.2f}s")
            print(f"  Mean: {np.mean(times):.2f}s")
            print(f"  Median: {np.median(times):.2f}s")

            # Best and worst
            best_5 = sorted(successful, key=lambda x: x["error_mean"])[:5]
            worst_5 = sorted(successful, key=lambda x: x["error_mean"])[-5:]

            print(f"\nTop 5 best results:")
            for r in best_5:
                print(f"  {r['function']:<25} | Error: {_format_value(r['error_mean'], 12, 6)}")

            print(f"\nTop 5 worst results:")
            for r in worst_5:
                print(f"  {r['function']:<25} | Error: {_format_value(r['error_mean'], 12, 6)}")

            exhausted = sum(r.get("n_budget_exhausted", 0) for r in successful)
            if exhausted:
                print(
                    f"\n{exhausted} run(s) stopped at the evaluation budget (scored on best-so-far)."
                )
            mismatched = sum(r.get("n_cost_mismatch", 0) for r in successful)
            if mismatched:
                print(
                    f"\nWarning: {mismatched} run(s) reported a best_cost that does not match "
                    "f(best_x); errors above use the verified value."
                )

        print("=" * 102)

    def save_results(self, filepath: Union[str, Path], format: str = "csv"):
        """
        Save results to file.

        Parameters
        ----------
        filepath : str or Path
            Output file path
        format : str, default='csv'
            Output format ('csv' or 'json')
        """
        filepath = Path(filepath)

        if format == "csv":
            with open(filepath, "w", newline="") as f:
                if self.results:
                    # A run mixing successes and failures produces heterogeneous result
                    # dicts (run_single's exception/bounds_error/invalid_result branches
                    # return fewer keys than a successful run's), so fieldnames must be
                    # the union across every result, not just the first one's keys --
                    # DictWriter raises ValueError on any row with a key outside
                    # fieldnames, and a fieldnames list based on a single result can't
                    # anticipate keys introduced only by a different result's status.
                    # Traces (record_trace=True) are nested arrays and don't fit a CSV
                    # cell; they're kept in the JSON output only.
                    fieldnames: List[str] = []
                    seen = {"trace"}
                    for result in self.results:
                        for key in result:
                            if key not in seen:
                                seen.add(key)
                                fieldnames.append(key)
                    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(self.results)
        elif format == "json":
            with open(filepath, "w") as f:
                json.dump(self.results, f, indent=2)
        else:
            raise ValueError(f"Unknown format: {format}. Use 'csv' or 'json'")

        if self.verbose:
            print(f"\nResults saved to: {filepath}")

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics.

        Returns
        -------
        dict
            Summary statistics
        """
        successful = [r for r in self.results if r.get("success", False)]

        if not successful:
            return {"n_results": len(self.results), "n_successful": 0}

        errors = [r["error"] for r in successful]
        times = [r["time"] for r in successful]

        return {
            "algorithm": self.algorithm_name,
            "n_results": len(self.results),
            "n_successful": len(successful),
            "success_rate": len(successful) / len(self.results),
            "error_mean": float(np.mean(errors)),
            "error_median": float(np.median(errors)),
            "error_std": float(np.std(errors)),
            "error_min": float(np.min(errors)),
            "error_max": float(np.max(errors)),
            "time_total": float(np.sum(times)),
            "time_mean": float(np.mean(times)),
            "time_median": float(np.median(times)),
            "n_evals_mean": float(np.mean([r.get("n_evals", np.nan) for r in successful])),
            "n_cost_mismatch": sum(1 for r in successful if r.get("cost_mismatch")),
        }


class _ObjectiveArchive:
    """Counts evaluations of a vector-valued objective, enforces a budget, and keeps
    every objective vector it returned, so a run stopped by its budget can still be
    scored on the non-dominated set of everything it evaluated."""

    def __init__(self, func: Callable[[np.ndarray], np.ndarray], max_evals: Optional[int]):
        self.func = func
        self.max_evals = max_evals
        self.n_evals = 0
        self._objectives: List[np.ndarray] = []

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.max_evals is not None and self.n_evals >= self.max_evals:
            raise BudgetExhausted(self.max_evals)
        value = np.asarray(self.func(x), dtype=float)
        self.n_evals += 1
        self._objectives.append(value.copy())
        return value

    @property
    def objectives(self) -> np.ndarray:
        return np.asarray(self._objectives, dtype=float)


class MOBenchmarkRunner:
    """
    Systematic benchmarking tool for multi-objective optimization algorithms,
    run against ``bananabench.multiobjective.MO_SUITE`` (ZDT and DTLZ).

    Unlike ``BenchmarkRunner`` (which assumes a scalar cost and a single
    ``known_minimum``), a multi-objective algorithm returns a whole
    approximation of the Pareto front, so quality is measured against the
    true front via IGD and hypervolume (any number of objectives; see
    ``multiobjective.hypervolume``).

    Parameters
    ----------
    algorithm : callable
        Multi-objective optimizer with signature
        ``algorithm(func, bounds, n_objectives, **kwargs) -> (X, F)``, where
        ``X`` is an ``(n_points, dim)`` array of decision vectors and ``F``
        is the corresponding ``(n_points, n_objectives)`` array of objective
        values.
    algorithm_name : str, optional
        Name of the algorithm for reporting.
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool, default=True
        Whether to print progress information.
    n_objectives : int, optional
        Number of objectives for DTLZ problems (default 3). ZDT problems are
        always bi-objective.
    max_evals : int or callable, optional
        Evaluation budget per run, a count or a function of the dimension, as
        for ``BenchmarkRunner``. A run that hits the budget is scored on the
        non-dominated set of every objective vector it evaluated
        (``status="budget_exhausted"``).

    Attributes
    ----------
    results : list
        List of result dictionaries for each run, each with ``n_evals``.

    Examples
    --------
    >>> def my_mo_algo(func, bounds, n_objectives, pop_size=100):
    ...     # Your multi-objective optimization code
    ...     return X, F
    >>>
    >>> runner = MOBenchmarkRunner(my_mo_algo, algorithm_name='MyMOAlgo')
    >>> results = runner.run_suite(functions=['zdt1', 'dtlz2'])
    """

    def __init__(
        self,
        algorithm: Callable,
        algorithm_name: Optional[str] = None,
        seed: Optional[int] = None,
        verbose: bool = True,
        n_objectives: Optional[int] = None,
        max_evals: Optional[Budget] = None,
    ):
        self.algorithm = algorithm
        self.algorithm_name = algorithm_name or "UnnamedMOAlgorithm"
        self.seed = seed
        self.verbose = verbose
        self.n_objectives = n_objectives
        self.max_evals = max_evals
        self.results: List[Dict[str, Any]] = []

    def run_single(
        self,
        function_name: str,
        dim: Optional[int] = None,
        reference_point: Optional[np.ndarray] = None,
        **algorithm_kwargs,
    ) -> Dict[str, Any]:
        """
        Run the algorithm on a single multi-objective problem.

        Parameters
        ----------
        function_name : str
            Name of a problem in ``multiobjective.MO_SUITE``.
        dim : int, optional
            Dimension to use (if None, uses the problem's default: 30 or 10 for
            ZDT, ``M + k - 1`` for DTLZ).
        reference_point : ndarray, optional
            Reference point for the hypervolume indicator (if None, uses
            1.1x the true front's per-objective max, the common convention).
        **algorithm_kwargs
            Additional arguments passed to the algorithm.

        Returns
        -------
        dict
            Result dictionary with IGD, hypervolume, evaluations and timing.
        """
        n_objectives = get_mo_n_objectives(function_name, self.n_objectives)
        func = get_mo_function(function_name, n_objectives)
        dim = dim or get_mo_default_dim(function_name, n_objectives)
        bounds = get_mo_bounds(function_name, dim)
        archive = _ObjectiveArchive(func, _resolve_budget(self.max_evals, dim))
        base = {
            "function": function_name,
            "dimension": dim,
            "n_objectives": n_objectives,
            "algorithm": self.algorithm_name,
        }

        if self.seed is not None:
            np.random.seed(self.seed)

        # Isolate one function's failure from the rest of the suite, mirroring
        # BenchmarkRunner.run_single -- without this, run_suite (which calls
        # run_single once per function with no try/except of its own) would let a
        # single bad algorithm call take down the entire batch instead of just
        # that one function's result.
        start_time = time.time()
        status = "success"
        try:
            try:
                _, F = self.algorithm(archive, bounds, n_objectives, **algorithm_kwargs)
            except BudgetExhausted:
                status = "budget_exhausted"
                F = archive.objectives
            elapsed = time.time() - start_time
        except Exception as e:
            return {
                **base,
                "status": "exception",
                "error_message": str(e),
                "time": float(time.time() - start_time),
                "success": False,
                "n_evals": archive.n_evals,
            }

        F = np.asarray(F, dtype=float).reshape(-1, n_objectives)
        if F.shape[0] == 0:
            return {
                **base,
                "status": "failed",
                "time": float(elapsed),
                "success": False,
                "n_evals": archive.n_evals,
            }
        front = non_dominated_front(F)
        true_front = get_pareto_front(function_name, n_objectives=n_objectives)
        ref = (
            np.asarray(reference_point, dtype=float)
            if reference_point is not None
            else np.max(true_front, axis=0) * 1.1
        )

        return {
            **base,
            "n_solutions": int(F.shape[0]),
            "n_nondominated": int(front.shape[0]),
            "igd": igd(front, true_front),
            "hypervolume": hypervolume(front, ref, seed=self.seed),
            "n_evals": archive.n_evals,
            "max_evals": archive.max_evals,
            "time": float(elapsed),
            "status": status,
            "success": True,
        }

    def run_suite(
        self,
        functions: Optional[List[str]] = None,
        dimensions: Optional[Dict[str, int]] = None,
        **algorithm_kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Run the algorithm across multiple multi-objective problems.

        Parameters
        ----------
        functions : list of str, optional
            Problem names to test (if None, uses every problem in MO_SUITE).
        dimensions : dict, optional
            Custom dimensions per function {function_name: dim}.
        **algorithm_kwargs
            Additional arguments passed to the algorithm.

        Returns
        -------
        list of dict
            List of result dictionaries.
        """
        if functions is None:
            functions = get_mo_function_list()
        dimensions = dimensions or {}

        if self.verbose:
            print("=" * 82)
            print(f"MULTI-OBJECTIVE BENCHMARK: {self.algorithm_name}")
            print("=" * 82)
            print(
                f"{'Function':<10} | {'Dim':>4} | {'M':>2} | {'IGD':>12} | "
                f"{'Hypervolume':>12} | {'Evals':>8} | {'Time':>7}"
            )
            print("-" * 82)

        all_results = []
        for function_name in functions:
            dim = dimensions.get(function_name, None)
            result = self.run_single(function_name, dim=dim, **algorithm_kwargs)
            all_results.append(result)
            self.results.append(result)

            if self.verbose:
                prefix = (
                    f"{result['function']:<10} | {result['dimension']:4d} | "
                    f"{result['n_objectives']:2d} | "
                )
                if result.get("success", True) is False:
                    print(
                        f"{prefix}{'FAILED':>12} | {'N/A':>12} | "
                        f"{result['n_evals']:>8d} | {result['time']:6.2f}s"
                    )
                else:
                    print(
                        f"{prefix}{_format_value(result['igd'], 12, 6)} | "
                        f"{_format_value(result['hypervolume'], 12, 6)} | "
                        f"{result['n_evals']:>8d} | {result['time']:6.2f}s"
                    )

        if self.verbose:
            print("=" * 82)

        return all_results


def quick_benchmark(
    algorithm: Callable,
    function_names: Optional[List[str]] = None,
    n_runs: int = 1,
    show_progress: bool = True,
    max_evals: Optional[Budget] = None,
    verify: bool = True,
    **algorithm_kwargs,
) -> List[Dict[str, Any]]:
    """
    Quick benchmark helper function.

    Parameters
    ----------
    algorithm : callable
        Optimization algorithm
    function_names : list of str, optional
        Functions to test (if None, tests common subset)
    n_runs : int, default=1
        Number of runs per function
    show_progress : bool, default=True
        Show progress bars
    max_evals : int or callable, optional
        Evaluation budget per run (see ``BenchmarkRunner``)
    verify : bool, default=True
        Score the re-evaluated f(best_x) instead of the reported cost (see
        ``BenchmarkRunner``)
    **algorithm_kwargs
        Additional arguments for the algorithm

    Returns
    -------
    list of dict
        Benchmark results

    Examples
    --------
    >>> def my_algo(func, bounds, max_iter=1000):
    ...     # optimization code
    ...     return best_x, best_cost
    >>> results = quick_benchmark(my_algo, function_names=['sphere', 'ackley'])
    """
    if function_names is None:
        # Default subset of common functions
        function_names = [
            "sphere",
            "ackley",
            "rastrigin",
            "rosenbrock",
            "griewank",
            "beale",
            "booth",
            "matyas",
            "himmelblau",
            "easom",
        ]

    runner = BenchmarkRunner(
        algorithm,
        algorithm_name="QuickBenchmark",
        n_runs=n_runs,
        verbose=True,
        show_progress=show_progress,
        max_evals=max_evals,
        verify=verify,
    )
    results = runner.run_suite(functions=function_names, **algorithm_kwargs)
    return results


def compare_algorithms(runners: List["BenchmarkRunner"]) -> str:
    """
    Build a side-by-side comparison table across multiple benchmarked algorithms.

    Each runner must have already had ``run_suite()`` called at least once, so
    ``get_summary_stats()`` has results to summarize. This is the direct
    replacement for looping over runners and saving separate result files by
    hand (see USER_GUIDE.md's "Comparing Multiple Algorithms") -- it produces
    one readable table instead.

    Parameters
    ----------
    runners : list of BenchmarkRunner
        At least two runners, each already benchmarked via ``run_suite()``.

    Returns
    -------
    str
        A formatted comparison table, one row per algorithm, ranked by mean
        error ascending (best first). An algorithm with zero successful runs
        has no error to rank by, so it always sorts last.

    Raises
    ------
    ValueError
        If fewer than two runners are given -- there's nothing to compare
        with only one.

    Examples
    --------
    >>> runner_a = BenchmarkRunner(algo_a, algorithm_name='AlgoA')
    >>> runner_a.run_suite(functions=['sphere', 'ackley'])
    >>> runner_b = BenchmarkRunner(algo_b, algorithm_name='AlgoB')
    >>> runner_b.run_suite(functions=['sphere', 'ackley'])
    >>> print(compare_algorithms([runner_a, runner_b]))
    """
    if len(runners) < 2:
        raise ValueError("compare_algorithms needs at least 2 runners to compare.")

    pairs = [(runner.algorithm_name, runner.get_summary_stats()) for runner in runners]
    pairs.sort(
        key=lambda pair: (
            pair[1].get("n_successful", 0) == 0,
            pair[1].get("error_mean", float("inf")),
        )
    )

    lines = [
        "=" * 102,
        "Algorithm Comparison",
        "=" * 102,
        f"{'Algorithm':<25} | {'Success Rate':>12} | {'Error Mean':>12} | "
        f"{'Error Median':>13} | {'Evals Mean':>10} | {'Time Mean':>10}",
        "-" * 102,
    ]
    for name, stats in pairs:
        if stats.get("n_successful", 0) == 0:
            lines.append(f"{name:<25} | {'no successful runs':^69}")
            continue
        lines.append(
            f"{name:<25} | {stats['success_rate'] * 100:>11.1f}% | "
            f"{stats['error_mean']:>12.6g} | {stats['error_median']:>13.6g} | "
            f"{stats['n_evals_mean']:>10.0f} | {stats['time_mean']:>9.3f}s"
        )
    lines.append("=" * 102)

    return "\n".join(lines)
