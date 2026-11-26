"""
Performance analysis for benchmark results.

Implements the standard ways the optimization literature compares solvers
across a set of problems, rather than by one aggregate number:

- **Performance profiles** (Dolan & Moré 2002): for each solver, the fraction of
  problems it solves within a factor ``tau`` of the best solver's cost.
- **Data profiles** (Moré & Wild 2009): for each solver, the fraction of problems
  it solves within a budget of ``alpha`` simplex gradients, i.e. ``alpha * (n + 1)``
  function evaluations, under the convergence test
  ``f(x0) - f(x) >= (1 - tau) * (f(x0) - f_L)``.
- **Expected running time** (ERT, called aRT in COCO): evaluations spent across
  all runs before reaching a target, divided by the number of runs that reached it.
- **Runtime ECDFs** (COCO): the fraction of (problem, target) pairs solved within
  each evaluation budget.
- **Statistical comparison** (Demšar 2006): Friedman test with average ranks,
  Iman–Davenport correction and Nemenyi critical difference, plus pairwise
  Wilcoxon signed-rank tests with Holm's step-down correction. This part needs
  SciPy (``pip install 'banana-bench[stats]'``).

The profile functions take plain mappings, so they work on any data, not only
``BenchmarkRunner`` output. The ``*_from_runners`` adapters build those inputs
from runners that used ``record_trace=True``.

References
----------
.. [1] Dolan, E. D., & Moré, J. J. (2002). Benchmarking optimization software
       with performance profiles. Mathematical Programming, 91(2), 201-213.
.. [2] Moré, J. J., & Wild, S. M. (2009). Benchmarking derivative-free
       optimization algorithms. SIAM Journal on Optimization, 20(1), 172-191.
.. [3] Hansen, N., Auger, A., Brockhoff, D., & Tušar, T. (2022). Anytime
       performance assessment in blackbox optimization benchmarking. IEEE
       Transactions on Evolutionary Computation, 26(6), 1293-1305.
       (COCO performance assessment; arXiv:1605.03560.)
.. [4] Auger, A., & Hansen, N. (2005). Performance evaluation of an advanced
       local search evolutionary algorithm. Proc. IEEE CEC 2005, 1777-1784.
       (Expected running time as "success performance".)
.. [5] Demšar, J. (2006). Statistical comparisons of classifiers over multiple
       data sets. Journal of Machine Learning Research, 7, 1-30.
.. [6] Holm, S. (1979). A simple sequentially rejective multiple test
       procedure. Scandinavian Journal of Statistics, 6(2), 65-70.

License: MIT
"""

from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy import stats as _scipy_stats

    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised via monkeypatch in tests
    _scipy_stats = None
    SCIPY_AVAILABLE = False

if TYPE_CHECKING:  # pragma: no cover
    from .benchmarking import BenchmarkRunner

__all__ = [
    "SCIPY_AVAILABLE",
    "COCO_TARGETS",
    "performance_profile",
    "data_profile",
    "evaluations_to_target",
    "expected_running_time",
    "runtime_ecdf",
    "histories_from_runners",
    "ert_from_runners",
    "performance_profile_from_runners",
    "statistical_comparison",
]

#: COCO's default target precisions: 51 values from 10^2 down to 10^-8, evenly
#: spaced on a log scale (ratio 10^0.2 between neighbours).
COCO_TARGETS = 10.0 ** np.linspace(2, -8, 51)

# Two-tailed Nemenyi critical values q_alpha for k = 2..10 classifiers
# (Demšar 2006, Table 5(a)): studentized range statistic divided by sqrt(2).
_NEMENYI_Q = {
    0.05: [1.960, 2.343, 2.569, 2.728, 2.850, 2.949, 3.031, 3.102, 3.164],
    0.10: [1.645, 2.052, 2.291, 2.459, 2.589, 2.693, 2.780, 2.855, 2.920],
}


def _sorted_keys(mappings: Sequence[Mapping[Any, Any]]) -> List[Any]:
    keys: Dict[Any, None] = {}
    for mapping in mappings:
        for key in mapping:
            keys.setdefault(key, None)
    try:
        return sorted(keys)
    except TypeError:
        return list(keys)


# ---------------------------------------------------------------------------
# Performance profiles
# ---------------------------------------------------------------------------


def performance_profile(
    costs: Mapping[str, Mapping[Any, float]],
    taus: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Dolan–Moré performance profiles.

    Parameters
    ----------
    costs : mapping
        ``{solver: {problem: cost}}`` where ``cost`` is a non-negative
        performance measure, lower is better (e.g. evaluations or expected
        running time to reach a target, or CPU time). ``inf``/``NaN``, or a
        problem missing for a solver, means that solver failed on it.
    taus : array_like, optional
        Ratio thresholds (>= 1) at which to evaluate the profiles. Defaults to
        every distinct finite performance ratio, so the step function is exact.

    Returns
    -------
    taus : ndarray
        The thresholds used.
    profiles : dict
        ``{solver: rho}`` where ``rho[i]`` is the fraction of problems whose
        performance ratio ``r_{p,s} = t_{p,s} / min_s t_{p,s}`` is ``<= taus[i]``.
        ``rho(1)`` is the fraction of problems on which the solver is (jointly)
        best; ``rho`` as ``tau -> inf`` is the fraction it solves at all.

    Notes
    -----
    A failure never satisfies any ``tau``, which is equivalent to Dolan & Moré's
    ratio ``r_M`` exceeding every plotted threshold. When the best cost on a
    problem is exactly 0, solvers that also achieve 0 get ratio 1 and every
    other solver ratio ``inf``, since no finite multiple of 0 exceeds 0.
    """
    solvers = list(costs)
    if not solvers:
        raise ValueError("performance_profile needs at least one solver.")
    problems = _sorted_keys([costs[s] for s in solvers])
    if not problems:
        raise ValueError("performance_profile needs at least one problem.")

    t = np.array([[float(costs[s].get(p, np.inf)) for s in solvers] for p in problems], dtype=float)
    t[~np.isfinite(t)] = np.inf
    if np.any(t < 0):
        raise ValueError("Performance measures must be non-negative.")

    best = np.min(t, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = t / best
    ratios[(best == 0) & (t == 0)] = 1.0
    ratios[(best == 0) & (t > 0)] = np.inf
    ratios[~np.isfinite(best[:, 0]), :] = np.inf

    if taus is None:
        finite = ratios[np.isfinite(ratios)]
        taus = np.unique(np.concatenate([[1.0], finite]))
    taus = np.asarray(taus, dtype=float)
    profiles = {
        s: np.array([np.mean(ratios[:, j] <= tau) for tau in taus]) for j, s in enumerate(solvers)
    }
    return taus, profiles


# ---------------------------------------------------------------------------
# Traces: evaluations to target, ERT
# ---------------------------------------------------------------------------


def evaluations_to_target(trace: Any, target: float) -> float:
    """
    Evaluations a run needed before its best-so-far value reached ``target``.

    Parameters
    ----------
    trace : array_like, shape (k, 2)
        Best-so-far trace from :class:`~bananabench.tracking.TrackedFunction`:
        ``[evaluation_index, best_so_far]`` rows.
    target : float
        Function value to reach (``best_so_far <= target``).

    Returns
    -------
    float
        The 1-based evaluation index, or ``inf`` if the run never reached it.
    """
    trace = np.asarray(trace, dtype=float).reshape(-1, 2)
    hit = np.nonzero(trace[:, 1] <= target)[0]
    return float(trace[hit[0], 0]) if hit.size else float("inf")


def expected_running_time(evals_to_target: Sequence[float], total_evals: Sequence[float]) -> float:
    """
    Expected running time (ERT; COCO's "average runtime", aRT).

    ``ERT = (sum of runtimes of successful runs + sum of all evaluations of
    unsuccessful runs) / number of successful runs``. This is the expected
    number of evaluations to reach the target if unsuccessful runs were
    restarted independently (Auger & Hansen 2005; Hansen et al. 2022).

    Parameters
    ----------
    evals_to_target : sequence of float
        Per run, the evaluations needed to reach the target (``inf`` if never
        reached), e.g. from :func:`evaluations_to_target`.
    total_evals : sequence of float
        Per run, the total evaluations the run performed.

    Returns
    -------
    float
        The ERT, or ``inf`` if no run reached the target.
    """
    rt = np.asarray(evals_to_target, dtype=float)
    total = np.asarray(total_evals, dtype=float)
    if rt.shape != total.shape:
        raise ValueError("evals_to_target and total_evals must have the same length.")
    success = np.isfinite(rt)
    n_success = int(np.sum(success))
    if n_success == 0:
        return float("inf")
    return float((np.sum(rt[success]) + np.sum(total[~success])) / n_success)


# ---------------------------------------------------------------------------
# Data profiles
# ---------------------------------------------------------------------------


def data_profile(
    histories: Mapping[str, Mapping[Any, Tuple[float, Any]]],
    dims: Mapping[Any, int],
    tau: float = 1e-3,
    f_lowest: Optional[Mapping[Any, float]] = None,
    alphas: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Moré–Wild data profiles.

    Parameters
    ----------
    histories : mapping
        ``{solver: {problem: (f_x0, trace)}}`` with ``f_x0`` the value at the
        starting point and ``trace`` a best-so-far trace (see
        :func:`evaluations_to_target`).
    dims : mapping
        ``{problem: n}``, the dimension of each problem.
    tau : float, default=1e-3
        Accuracy level of the convergence test. Moré & Wild report
        ``tau = 10^-1, 10^-3, 10^-5, 10^-7``.
    f_lowest : mapping, optional
        ``{problem: f_L}``. Defaults to Moré & Wild's choice: the smallest value
        any solver reached on that problem within its budget. Passing the known
        global minimum instead gives a stricter, absolute test.
    alphas : array_like, optional
        Budgets in units of simplex gradients (``n + 1`` evaluations). Defaults
        to every distinct finite ``t_{p,s} / (n_p + 1)``, so the step function
        is exact.

    Returns
    -------
    alphas : ndarray
    profiles : dict
        ``{solver: d}`` where ``d[i]`` is the fraction of problems solved within
        ``alphas[i] * (n_p + 1)`` evaluations.

    Notes
    -----
    Moré & Wild start every solver from the same ``x0``. When solvers choose
    their own starting points (as most benchmark algorithms do), pass each
    run's own first evaluation as ``f_x0``, as :func:`histories_from_runners`
    does. The convergence test is then relative to each run's own start.
    """
    if not 0 < tau < 1:
        raise ValueError("tau must be in (0, 1).")
    solvers = list(histories)
    problems = _sorted_keys([histories[s] for s in solvers])

    if f_lowest is None:
        f_lowest = {}
        for p in problems:
            finals = [
                np.asarray(histories[s][p][1], dtype=float).reshape(-1, 2)[:, 1].min()
                for s in solvers
                if p in histories[s] and np.asarray(histories[s][p][1]).size
            ]
            finite = [v for v in finals if np.isfinite(v)]
            f_lowest[p] = min(finite) if finite else np.nan

    ratios = np.full((len(problems), len(solvers)), np.inf)
    for i, p in enumerate(problems):
        f_l = float(f_lowest[p])
        for j, s in enumerate(solvers):
            if p not in histories[s] or not np.isfinite(f_l):
                continue
            f0, trace = histories[s][p]
            f0 = float(f0)
            if not np.isfinite(f0):
                continue
            # f(x0) - f(x) >= (1 - tau)(f(x0) - f_L)  <=>  f(x) <= f_L + tau (f(x0) - f_L)
            target = f_l + tau * (f0 - f_l)
            t = evaluations_to_target(trace, target)
            ratios[i, j] = t / (int(dims[p]) + 1)

    if alphas is None:
        finite = ratios[np.isfinite(ratios)]
        alphas = np.unique(np.concatenate([[0.0], finite]))
    alphas = np.asarray(alphas, dtype=float)
    profiles = {
        s: np.array([np.mean(ratios[:, j] <= a) for a in alphas]) for j, s in enumerate(solvers)
    }
    return alphas, profiles


# ---------------------------------------------------------------------------
# Runtime ECDF
# ---------------------------------------------------------------------------


def runtime_ecdf(
    histories: Mapping[str, Mapping[Any, Any]],
    f_opt: Mapping[Any, float],
    targets: Optional[Sequence[float]] = None,
    dims: Optional[Mapping[Any, int]] = None,
    budgets: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    COCO-style empirical cumulative distribution of runtimes.

    Every (problem, target) pair contributes one runtime: the evaluations a
    solver needed to reach ``f_opt[problem] + target``. The ECDF at budget
    ``b`` is the fraction of all pairs solved within ``b`` evaluations, with
    unsolved pairs kept in the denominator.

    Parameters
    ----------
    histories : mapping
        ``{solver: {problem: trace}}``, or ``{solver: {problem: (f_x0, trace)}}``
        as for :func:`data_profile` (the start value is ignored).
    f_opt : mapping
        ``{problem: optimal (or reference) value}``.
    targets : sequence of float, optional
        Absolute precisions ``f - f_opt``. Defaults to :data:`COCO_TARGETS`.
    dims : mapping, optional
        ``{problem: n}``. When given, budgets are evaluations divided by
        dimension, COCO's usual x-axis.
    budgets : array_like, optional
        Budgets at which to evaluate the ECDF. Defaults to every distinct
        finite runtime, so the step function is exact.

    Returns
    -------
    budgets : ndarray
    ecdfs : dict
        ``{solver: fraction}`` evaluated at each budget.

    Notes
    -----
    COCO additionally bootstraps "simulated restarts" of unsuccessful runs. This
    function reports the plain ECDF of the runs as executed, which is the
    honest view when a hard evaluation budget was imposed (simulated restarts
    can be too pessimistic then; see Hansen et al. 2022).
    """
    target_values: np.ndarray = (
        COCO_TARGETS if targets is None else np.asarray(targets, dtype=float)
    )
    runtimes: Dict[str, np.ndarray] = {}
    problems = _sorted_keys(list(histories.values()))
    for solver, runs in histories.items():
        values = []
        for p in problems:
            if p not in runs:
                values.extend([np.inf] * len(target_values))
                continue
            entry = runs[p]
            trace = entry[1] if isinstance(entry, tuple) else entry
            scale = float(dims[p]) if dims is not None else 1.0
            values.extend(
                evaluations_to_target(trace, float(f_opt[p]) + float(target)) / scale
                for target in target_values
            )
        runtimes[solver] = np.asarray(values, dtype=float)

    if budgets is None:
        finite = np.concatenate([r[np.isfinite(r)] for r in runtimes.values()])
        budgets = np.unique(finite) if finite.size else np.array([1.0])
    budgets = np.asarray(budgets, dtype=float)
    ecdfs = {s: np.array([np.mean(r <= b) for b in budgets]) for s, r in runtimes.items()}
    return budgets, ecdfs


# ---------------------------------------------------------------------------
# Adapters for BenchmarkRunner results
# ---------------------------------------------------------------------------


def histories_from_runners(
    runners: Sequence["BenchmarkRunner"],
) -> Tuple[Dict[str, Dict[str, Tuple[float, np.ndarray]]], Dict[str, float], Dict[str, int]]:
    """
    Collect traces from runners that used ``record_trace=True``.

    Each (function, run) pair becomes one problem instance, keyed
    ``"<function>#<run>"``, the way COCO treats repeated runs as instances.

    Returns
    -------
    histories : dict
        ``{algorithm_name: {problem: (first_cost, trace)}}``, the input format
        of :func:`data_profile` and :func:`runtime_ecdf`.
    f_opt : dict
        ``{problem: known_minimum}``.
    dims : dict
        ``{problem: dimension}``.

    Raises
    ------
    ValueError
        If a runner has no successful traced runs, or two runners share an
        algorithm name.
    """
    histories: Dict[str, Dict[str, Tuple[float, np.ndarray]]] = {}
    f_opt: Dict[str, float] = {}
    dims: Dict[str, int] = {}
    for runner in runners:
        name = runner.algorithm_name
        if name in histories:
            raise ValueError(f"Two runners share the algorithm name {name!r}.")
        runs: Dict[str, Tuple[float, np.ndarray]] = {}
        for r in runner.results:
            if "trace" not in r:
                continue
            key = f"{r['function']}#{r.get('run', 1)}"
            runs[key] = (
                float(r["first_cost"]) if r.get("first_cost") is not None else np.nan,
                np.asarray(r["trace"], dtype=float).reshape(-1, 2),
            )
            if "known_minimum" in r:
                f_opt[key] = float(r["known_minimum"])
            else:
                from .metadata import get_function_info

                f_opt[key] = float(get_function_info(r["function"])["known_minimum"])
            dims[key] = int(r["dimension"])
        if not runs:
            raise ValueError(
                f"Runner {name!r} has no traced runs; create it with record_trace=True "
                "and call run_suite() first."
            )
        histories[name] = runs
    return histories, f_opt, dims


def ert_from_runners(
    runners: Sequence["BenchmarkRunner"], target: float = 1e-8
) -> Dict[str, Dict[str, float]]:
    """
    Expected running time per algorithm and function.

    Parameters
    ----------
    runners : sequence of BenchmarkRunner
        Runners that used ``record_trace=True``.
    target : float, default=1e-8
        Absolute precision: a run succeeds once ``f - known_minimum <= target``.

    Returns
    -------
    dict
        ``{algorithm_name: {function: ERT}}`` (``inf`` where no run succeeded).
    """
    histories, f_opt, _ = histories_from_runners(runners)
    totals = {
        runner.algorithm_name: {
            f"{r['function']}#{r.get('run', 1)}": float(r.get("n_evals", np.nan))
            for r in runner.results
        }
        for runner in runners
    }
    table: Dict[str, Dict[str, float]] = {}
    for name, runs in histories.items():
        per_function: Dict[str, Tuple[List[float], List[float]]] = {}
        for key, (_, trace) in runs.items():
            function = key.rsplit("#", 1)[0]
            rts, tots = per_function.setdefault(function, ([], []))
            rts.append(evaluations_to_target(trace, f_opt[key] + target))
            tots.append(totals[name][key])
        table[name] = {
            fn: expected_running_time(rts, tots) for fn, (rts, tots) in per_function.items()
        }
    return table


def performance_profile_from_runners(
    runners: Sequence["BenchmarkRunner"],
    target: float = 1e-8,
    taus: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Performance profiles using expected running time to ``target`` as the cost.

    The per-function ERT is the cost measure, the same choice COCO makes when
    it compares aRT ratios. A function no run of an algorithm solved counts as
    a failure for that algorithm.
    """
    return performance_profile(ert_from_runners(runners, target=target), taus=taus)


# ---------------------------------------------------------------------------
# Statistical comparison (SciPy)
# ---------------------------------------------------------------------------


def _check_scipy() -> None:
    if not SCIPY_AVAILABLE:
        raise ImportError(
            "statistical_comparison requires SciPy. Install with: pip install 'banana-bench[stats]'"
        )


def _holm(p_values: Sequence[float]) -> np.ndarray:
    """Holm (1979) step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(p_values, dtype=float)
    m = p.size
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def _per_function_metric(runner: "BenchmarkRunner", metric: str) -> Dict[str, float]:
    by_function: Dict[str, List[float]] = {}
    for r in runner.results:
        by_function.setdefault(r["function"], [])
        if r.get("success") and metric in r and r[metric] is not None:
            by_function[r["function"]].append(float(r[metric]))
    return {fn: (float(np.mean(v)) if v else np.inf) for fn, v in by_function.items()}


def statistical_comparison(
    runners: Sequence["BenchmarkRunner"],
    metric: str = "error",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Compare algorithms across functions with non-parametric tests (Demšar 2006).

    Each algorithm is summarized per function by the mean of ``metric`` over
    its successful runs (a function with no successful run counts as ``inf``,
    i.e. the worst). Functions are the "data sets" of Demšar's procedure.

    - **Friedman test** (three or more algorithms) on per-function ranks, with
      the Iman–Davenport F correction Demšar recommends, average ranks, and the
      Nemenyi critical difference: two algorithms whose average ranks differ
      by more than ``critical_difference`` differ significantly.
    - **Wilcoxon signed-rank tests** for every pair, with Holm's step-down
      correction for multiple comparisons.

    Parameters
    ----------
    runners : sequence of BenchmarkRunner
        At least two runners, benchmarked on the same functions.
    metric : str, default="error"
        Per-run result key to compare, lower is better (e.g. ``"error"``,
        ``"n_evals"``, ``"time"``).
    alpha : float, default=0.05
        Significance level (0.05 or 0.10 for the critical difference).

    Returns
    -------
    dict
        ``algorithms``, ``functions``, ``average_ranks``, ``friedman``
        (statistic, p-value, Iman–Davenport F and p-value; ``None`` for two
        algorithms), ``critical_difference`` (``None`` if unavailable),
        ``pairwise`` (list of dicts with ``a``, ``b``, ``statistic``,
        ``p_value``, ``p_holm``, ``significant``, ``better``), and ``report``,
        a formatted text summary.

    Raises
    ------
    ImportError
        If SciPy is not installed.
    ValueError
        If fewer than two runners are given or they share no function.
    """
    _check_scipy()
    if len(runners) < 2:
        raise ValueError("statistical_comparison needs at least 2 runners.")
    names = [r.algorithm_name for r in runners]
    if len(set(names)) != len(names):
        raise ValueError("Every runner needs a distinct algorithm_name.")

    metrics = [_per_function_metric(r, metric) for r in runners]
    functions = sorted(set.intersection(*(set(m) for m in metrics)))
    if not functions:
        raise ValueError("The runners share no benchmarked function.")
    data = np.array([[m[fn] for m in metrics] for fn in functions], dtype=float)
    n, k = data.shape

    # Ranks per function (1 = best; ties get the average rank, as in Demšar).
    ranks = np.vstack([_scipy_stats.rankdata(row) for row in data])
    average_ranks = dict(zip(names, ranks.mean(axis=0).tolist()))

    friedman = None
    critical_difference = None
    if k >= 3:
        chi2 = 12 * n / (k * (k + 1)) * (np.sum(ranks.mean(axis=0) ** 2) - k * (k + 1) ** 2 / 4)
        p_chi2 = float(_scipy_stats.chi2.sf(chi2, k - 1))
        denom = n * (k - 1) - chi2
        if denom > 0:
            f_stat = (n - 1) * chi2 / denom
            p_f = float(_scipy_stats.f.sf(f_stat, k - 1, (k - 1) * (n - 1)))
        else:  # every function ranks the algorithms identically
            f_stat, p_f = float("inf"), 0.0
        friedman = {
            "statistic": float(chi2),
            "p_value": p_chi2,
            "iman_davenport_f": float(f_stat),
            "iman_davenport_p": p_f,
        }
        q = _NEMENYI_Q.get(alpha)
        if q is not None and k <= len(q) + 1:
            critical_difference = float(q[k - 2] * np.sqrt(k * (k + 1) / (6 * n)))

    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    raw_p: List[float] = []
    pairwise: List[Dict[str, Any]] = []
    for i, j in pairs:
        diff = data[:, i] - data[:, j]
        both_inf = ~np.isfinite(data[:, i]) & ~np.isfinite(data[:, j])
        diff = np.where(both_inf, 0.0, diff)
        if np.all(diff == 0):
            stat, p = float("nan"), 1.0
        else:
            # An inf vs finite comparison has a well-defined sign but no magnitude;
            # signed ranks only need the order, so clip to a large finite value.
            finite_max = np.max(np.abs(diff[np.isfinite(diff)]), initial=1.0)
            diff = np.nan_to_num(diff, posinf=10 * finite_max, neginf=-10 * finite_max)
            res = _scipy_stats.wilcoxon(diff, zero_method="wilcox")
            stat, p = float(res.statistic), float(res.pvalue)
        raw_p.append(p)
        better = names[i] if np.median(diff) < 0 else names[j] if np.median(diff) > 0 else None
        pairwise.append(
            {"a": names[i], "b": names[j], "statistic": stat, "p_value": p, "better": better}
        )
    for entry, adjusted in zip(pairwise, _holm(raw_p)):
        entry["p_holm"] = float(adjusted)
        entry["significant"] = bool(adjusted < alpha)

    lines = [
        "=" * 78,
        f"Statistical comparison on {n} function(s), metric '{metric}' (lower is better)",
        "=" * 78,
        f"{'Algorithm':<30} | {'Average rank':>12}",
        "-" * 78,
    ]
    for name, rank in sorted(average_ranks.items(), key=lambda item: item[1]):
        lines.append(f"{name:<30} | {rank:>12.3f}")
    lines.append("-" * 78)
    if friedman is not None:
        lines.append(
            f"Friedman chi2 = {friedman['statistic']:.3f} (p = {friedman['p_value']:.3g}); "
            f"Iman-Davenport F = {friedman['iman_davenport_f']:.3f} "
            f"(p = {friedman['iman_davenport_p']:.3g})"
        )
        if critical_difference is not None:
            lines.append(f"Nemenyi critical difference (alpha={alpha}): {critical_difference:.3f}")
    else:
        lines.append("Friedman test needs at least 3 algorithms; showing pairwise tests only.")
    lines.append("-" * 78)
    lines.append(f"{'Pair':<40} | {'p (Holm)':>10} | Result")
    for e in pairwise:
        verdict = (
            f"{e['better']} better"
            if e["significant"] and e["better"]
            else "no significant difference"
        )
        lines.append(f"{e['a'] + ' vs ' + e['b']:<40} | {e['p_holm']:>10.3g} | {verdict}")
    lines.append("=" * 78)

    return {
        "algorithms": names,
        "functions": functions,
        "metric": metric,
        "alpha": alpha,
        "average_ranks": average_ranks,
        "friedman": friedman,
        "critical_difference": critical_difference,
        "pairwise": pairwise,
        "report": "\n".join(lines),
    }
