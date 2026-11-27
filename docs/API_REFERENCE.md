# API Reference

This document provides detailed API documentation for the core components of `banana-bench`.

## Core Components

### `BenchmarkRunner`

The `BenchmarkRunner` class is the main engine for systematically testing optimization algorithms.

```python
from bananabench import BenchmarkRunner
```

#### Constructor

```python
BenchmarkRunner(
    algorithm: Callable,
    algorithm_name: Optional[str] = None,
    n_runs: int = 1,
    seed: Optional[int] = None,
    verbose: bool = True,
    show_progress: bool = True,
    n_jobs: int = 1,
    max_evals: Optional[Union[int, Callable[[int], int]]] = None,
    record_trace: bool = False,
    verify: bool = True,
)
```

**Parameters:**
- `algorithm`: Callable with signature `func(f, bounds, **kwargs) -> (best_x, best_cost)`. `f` is a
  `TrackedFunction` (see `tracking` below) that counts evaluations and enforces `max_evals`.
- `algorithm_name`: String name for the report.
- `n_runs`: Number of independent runs per function (default: 1).
- `seed`: Random seed for reproducibility.
- `verbose`: Print progress to stdout.
- `show_progress`: Show tqdm progress bars (default: True).
- `n_jobs`: Number of parallel jobs.
  - `None` or `1`: Serial execution.
  - `-1`: Use all available CPU cores.
  - `int > 1`: Use specific number of cores.
- `max_evals`: Evaluation budget per run, a fixed count or a function of the dimension
  (`lambda d: 10_000 * d`). When the algorithm reaches it, the runner catches `BudgetExhausted` and
  scores the best point evaluated so far (`status="budget_exhausted"`, still a success), so
  algorithms don't need to handle the budget. `None` (default) means unlimited.
- `record_trace`: Store each run's best-so-far `trace` (`[[evaluation, best], ...]`) and
  `first_cost`, which `analysis` needs for data profiles, ERT and runtime ECDFs.
- `verify`: Re-evaluate the function at the reported `best_x` (not counted against the budget) and
  score that value. The algorithm's own figure is kept as `reported_cost`, and `cost_mismatch` is
  `True` when the two disagree. `verify=False` scores reported costs unchecked (the pre-0.4.0
  behavior).

**Per-run result keys** (`runner.results`): `function`, `dimension`, `algorithm`, `run`, `status`
(`success`, `budget_exhausted`, `failed`, `invalid_result`, `exception`, `bounds_error`),
`success`, `best_cost`, `reported_cost`, `cost_mismatch`, `known_minimum`, `error`,
`distance_to_optimum`, `n_evals`, `max_evals`, `time`, and with `record_trace`, `trace` and
`first_cost`. Failed runs carry only the keys that apply.

#### Methods

**`run_suite(functions=None, dimensions=None, **kwargs)`**

Run the benchmark suite on selected functions.

- `functions`: List of function names (e.g., `['sphere', 'ackley']`). If None, runs all 70+ functions.
- `dimensions`: Dictionary mapping function names to custom dimensions (e.g., `{'sphere': 10}`).
- `**kwargs`: Additional arguments passed to your optimization algorithm.

**Returns:** A list of per-function aggregate dictionaries (`error_mean`/`_std`/`_min`/`_max`,
`time_mean`, `n_evals_mean`, `n_budget_exhausted`, `n_cost_mismatch`, `success_rate`, ...).

**`save_results(filepath, format='csv')`**

Save benchmark results to a file.
- `format`: 'csv' or 'json'.

A run mixing successes and failures saves correctly under either format — a failed run's result
dict has fewer keys than a successful one's, and the CSV writer accounts for that (blank cells for
keys a given row doesn't have) rather than requiring every row to share one key set. Traces
(`record_trace=True`) are kept in JSON output only, since they don't fit a CSV cell.

**`get_summary_stats()`**

Returns a dictionary containing aggregate statistics (mean error, success rate, etc.).

### `Wrappers`

Wrappers modify the behavior of benchmark functions. They follow the decorator pattern and are callable like normal functions.

**`NoisyFunction(func, noise_type='gaussian', scale=0.1, seed=None)`**
- `func`: The benchmark function to wrap.
- `noise_type`: 'gaussian' (normal distribution) or 'uniform'.
- `scale`: Standard deviation (for gaussian) or half-width (for uniform).
- `seed`: Integer seed for reproducible noise. Uses a local random number generator.

**`ShiftedFunction(func, shift)`**
- `func`: The benchmark function to wrap.
- `shift`: List or array of shift values. Must match function dimension.
  - New global minimum location: $x^* + \text{shift}$

**`RotatedFunction(func, matrix)`**
- `func`: The benchmark function to wrap.
- `matrix`: An orthogonal rotation matrix ($n \times n$).
  - Evaluates $f(M \cdot x)$.

**Composing wrappers**: since each one just takes `Callable[[np.ndarray], float]` (including
another wrapper instance), they stack by nesting —
`NoisyFunction(RotatedFunction(ShiftedFunction(f, shift), matrix), scale=0.05)`. Nesting order
changes the result (a shift applied before vs. after a rotation are different transforms); see
`USER_GUIDE.md`'s "Composing Wrappers" for the exact semantics.

### `MOBenchmarkRunner`

The multi-objective counterpart to `BenchmarkRunner`, run against
`bananabench.multiobjective.MO_SUITE` (ZDT and DTLZ). A multi-objective algorithm returns a whole
Pareto-front approximation rather than a single best cost, so quality is measured against the true
front via IGD and hypervolume (any number of objectives).

```python
from bananabench import MOBenchmarkRunner
```

#### Constructor

```python
MOBenchmarkRunner(
    algorithm: Callable,
    algorithm_name: Optional[str] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
    n_objectives: Optional[int] = None,
    max_evals: Optional[Union[int, Callable[[int], int]]] = None,
)
```

**Parameters:**
- `algorithm`: Callable with signature `func(f, bounds, n_objectives, **kwargs) -> (X, F)`, where
  `X` is an `(n_points, dim)` array of decision vectors and `F` is the corresponding
  `(n_points, n_objectives)` array of objective values.
- `algorithm_name`: String name for the report.
- `seed`: Random seed for reproducibility.
- `verbose`: Print progress to stdout.
- `n_objectives`: Number of objectives M for DTLZ problems (default 3). ZDT is always bi-objective;
  passing another value for a ZDT problem raises `ValueError`.
- `max_evals`: Evaluation budget per run, as for `BenchmarkRunner`. A run that reaches it is scored
  on the non-dominated set of every objective vector it evaluated (`status="budget_exhausted"`).

#### Methods

**`run_suite(functions=None, dimensions=None, **kwargs)`**

Run the algorithm across multiple multi-objective problems.

- `functions`: List of function names (e.g., `['zdt1', 'dtlz2']`). If None, runs every problem in
  `MO_SUITE`.
- `dimensions`: Dictionary mapping function names to custom dimensions.
- `**kwargs`: Additional arguments passed to your optimization algorithm.

**Returns:** A list of result dictionaries. On success: `status` (`"success"` or
`"budget_exhausted"`), `success=True`, `n_objectives`, `igd`, `hypervolume`, `n_solutions`,
`n_nondominated`, `n_evals`, `max_evals`, and `time`. If your
algorithm raises for a given function, that function's dict instead carries `status="exception"`,
`success=False`, and `error_message` — one failing function doesn't abort the rest of the suite.

**`run_single(function_name, dim=None, reference_point=None, **kwargs)`**

Run the algorithm on a single problem. `reference_point` overrides the default hypervolume
reference point (1.1x the true front's per-objective max). Isolates an exception from your
algorithm into a failure result dict rather than raising, matching `BenchmarkRunner.run_single`.

**`save_results(filepath, format='csv')`**

Save every result so far as `'csv'` or `'json'`, as `BenchmarkRunner.save_results` does. The CSV
header is the union of keys across all results, so a suite mixing successes and failures still
saves.

**`get_summary_stats()`**

Summary over successful runs: `n_results`, `n_successful`, `success_rate`, `igd_mean`,
`igd_median`, `igd_std`, `igd_min`, `igd_max`, `time_total`, `time_mean`, `n_evals_mean`,
`n_budget_exhausted`. There's no mean hypervolume: each problem has its own reference point, so
hypervolumes from different problems aren't comparable. Compare them per problem instead.

### Extended Modules

These modules are imported eagerly (`bananabench.analysis`, `.cec`, `.constraints`,
`.engineering`, `.g_suite`, `.gradients`, `.multiobjective`, `.plotly_viz`, `.tracking`). All but
two need nothing beyond NumPy. `plotly_viz` degrades gracefully if `plotly` is not installed
(`pip install 'banana-bench[interactive]'`), and `analysis.statistical_comparison` needs SciPy
(`pip install 'banana-bench[stats]'`).

**`tracking` — evaluation counting and budgets**
- `TrackedFunction(func, max_evals=None, record_trace=True)`: wraps any objective and returns
  exactly `func(x)`. Attributes: `n_evals`, `best_f`, `best_x` (a copy), `first_f`, `budget_left`,
  and `trace`, an `(k, 2)` array of `[evaluation_index, best_so_far]` holding the first evaluation
  and every strict improvement. NaN/inf values count as evaluations but never as improvements.
  `reset()` clears it. Picklable, and composes with `wrappers`/`cec`.
- `BudgetExhausted`: raised instead of performing evaluation `max_evals + 1`.

**`analysis` — comparing solvers across problems**
- `performance_profile(costs, taus=None) -> (taus, {solver: rho})`: Dolan & Moré (2002).
  `costs` is `{solver: {problem: cost}}`, lower is better; `inf`/missing means failure.
- `data_profile(histories, dims, tau=1e-3, f_lowest=None, alphas=None) -> (alphas, {solver: d})`:
  Moré & Wild (2009). `histories` is `{solver: {problem: (f_x0, trace)}}`; the convergence test is
  `f(x0) - f(x) >= (1 - tau)(f(x0) - f_L)`, and budgets are in simplex gradients (`n + 1`
  evaluations).
- `expected_running_time(evals_to_target, total_evals)`: COCO's aRT. All evaluations of
  unsuccessful runs plus the runtimes of successful runs, divided by the number of successes.
- `runtime_ecdf(histories, f_opt, targets=None, dims=None, budgets=None)`: fraction of
  (problem, target) pairs solved within each budget. Defaults to `COCO_TARGETS` (51 targets from
  1e2 to 1e-8), with evaluations per dimension when `dims` is given.
- `evaluations_to_target(trace, target)`: first evaluation whose best-so-far is `<= target`
  (`inf` if never).
- Adapters for `BenchmarkRunner(record_trace=True)` output: `histories_from_runners(runners) ->
  (histories, f_opt, dims)` (each function/run pair is one problem instance),
  `ert_from_runners(runners, target=1e-8)`, and `performance_profile_from_runners(runners,
  target=1e-8)`.
- `statistical_comparison(runners, metric="error", alpha=0.05) -> dict`: Demšar (2006).
  - Average ranks, the Friedman test with the Iman–Davenport correction, and the Nemenyi critical
    difference (3 or more algorithms).
  - Pairwise Wilcoxon signed-rank tests with Holm's correction.
  - Returns structured results plus a text `report`.

**`cec` — CEC-style composition framework**
- `CECFunction(base_func, shift_vector=None, rotation_matrix=None, bias=0.0)`: wraps any
  callable — the bundled suite or your own function, `base_func` isn't restricted to
  `bananabench`'s functions — with an optional shift, rotation, and additive bias, evaluated as
  `base_func(rotation @ (x - shift)) + bias`. See `examples/09_cec_style_composition.py` for
  wrapping/composing a user-defined function this way.
- `CECCompositionFunction(functions, sigma, biases)`: combines several `CECFunction` instances
  into a single weighted composition, a hallmark of the CEC 2017/2020 benchmark suites.

**`g_suite` — constrained G-function suite**
- `g01(x)` ... `g24(x)`: the complete classic G-suite (all 24 problems) from the 2006 CEC Special
  Session on Constrained Real-Parameter Optimization. Each returns
  `(objective, inequality_violations, equality_violations)` as `(float, np.ndarray, np.ndarray)`.
  `g20`'s documented "best known" point is itself acknowledged as slightly infeasible in the
  source report — treat its `known_minimum` as indicative, not a strict target.
- `G_SUITE`: registry dict keyed by function name, with `function`, `dim`, `n_inequality`,
  `n_equality`, and `known_minimum` — a lighter-weight counterpart to `ZDT_SUITE`/
  `BENCHMARK_SUITE` (no `bounds`/`optimal_point`/`properties`, since those vary in shape per
  problem in ways that don't fit a single schema cleanly; see each function's own docstring).
- `get_g_function_list()`: registry accessor, analogous to `get_mo_function_list()`.

**`engineering` — classic engineering design suite**
- `spring_design(x)`, `pressure_vessel(x)`, `welded_beam(x)`, `gear_train(x)`: four widely-used
  mechanical/structural design benchmarks (tension/compression spring, pressure vessel, welded
  beam, gear train), each from its own original paper rather than a shared suite report — see
  `bananabench.engineering`'s module docstring and each function's docstring for sources. Follows
  `g_suite`'s convention: each returns `(objective, inequality_violations, equality_violations)`
  as `(float, np.ndarray, np.ndarray)`; none of the four has an equality constraint.
  `gear_train` has no constraints at all beyond its box bounds.
- `ENGINEERING_SUITE`: registry dict keyed by function name, same shape as `G_SUITE` (`function`,
  `dim`, `n_inequality`, `n_equality`, `known_minimum`) and for the same reason (bounds vary too
  much per-problem to fit one schema).
- `get_engineering_function_list()`: registry accessor, analogous to `get_g_function_list()`.

**`constraints` — feasibility-comparison helper for the G-suite/engineering suites**
- `feasibility_compare(a, b, tol=1e-6) -> int`: ranks two
  `(objective, inequality_violations, equality_violations)` results (the return shape of every
  `g_suite`/`engineering` function) using Deb (2000)'s feasibility rule — feasible beats infeasible,
  then lower objective among feasible pairs, then smaller total violation among infeasible pairs.
  Returns `-1` if `a` is preferred, `1` if `b` is preferred, `0` on an exact tie. `BenchmarkRunner`
  assumes a scalar cost, so it doesn't fit this tuple shape directly — use this to rank results from
  the constrained suites instead of hand-rolling the comparison.
- `is_feasible(inequality_violations, equality_violations=(), tol=1e-6) -> bool`: `True` iff every
  inequality entry is `<= tol` and every equality entry's absolute value is `<= tol`.
- `constraint_violation(inequality_violations, equality_violations=(), tol=1e-6) -> float`: total
  violation magnitude — the sum of each violated inequality's positive excess plus each violated
  equality constraint's absolute deviation from zero. `0.0` exactly when `is_feasible` is `True`.

**`multiobjective` — ZDT and DTLZ multi-objective suites**
- `zdt1(x)` ... `zdt4(x)`, `zdt6(x)`: the five real-valued ZDT problems (ZDT5 is binary-encoded and
  not included). Each returns a 2-element objective array `[f1, f2]` (both minimized) instead of a
  single float.
- `dtlz1(x, n_objectives=3)` ... `dtlz7(x, n_objectives=3)`: the DTLZ suite (Deb, Thiele, Laumanns
  & Zitzler 2005), for any number of objectives M >= 2, with `n = M + k - 1` variables
  (`k` = 5 for DTLZ1, 10 for DTLZ2–6, 20 for DTLZ7). `dtlz4` also takes `alpha=100`.
- `ZDT_SUITE`, `DTLZ_SUITE`, and `MO_SUITE` (both). Entries have `function`, `n_objectives`,
  `default_dim`, `bounds` (a `dim -> [(min, max), ...]` callable) and `properties`; DTLZ entries also
  have `k` and `optimal_distance_value`.
- `get_mo_function_list()`, `get_mo_bounds(name, dim=None, n_objectives=None)`,
  `get_mo_default_dim(name, n_objectives=None)`, `get_mo_n_objectives(name, n_objectives=None)`,
  `get_mo_function(name, n_objectives=None)`: registry accessors. `get_mo_function` binds M for
  DTLZ.
- `get_pareto_front(name, n_points=200, n_objectives=None)`: a sample of the true Pareto front, for
  use as the reference front in quality indicators.
- `das_dennis(n_objectives, n_partitions)`: Das–Dennis uniform points on the simplex,
  `C(H + M - 1, M - 1)` of them.
- `igd(front, reference_front)`: Inverted Generational Distance between an approximation front and
  the true front (lower is better).
- `hypervolume(front, reference_point, method="auto", n_samples=100_000, seed=None)`: hypervolume
  indicator for any number of objectives (higher is better).
  - `"exact"` uses a sweep for 2 objectives, slicing for 3, and WFG (While, Bradstreet & Barone 2012)
    beyond that.
  - `"monte_carlo"` gives a seeded estimate.
  - `"auto"` is exact except for more than 100 points in 4+ objectives.
- `non_dominated_front(points)`: filters a point set down to its Pareto-non-dominated subset.

**`gradients` — finite-difference gradient estimation**
- `approximate_gradient(func, x, method='central', epsilon=1e-8)`: estimates the gradient of a
  black-box function via `'forward'`, `'backward'`, or `'central'` finite differences.
- `GradientWrapper(func, method='central', epsilon=1e-8)`: equips a black-box function with
  `.gradient(x)` and `.fun_and_grad(x)` methods (useful as a SciPy `jac`), and tracks
  `.evaluations` / `.grad_evaluations` counts.

**`plotly_viz` — interactive Plotly visualizations** (requires `plotly`)
- `plot_surface_interactive(func, bounds, resolution=100, log_scale=False, show=True)`: interactive
  3D surface plot of a 2D function.
- `plot_contour_interactive(func, bounds, resolution=100, log_scale=False, show=True)`: interactive
  2D contour plot.
- Both raise `ImportError` with an install hint if `plotly` is not available.

### Command-line interface

The `banana-bench` entry point (equivalently `python -m bananabench`) supports `--list`, `--info
FUNCTION`, `--metadata FUNCTION`, and `--function FUNCTION --values ...` / `--input FILE.csv
--output results.json`. All of these accept `--suite {scalar,constrained,engineering,multiobjective}`
(default `scalar`) to resolve the function name against `BENCHMARK_SUITE`, `g_suite.G_SUITE`
(`g01`-`g24`), `engineering.ENGINEERING_SUITE` (`spring_design`, `pressure_vessel`, `welded_beam`,
`gear_train`), or `multiobjective.MO_SUITE` (`zdt1`-`zdt4`, `zdt6`, `dtlz1`-`dtlz7`)
respectively. Evaluation output is shaped per suite: `scalar` returns a plain number,
`constrained`/`engineering` return
`{"objective": ..., "inequality_violations": [...], "equality_violations": [...]}`, and
`multiobjective` returns `{"objectives": [f1, ..., fM]}`. `--objectives M` sets the number of
objectives for DTLZ problems (default 3); it's only valid with `--suite multiobjective`, and ZDT
problems reject anything but 2. `--version` prints the installed version.
Run `banana-bench --help` for the full flag reference and examples.

```bash
banana-bench --list --suite constrained
banana-bench --function g01 --suite constrained --values 1 1 1 1 1 1 1 1 1 3 3 3 1
banana-bench --function spring_design --suite engineering --values 0.05 0.25 2.0
banana-bench --function zdt1 --suite multiobjective --values 0.5 0 0
banana-bench --function dtlz2 --suite multiobjective --objectives 3 --values 0.5 0.5 0.5 0.5
```

### `quick_benchmark`

A helper function for rapid testing without creating a class instance.

```python
from bananabench import quick_benchmark

results = quick_benchmark(
    my_optimizer,
    function_names=['sphere', 'ackley'],
    n_runs=5,
    max_evals=lambda d: 1000 * d,  # optional; verify=True by default, as for BenchmarkRunner
    max_iter=100,
)
```

### `compare_algorithms(runners)`

Builds a side-by-side text comparison table across multiple already-benchmarked
`BenchmarkRunner` instances (each must have had `run_suite()` called at least once), ranked by
mean error ascending, with mean evaluations per run alongside error and time. An algorithm with
zero successful runs sorts last rather than raising. Raises `ValueError` if fewer than two runners
are given. For significance tests rather than a table, see `analysis.statistical_comparison`.

```python
from bananabench import BenchmarkRunner, compare_algorithms

runner_a = BenchmarkRunner(algo_a, algorithm_name='AlgoA')
runner_a.run_suite(functions=['sphere', 'ackley'])
runner_b = BenchmarkRunner(algo_b, algorithm_name='AlgoB')
runner_b.run_suite(functions=['sphere', 'ackley'])

print(compare_algorithms([runner_a, runner_b]))
```

## Utility Functions

### Bounds & Geometry

**`normalize_bounds(bounds, dim)`**
Converts various bound formats into a standard list of `(min, max)` tuples: a single `(min, max)`
replicated across `dim` dimensions, or an explicit per-dimension list of length `dim`. Raises
`ValueError` for anything else, rather than guessing — e.g. an explicit bounds list whose length
doesn't match `dim`.

**`check_bounds(point, bounds)`**
Returns `True` if point is within bounds.

**`clip_to_bounds(point, bounds)`**
Constrains a point to lie within the specified bounds.

**`generate_random_point(bounds, method='uniform')`**
Generates a random point within bounds. Methods: `'uniform'`, `'normal'`, `'center_biased'`.

### Metadata Access

**`get_all_functions()`**
Returns a list of all available function names.

**`get_function_info(name)`**
Returns a dictionary with keys: `function`, `properties`, `bounds`, `default_dim`, `known_minimum`, `optimal_point`.
`properties` is a list of tags such as `continuous`/`discontinuous`, `differentiable`/`non-differentiable`,
`separable`/`non-separable`, `scalable`, and `unimodal`/`multimodal`/`convex`/`non-convex` where established.

**`query_functions(separable=None, scalable=None, differentiable=None, continuous=None, unimodal=None, convex=None)`**
Finds function names by combining `properties` tags declaratively instead of hand-filtering
`BENCHMARK_SUITE` yourself — e.g. `query_functions(separable=False, unimodal=False, scalable=True)`
returns every non-separable, multimodal, scalable function. Each parameter is `True`/`False`/`None`
(don't filter on that axis); a function carrying neither tag on a requested axis (modality/convexity
aren't always asserted — see `get_function_info`) matches neither `True` nor `False` for it. `scalable`
has no opposite tag in the schema, so `scalable=False` means the tag's *absence* (fixed-dimension).
Scoped to the scalar suite only — `g_suite`/`engineering`/`multiobjective` don't carry this schema.

## Type Hints

The package is fully typed. You can import types for static analysis:

```python
from typing import Callable, List, Tuple
import numpy as np

OptimizerType = Callable[
    [Callable[[np.ndarray], float], List[Tuple[float, float]]],
    Tuple[np.ndarray, float],
]
```
