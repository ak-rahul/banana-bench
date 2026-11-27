# User Guide

This guide provides detailed instructions on how to use `banana-bench` effectively.

## 🎯 Using Benchmark Metadata

The package provides comprehensive metadata for all 70+ functions, eliminating the need to manually specify bounds and known minima.

```python
from bananabench import BENCHMARK_SUITE, get_function_info, get_bounds
import numpy as np
```

### Accessing Function Information

You can retrieve metadata for any specific function using `get_function_info`:

```python
info = get_function_info('ackley')
func = info['function']
known_min = info['known_minimum']

# Use get_bounds() rather than replicating info['bounds'] by hand — it already
# knows whether a function's bounds are a single value to replicate (like
# ackley's) or one explicit (min, max) per dimension (like branin's).
bounds = get_bounds('ackley')  # 10D by default
bounds_5d = get_bounds('ackley', dim=5)
```

### Metadata Helper Functions

| Function | Description |
|----------|-------------|
| `BENCHMARK_SUITE` | Dictionary with all 70+ functions and metadata |
| `get_all_functions()` | Returns list of all function names |
| `get_function_info(name)` | Returns metadata for specific function |
| `get_bounds(name, dim=None)` | Returns bounds for given dimension |
| `get_function_list()` | Returns formatted string with all functions |
| `query_functions(...)` | Returns function names matching a combination of `properties` tags |

### Querying by Mathematical Properties

Rather than hand-filtering `BENCHMARK_SUITE` for functions with a given combination of
properties, `query_functions()` does it declaratively:

```python
from bananabench import query_functions

# Every non-separable, multimodal, scalable function
names = query_functions(separable=False, unimodal=False, scalable=True)

# Every convex function, regardless of separability/modality
convex_names = query_functions(convex=True)
```

Each parameter is `True` (require the tag), `False` (require its opposite tag), or `None`
(don't filter on that axis, the default). `scalable=False` means the `scalable` tag is
*absent* (fixed-dimension) — there's no separate "non-scalable" tag in the schema.

## 🔄 Function Wrappers

Test the robustness of your optimizer by modifying benchmark functions.

### Noisy Functions
Simulate noisy objective functions to see if your algorithm gets misled.

```python
from bananabench import ackley, NoisyFunction

# Add Gaussian noise (mean=0, scale=0.1)
noisy_ackley = NoisyFunction(ackley, noise_type='gaussian', scale=0.1, seed=42)

val = noisy_ackley(np.zeros(2)) # Returns ackley(0) + noise
```

### Shifted Functions
Shift the global optimum to a new location to prevent zero-bias overfitting.

```python
from bananabench import sphere, ShiftedFunction

# Move optimum from [0,0] to [2,2]
shifted_sphere = ShiftedFunction(sphere, shift=[2.0, 2.0])
```

### Rotated Functions
Rotate the function landscape to test invariance to coordinate system changes.
`RotatedFunction` itself only needs NumPy — SciPy below is just a convenient way
to build the rotation matrix (install with `pip install banana-bench[dev]`, or
build any orthogonal matrix by hand).

```python
from bananabench import rastrigin, RotatedFunction
from scipy.spatial.transform import Rotation

# Create rotation matrix
matrix = Rotation.from_euler('z', 45, degrees=True).as_matrix()
rotated_rastrigin = RotatedFunction(rastrigin, matrix=matrix)
```

### Composing Wrappers

Since each wrapper is itself just a callable, they stack by nesting — combine noise, a
shifted optimum, and a rotated landscape into one function:

```python
from bananabench import sphere, NoisyFunction, ShiftedFunction, RotatedFunction

shift = [2.0, 3.0]
matrix = [[0, -1], [1, 0]]  # 90-degree rotation

stressed_sphere = NoisyFunction(
    RotatedFunction(ShiftedFunction(sphere, shift=shift), matrix=matrix),
    scale=0.05,
)
```

**Order matters.** `RotatedFunction(ShiftedFunction(f, shift), matrix)` evaluates
`f(matrix @ x - shift)` (shift applied in the *original* coordinate frame, then rotated),
while `ShiftedFunction(RotatedFunction(f, matrix), shift)` evaluates `f(matrix @ (x - shift))`
(shift applied in the *already-rotated* frame) — these are different functions in general, so
pick the nesting order that matches what you're actually modeling. This composability isn't
limited to the bundled suite either: since every wrapper just takes `Callable[[np.ndarray],
float]`, it works identically on your own research function.

---

## 🔬 Systematic Benchmarking

The `BenchmarkRunner` class allows for rigorous testing of optimization algorithms.

### Quick Benchmarking

For a rapid check of your algorithm:

```python
from bananabench.benchmarking import quick_benchmark

def my_optimizer(func, bounds, max_iter=1000):
   # ... your implementation ...
   return best_x, best_cost

results = quick_benchmark(
    my_optimizer,
    function_names=['sphere', 'ackley', 'rastrigin'],
    n_runs=5,
    max_iter=1000
)
```

### Detailed Benchmarking

For comprehensive results, use the `BenchmarkRunner` class.

```python
from bananabench.benchmarking import BenchmarkRunner

runner = BenchmarkRunner(
    algorithm=my_optimizer,
    algorithm_name='MyOptimizer',
    n_runs=10,       # 10 independent runs per function
    seed=42,         # For reproducibility
    verbose=True,    # Show progress
    show_progress=True, # Progress bars
    n_jobs=4         # Parallel execution
)
```
**Parallel Execution**: Set `n_jobs > 1` to use multiple CPU cores (e.g., `n_jobs=-1` uses all available cores). This significantly speeds up benchmarking when running many repetitions or expensive functions.

```python
# Run on specific functions
results = runner.run_suite(
    functions=['sphere', 'ackley', 'rastrigin', 'rosenbrock', 'griewank'],
    max_iter=2000
)

# Save results
runner.save_results('results.csv')
```

### Fair Benchmarking with Evaluation Budgets

Two optimizers are only comparable at equal cost. Give every run the same evaluation budget with
`max_evals`: a fixed count, or a function of the problem dimension, which is the usual convention
in COCO and in Moré & Wild's data profiles.

```python
runner = BenchmarkRunner(
    my_optimizer,
    algorithm_name='MyOptimizer',
    n_runs=10,
    max_evals=lambda dim: 1000 * dim,  # 1000 evaluations per dimension
)
results = runner.run_suite(functions=['sphere', 'rastrigin'])
```

Your algorithm doesn't need to know about the budget. It receives a
`bananabench.tracking.TrackedFunction` that counts every call. When the budget runs out, that
function raises `BudgetExhausted`, and the runner scores the best point the run had evaluated
(`status='budget_exhausted'`, still counted as a success). Every run reports `n_evals`, so
`compare_algorithms` can show evaluations next to error.

**Result verification.** By default (`verify=True`) the runner re-evaluates the function at your
algorithm's reported `best_x` and scores that value. This re-evaluation isn't counted against the
budget. The algorithm's own figure is kept as `reported_cost`, and `cost_mismatch=True` marks a run
whose report didn't match what its point actually achieves. A buggy optimizer can't post a perfect
score it didn't earn. Pass `verify=False` to score reported costs unchecked. Results are checked
either way: a `best_x` of the wrong length or with NaN values is an `invalid_result`, and a
`best_x` outside the search bounds gets `out_of_bounds=True`. `MOBenchmarkRunner` verifies too,
re-evaluating every returned point in `X`.

`save_results(..., format='json')` writes strict JSON: NaN and infinite values become `null`, so
any JSON parser can read the file.

`TrackedFunction` also works on its own, around any function:

```python
from bananabench import TrackedFunction, sphere

f = TrackedFunction(sphere, max_evals=500)
# ... hand f to any optimizer, e.g. scipy.optimize.minimize(f, x0) ...
print(f.n_evals, f.best_f, f.best_x)
print(f.trace)  # [[evaluation, best_so_far], ...] at every improvement
```

See `examples/11_budgeted_benchmarking.py`.

### Performance and Data Profiles

A single mean error hides most of what matters: how fast each solver gets there, how reliably, and
on which problems. `bananabench.analysis` implements the standard tools for comparing solvers
across a problem set. They need each run's best-so-far trace, so create runners with
`record_trace=True`.

```python
from bananabench import BenchmarkRunner
from bananabench.analysis import (
    data_profile, ert_from_runners, histories_from_runners,
    performance_profile_from_runners, runtime_ecdf,
)

runners = []
for algorithm, name in [(algo_a, 'A'), (algo_b, 'B')]:
    runner = BenchmarkRunner(algorithm, name, n_runs=5, max_evals=lambda d: 1000 * d,
                             record_trace=True, verbose=False)
    runner.run_suite(functions=['sphere', 'rosenbrock', 'ackley', 'rastrigin'])
    runners.append(runner)

# Expected running time to reach f - f_opt <= 1e-6 (COCO's aRT)
ert = ert_from_runners(runners, target=1e-6)          # {algorithm: {function: ERT}}

# Dolan-More performance profiles, using ERT as the cost
taus, perf = performance_profile_from_runners(runners, target=1e-6)

# More-Wild data profiles and a COCO-style runtime ECDF
histories, f_opt, dims = histories_from_runners(runners)
alphas, data = data_profile(histories, dims, tau=1e-3)
budgets, ecdf = runtime_ecdf(histories, f_opt, dims=dims)
```

How to read them:
- **Performance profile.** `perf[name][i]` is the fraction of problems on which a solver is within
  a factor `taus[i]` of the best solver. Its value at `tau = 1` is how often it wins; its value as
  `tau` grows is how often it succeeds at all (Dolan & Moré 2002).
- **Data profile.** `data[name][i]` is the fraction of problems solved within `alphas[i]` simplex
  gradients, i.e. `alphas[i] * (n + 1)` evaluations. A problem counts as solved under the
  convergence test `f(x0) - f(x) >= (1 - tau)(f(x0) - f_L)` (Moré & Wild 2009).
- **Runtime ECDF.** `ecdf[name][i]` is the fraction of (problem, target) pairs reached within
  `budgets[i]` evaluations per dimension. It uses COCO's 51 targets from 1e2 to 1e-8 by default.

All the profile functions take plain mappings, so they also work on results from outside
`bananabench`. Plot them with `plot_performance_profile`, `plot_data_profile` and
`plot_runtime_ecdf` (see `VISUALIZATION.md`), and see `examples/12_performance_profiles.py`.

### Statistical Comparison

To know whether one algorithm is *significantly* better across functions rather than just
lucky, use Demšar's (2006) procedure:
- average ranks;
- the Friedman test with the Iman–Davenport correction;
- the Nemenyi critical difference;
- pairwise Wilcoxon signed-rank tests with Holm's correction.

It needs SciPy (`pip install 'banana-bench[stats]'`).

```python
from bananabench.analysis import statistical_comparison

result = statistical_comparison(runners, metric='error', alpha=0.05)
print(result['report'])
result['average_ranks']        # {algorithm: average rank across functions, 1 = best}
result['pairwise']             # per-pair p-values, Holm-adjusted p-values, and which is better
```

Each algorithm is summarized per function by its mean `metric` over successful runs. A function
with no successful run ranks it last. Use `metric='n_evals'` or `metric='time'` to compare cost
instead of quality.

### Testing for Center Bias

Most classic benchmark functions (sphere, Rastrigin, Ackley, Griewank, ...) have their optimum at
the origin, the exact center of a symmetric box. An algorithm that drifts toward the center, by
design or through a flawed operator, looks excellent on them, and the advantage disappears on
real problems. Kudela (2022) showed published methods doing this, and his follow-up found 47 of 90
methods center-biased. Before trusting good results on these functions, check:

```python
from bananabench.analysis import center_bias_test

result = center_bias_test(my_optimizer, n_runs=20, max_evals=lambda d: 1000 * d, seed=0)
print(result['report'])
result['center_biased']        # True when the geometric-mean ratio is above 10
result['ratios']               # {function: shifted mean error / unshifted mean error}
```

The test runs your algorithm on each function and again with the optimum moved by 10% of the
range, keeping the same bounds. An unbiased algorithm does about as well either way, so its ratios
stay near 1. The default functions include `schwefel2_26`, whose optimum isn't at the center, as a
control: its ratio should stay near 1 even for a biased algorithm. To make your own comparisons
robust to this, also benchmark on shifted functions (`ShiftedFunction`, or `cec.CECFunction` with a
`shift_vector`). See `examples/14_center_bias.py`.

## 🧩 Advanced Usage

### Integration with Scipy
You can easily benchmark `scipy.optimize` algorithms using a wrapper.

```python
from scipy.optimize import minimize
from bananabench import BenchmarkRunner

def scipy_wrapper(func, bounds, max_iter=100):
    # Scipy requires initial guess
    x0 = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds])
    
    res = minimize(
        func, 
        x0, 
        bounds=bounds, 
        method='L-BFGS-B', 
        options={'maxiter': max_iter}
    )
    return res.x, res.fun

runner = BenchmarkRunner(scipy_wrapper, "Scipy-LBFGS")
runner.run_suite(['sphere', 'rosenbrock'])
```

### Creating Custom Wrappers
You can create your own wrappers by inheriting from `BenchmarkWrapper`.

```python
from bananabench.wrappers import BenchmarkWrapper

class DiscreteFunction(BenchmarkWrapper):
    """Rounds input to nearest integer before evaluation."""
    
    def __call__(self, x):
        x_discrete = np.round(x)
        return self.func(x_discrete)

# Usage
from bananabench import sphere
discrete_sphere = DiscreteFunction(sphere)
print(discrete_sphere([1.2, 2.8])) # Evaluates sphere([1.0, 3.0])
```

### Comparing Multiple Algorithms

```python
from bananabench import compare_algorithms

algorithms = {
    'SimulatedAnnealing': simulated_annealing,
    'GeneticAlgorithm': genetic_algorithm
}

runners = []
for name, algo in algorithms.items():
    runner = BenchmarkRunner(algo, algorithm_name=name)
    runner.run_suite()
    runner.save_results(f'{name}_results.csv')  # still useful for raw per-run data
    runners.append(runner)

# One readable side-by-side table instead of eyeballing separate CSVs
print(compare_algorithms(runners))
```

`compare_algorithms()` ranks runners by mean error (best first); an algorithm with zero
successful runs always sorts last rather than raising an error.

### Ranking Constrained Results

`g_suite`/`engineering` functions return `(objective, inequality_violations, equality_violations)`
rather than a single float, so "lower objective wins" isn't a safe ranking rule on its own -- an
infeasible point with a great objective value is not actually a good solution.
`bananabench.constraints` implements the standard feasibility rule (Deb, 2000) that most published
results on these suites already rely on:

```python
from bananabench import g_suite
from bananabench.constraints import feasibility_compare, is_feasible

candidate_a = g_suite.g01(x_a)  # (objective, inequality_violations, equality_violations)
candidate_b = g_suite.g01(x_b)

print(is_feasible(candidate_a[1], candidate_a[2]))  # True/False

# -1 if candidate_a is preferred, 1 if candidate_b is preferred, 0 on an exact tie
winner = feasibility_compare(candidate_a, candidate_b)
```

Feasible solutions always beat infeasible ones; between two feasible solutions the lower objective
wins; between two infeasible solutions the smaller total constraint violation wins.

### Integration with Optuna

Any `BENCHMARK_SUITE` function plugs directly into Optuna's `objective(trial) -> float` interface --
see `examples/10_optuna_integration.py` for a full script. The short version:

```python
import optuna
from bananabench import metadata
from bananabench.utils import normalize_bounds

info = metadata.get_function_info('rastrigin')
bounds = normalize_bounds(info['bounds'], dim=2)

def objective(trial):
    x = [trial.suggest_float(f'x{i}', lo, hi) for i, (lo, hi) in enumerate(bounds)]
    return info['function'](x)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=200)
```

Requires Optuna (`pip install optuna`); it's not a `bananabench` dependency.

---

## 🔀 Multi-Objective Benchmarking

`BenchmarkRunner` assumes a single scalar cost, which doesn't fit multi-objective problems --
`MOBenchmarkRunner` is the counterpart for those, run against the bundled ZDT suite
(`zdt1`-`zdt4`, `zdt6`, bi-objective) and DTLZ suite (`dtlz1`-`dtlz7`, any number of objectives).

A multi-objective algorithm has a different signature than a single-objective one: it takes the
number of objectives and returns a whole population of candidate solutions, not a single best
point.

```python
from bananabench import MOBenchmarkRunner
import numpy as np

def random_search_mo(func, bounds, n_objectives, n_points=100, seed=None):
    # X: (n_points, dim) decision vectors: F: (n_points, n_objectives) objective values
    rng = np.random.default_rng(seed)
    bounds_array = np.array(bounds)
    lo, hi = bounds_array[:, 0], bounds_array[:, 1]
    X = lo + rng.random((n_points, len(bounds))) * (hi - lo)
    F = np.array([func(x) for x in X])
    return X, F

runner = MOBenchmarkRunner(random_search_mo, algorithm_name='RandomSearch', seed=42)
results = runner.run_suite(functions=['zdt1', 'zdt2'], n_points=200)

for r in results:
    print(f"{r['function']}: IGD={r['igd']:.4f}, hypervolume={r.get('hypervolume', 'N/A')}")
```

Each result dict carries `igd` (Inverted Generational Distance to the true Pareto front -- lower
is better, 0.0 means every reference point was reached exactly), `hypervolume`, `n_objectives`,
`n_solutions`, `n_nondominated`, `n_evals`, and `time`. If your algorithm raises for a given function, that function's result instead carries
`status='exception'`, `success=False`, and `error_message` -- one failing function doesn't abort
the rest of the suite, so you don't need to wrap `run_suite()` in your own `try`/`except`.

As with `BenchmarkRunner`, `runner.save_results('mo_results.csv')` (or `format='json'`) saves
every result, and `runner.get_summary_stats()` summarizes IGD, time and evaluations over the
successful runs.

### Many-Objective Problems (DTLZ)

DTLZ problems take the number of objectives as a parameter. Set it on the runner. ZDT problems stay
bi-objective, and asking for more raises `ValueError`.

```python
runner = MOBenchmarkRunner(random_search_mo, algorithm_name='RandomSearch',
                           n_objectives=5, max_evals=10_000)
results = runner.run_suite(functions=['dtlz1', 'dtlz2', 'dtlz7'])
```

Each DTLZ problem has `n = M + k - 1` variables, with the paper's `k` (5 for DTLZ1, 10 for
DTLZ2–6, 20 for DTLZ7), and is called as `dtlz2(x, n_objectives=M)`. With `max_evals`, a run that
reaches the budget is scored on the non-dominated set of everything it evaluated.
`hypervolume` works for any number of objectives:
- exact by default (WFG beyond three objectives);
- `method='monte_carlo'` gives a seeded estimate for large fronts.

`das_dennis(M, H)` gives uniformly spaced reference points on the simplex. See
`examples/13_many_objective_dtlz.py`.

### Working with the Pareto Front Directly

For custom quality metrics, or to plot the true front against your algorithm's approximation:

```python
from bananabench.multiobjective import get_pareto_front, non_dominated_front, igd

true_front = get_pareto_front('zdt1')          # (n_points, 2) array, sorted by f1
true_3d = get_pareto_front('dtlz2', n_objectives=3)  # points on the unit sphere
my_front = non_dominated_front(F)              # filter your algorithm's F down to non-dominated points
quality = igd(my_front, true_front)
```

---

## 🛠️ Utility Functions

The package includes utilities to help with common optimization tasks.

### Bounds Handling

**Normalization**
Replicate bounds for n-dimensions:
```python
from bananabench.utils import normalize_bounds
bounds = normalize_bounds((-5, 5), dim=10) # [(-5, 5), ..., (-5, 5)]
```

**Clipping**
Ensure points stay within search space:
```python
from bananabench.utils import clip_to_bounds
clamped = clip_to_bounds(point, bounds)
```

**Random Initialization**
Generate valid starting points:
```python
from bananabench.utils import generate_random_point
start_point = generate_random_point(bounds, method='uniform')
```

### Coordinate Transformation

Scale problems to a valid unit hypercube $[0, 1]^n$ for standardized processing:

```python
from bananabench.utils import scale_to_unit, scale_from_unit

unit_point = scale_to_unit(real_point, bounds)
real_point = scale_from_unit(unit_point, bounds)
```
