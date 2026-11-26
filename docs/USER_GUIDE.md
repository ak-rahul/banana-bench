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
(`zdt1`-`zdt4`, `zdt6`).

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
is better, 0.0 means every reference point was reached exactly), `hypervolume` (2-objective
problems only, since every ZDT problem is bi-objective), `n_solutions`, `n_nondominated`, and
`time`. If your algorithm raises for a given function, that function's result instead carries
`status='exception'`, `success=False`, and `error_message` -- one failing function doesn't abort
the rest of the suite, so you don't need to wrap `run_suite()` in your own `try`/`except`.

### Working with the Pareto Front Directly

For custom quality metrics, or to plot the true front against your algorithm's approximation:

```python
from bananabench.multiobjective import get_pareto_front, non_dominated_front, igd

true_front = get_pareto_front('zdt1')          # (n_points, 2) array, sorted by f1
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
