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
)
```

**Parameters:**
- `algorithm`: Callable with signature `func(f, bounds, **kwargs) -> (best_x, best_cost)`.
- `algorithm_name`: String name for the report.
- `n_runs`: Number of independent runs per function (default: 1).
- `seed`: Random seed for reproducibility.
- `verbose`: Print progress to stdout.
- `show_progress`: Show tqdm progress bars (default: True).
- `n_jobs`: Number of parallel jobs.
  - `None` or `1`: Serial execution.
  - `-1`: Use all available CPU cores.
  - `int > 1`: Use specific number of cores.

#### Methods

**`run_suite(functions=None, dimensions=None, **kwargs)`**

Run the benchmark suite on selected functions.

- `functions`: List of function names (e.g., `['sphere', 'ackley']`). If None, runs all 55+ functions.
- `dimensions`: Dictionary mapping function names to custom dimensions (e.g., `{'sphere': 10}`).
- `**kwargs`: Additional arguments passed to your optimization algorithm.

**Returns:** A list of result dictionaries.

**`save_results(filepath, format='csv')`**

Save benchmark results to a file.
- `format`: 'csv' or 'json'.

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

### `MOBenchmarkRunner`

The multi-objective counterpart to `BenchmarkRunner`, run against `bananabench.multiobjective.ZDT_SUITE`.
A multi-objective algorithm returns a whole Pareto-front approximation rather than a single best
cost, so quality is measured against the true front via IGD (always) and hypervolume (2-objective
problems only).

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
)
```

**Parameters:**
- `algorithm`: Callable with signature `func(f, bounds, n_objectives, **kwargs) -> (X, F)`, where
  `X` is an `(n_points, dim)` array of decision vectors and `F` is the corresponding
  `(n_points, n_objectives)` array of objective values.
- `algorithm_name`: String name for the report.
- `seed`: Random seed for reproducibility.
- `verbose`: Print progress to stdout.

#### Methods

**`run_suite(functions=None, dimensions=None, **kwargs)`**

Run the algorithm across multiple ZDT problems.

- `functions`: List of function names (e.g., `['zdt1', 'zdt2']`). If None, runs every problem in
  `ZDT_SUITE`.
- `dimensions`: Dictionary mapping function names to custom dimensions.
- `**kwargs`: Additional arguments passed to your optimization algorithm.

**Returns:** A list of result dictionaries, each with `igd`, `hypervolume` (2-objective problems
only), `n_solutions`, `n_nondominated`, and `time`.

**`run_single(function_name, dim=None, reference_point=None, **kwargs)`**

Run the algorithm on a single ZDT problem. `reference_point` overrides the default hypervolume
reference point (1.1x the true front's per-objective max).

### Extended Modules

These modules are imported eagerly (`bananabench.cec`, `.g_suite`, `.gradients`,
`.multiobjective`, `.plotly_viz`) but only `gradients`, `g_suite`, and `multiobjective` have no
extra dependencies; `plotly_viz` degrades gracefully if `plotly` is not installed
(`pip install 'banana-bench[interactive]'`).

**`cec` — CEC-style composition framework**
- `CECFunction(base_func, shift_vector=None, rotation_matrix=None, bias=0.0)`: wraps any
  benchmark function with an optional shift, rotation, and additive bias, evaluated as
  `base_func(rotation @ (x - shift)) + bias`.
- `CECCompositionFunction(functions, sigma, biases)`: combines several `CECFunction` instances
  into a single weighted composition, a hallmark of the CEC 2017/2020 benchmark suites.

**`g_suite` — constrained G-function suite**
- `g04(x)`, `g06(x)`, `g08(x)`, `g11(x)`: classic constrained benchmark problems from the 2006
  CEC Special Session on Constrained Real-Parameter Optimization. Each returns
  `(objective, inequality_violations, equality_violations)` as `(float, np.ndarray, np.ndarray)`.

**`multiobjective` — ZDT multi-objective suite**
- `zdt1(x)` ... `zdt4(x)`, `zdt6(x)`: the five real-valued ZDT problems (ZDT5 is binary-encoded and
  not included). Each returns a 2-element objective array `[f1, f2]` (both minimized) instead of a
  single float.
- `ZDT_SUITE`: registry dict keyed by function name, with `function`, `n_objectives`, `default_dim`,
  `bounds` (a `dim -> [(min, max), ...]` callable), and `properties`.
- `get_mo_function_list()`, `get_mo_bounds(name, dim=None)`: registry accessors, analogous to
  `get_all_functions()`/`get_bounds()` for the scalar suite.
- `get_pareto_front(name, n_points=200)`: a sampled true Pareto front for a ZDT problem, for use as
  the reference front in quality indicators.
- `igd(front, reference_front)`: Inverted Generational Distance between an approximation front and
  the true front (lower is better).
- `hypervolume(front, reference_point)`: 2D hypervolume indicator (higher is better); raises
  `NotImplementedError` for more than 2 objectives.
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

### `quick_benchmark`

A helper function for rapid testing without creating a class instance.

```python
from bananabench import quick_benchmark

results = quick_benchmark(
    my_optimizer,
    function_names=['sphere', 'ackley'],
    n_runs=5,
    max_iter=100
)
```

## Utility Functions

### Bounds & Geometry

**`normalize_bounds(bounds, dim)`**
Converts various bound formats into a standard list of `(min, max)` tuples.

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
