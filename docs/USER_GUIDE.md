# User Guide

This guide provides detailed instructions on how to use `banana-bench` effectively.

## 🎯 Using Benchmark Metadata

The package provides comprehensive metadata for all 55+ functions, eliminating the need to manually specify bounds and known minima.

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
| `BENCHMARK_SUITE` | Dictionary with all 55+ functions and metadata |
| `get_all_functions()` | Returns list of all function names |
| `get_function_info(name)` | Returns metadata for specific function |
| `get_bounds(name, dim=None)` | Returns bounds for given dimension |
| `get_function_list()` | Returns formatted string with all functions |

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
algorithms = {
    'SimulatedAnnealing': simulated_annealing,
    'GeneticAlgorithm': genetic_algorithm
}

for name, algo in algorithms.items():
    runner = BenchmarkRunner(algo, algorithm_name=name)
    results = runner.run_suite()
    runner.save_results(f'{name}_results.csv')
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
