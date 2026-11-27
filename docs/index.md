# banana-bench Documentation

`banana-bench` is a collection of 70+ standard mathematical benchmark functions for testing and
evaluating optimization algorithms, plus constrained, engineering and multi-objective suites, and
the tools to benchmark fairly and compare results: evaluation budgets, result verification,
performance and data profiles, significance tests and a center-bias test. Core functionality
depends only on NumPy; everything else is optional.

## Installation

```bash
pip install banana-bench
pip install "banana-bench[viz]"     # Matplotlib plotting
pip install "banana-bench[stats]"   # significance tests (SciPy)
pip install "banana-bench[all]"     # every optional dependency
```

## Quick Start

```python
import numpy as np
from bananabench import BenchmarkRunner, ackley

print(f"Ackley(0) = {ackley(np.zeros(5))}")

def random_search(func, bounds):
    """Samples until the runner's evaluation budget stops it."""
    low, high = np.array(bounds).T
    best_x, best_f = None, np.inf
    while True:
        x = np.random.uniform(low, high)
        fx = func(x)
        if fx < best_f:
            best_x, best_f = x, fx

runner = BenchmarkRunner(random_search, "RandomSearch", n_runs=5,
                         max_evals=lambda dim: 1000 * dim)
results = runner.run_suite(functions=['sphere', 'ackley'])
```

The runner counts every evaluation, stops each run at its budget and scores the best point the
run evaluated. See the [User Guide](USER_GUIDE.md) for everything else, and the
[Examples](examples.md) for runnable scripts.

```{toctree}
:maxdepth: 2
:caption: Contents:

USER_GUIDE.md
BENCHMARK_FUNCTIONS.md
API_REFERENCE.md
VISUALIZATION.md
examples.md
api/index.md
changelog.md
```
