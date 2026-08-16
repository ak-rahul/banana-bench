# banana-bench Documentation

`banana-bench` is a collection of 55+ standard mathematical benchmark functions for testing and
evaluating optimization algorithms, plus metadata, benchmarking, visualization, and wrapper
utilities built around them. Core functionality depends only on NumPy; everything else is optional.

## Installation

```bash
pip install banana-bench
# with visualization support
pip install banana-bench[viz]
```

## Quick Start

```python
import numpy as np
from bananabench import ackley, BenchmarkRunner

x = np.zeros(5)
print(f"Ackley(0) = {ackley(x)}")

def my_optimizer(func, bounds):
    return np.zeros(len(bounds)), 0.0

runner = BenchmarkRunner(my_optimizer, "MyAlgo", n_runs=5)
results = runner.run_suite(functions=['sphere', 'ackley'])
```

```{toctree}
:maxdepth: 2
:caption: Contents:

USER_GUIDE.md
BENCHMARK_FUNCTIONS.md
API_REFERENCE.md
VISUALIZATION.md
```
