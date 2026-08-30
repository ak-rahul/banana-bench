# banana-bench

[![PyPI version](https://img.shields.io/pypi/v/banana-bench)](https://pypi.org/project/banana-bench/)
[![Python](https://img.shields.io/pypi/pyversions/banana-bench)](https://pypi.org/project/banana-bench/)
[![Downloads](https://pepy.tech/badge/banana-bench)](https://pepy.tech/project/banana-bench)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/ak-rahul/banana-bench/blob/main/LICENSE.md)
[![Build Status](https://github.com/ak-rahul/banana-bench/actions/workflows/quality.yml/badge.svg)](https://github.com/ak-rahul/banana-bench/actions/workflows/quality.yml)
[![Docs](https://app.readthedocs.org/projects/banana-bench/badge/?version=latest)](https://banana-bench.readthedocs.io/en/latest/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A comprehensive collection of 70+ standard mathematical benchmark functions for testing and evaluating optimization algorithms.

> **Why "banana-bench"?** Rosenbrock's function — one of the most famous test functions in
> optimization, and part of this suite (`rosenbrock`) — is nicknamed the "banana function" for its
> curved, banana-shaped valley. This is the bench you run your optimizer through.

## 📚 Documentation

Full documentation is published at **[banana-bench.readthedocs.io](https://banana-bench.readthedocs.io/)**, built from the sources in the `docs/` directory:

- **[User Guide & Examples](docs/USER_GUIDE.md)**: Detailed tutorials on metadata, benchmarking, and utilities.
- **[Function Reference](docs/BENCHMARK_FUNCTIONS.md)**: Complete guide to all 70+ functions with domains and global minima.
- **[Visualization Gallery](docs/VISUALIZATION.md)**: Examples of all plotting and analysis tools.
- **[API Reference](docs/API_REFERENCE.md)**: Class and function definitions.

## 🎯 Features

- **70+ Benchmark Functions**: Multimodal, Unimodal, and Special functions.
- **Rich Metadata**: Access bounds, dimensions, known minima programmatically.
- **Visualization**: 2D/3D plots, convergence tracking, and heatmaps.
- **Benchmarking Tools**: Automated testing with `BenchmarkRunner`.
- **Zero Core Dependencies**: Only NumPy is required.

## 📦 Installation

### From PyPI
```bash
pip install banana-bench
```

### From Source
```bash
git clone https://github.com/ak-rahul/banana-bench.git
cd banana-bench
pip install -e .
```

To install with visualization support:
```bash
pip install banana-bench[viz]
```

## 🚀 Quick Start

```python
import numpy as np
from bananabench import ackley, BenchmarkRunner

# 1. Use a single function
x = np.zeros(5)
print(f"Ackley(0) = {ackley(x)}")

# 2. Run a benchmark suite
def my_optimizer(func, bounds):
    # Your optimization logic here...
    return np.zeros(len(bounds)), 0.0

runner = BenchmarkRunner(my_optimizer, "MyAlgo", n_runs=5)
results = runner.run_suite(functions=['sphere', 'ackley'])
```

For detailed usage, see the **[User Guide](docs/USER_GUIDE.md)**.
