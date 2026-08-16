# Visualization Gallery

The `banana-bench` package includes a powerful set of visualization tools to analyze optimization landscapes and algorithm performance.

## Installation

To use visualization features, you must install the optional dependencies:

```bash
pip install banana-bench[viz]
```
Or for all features:
```bash
pip install banana-bench[all]
```

## Plotting Functions

### 2D Function Landscape
Visualize the landscape of 2D functions (or 2D slices of n-D functions).

```python
from bananabench.visualization import plot_function_2d
import matplotlib.pyplot as plt

# Simple plot
plot_function_2d('ackley')
plt.show()

# Customize with optimum and resolution
plot_function_2d('rastrigin', cmap='inferno', resolution=200, show_optimum=True)
plt.show()

# Custom bounds
plot_function_2d('sphere', bounds=[(-10, 10), (-10, 10)])
plt.show()
```

### 3D Surface Plot
Create interactive 3D surface plots.

```python
from bananabench.visualization import plot_function_3d

# Basic 3D plot
plot_function_3d('sphere', elevation=30, azimuth=45)
plt.show()

# Use different colormap
plot_function_3d('griewank', cmap='plasma')
plt.show()
```

### Optimization Animation
Create animated GIFs of your optimization trajectory.

```python
from bananabench.visualization import animate_trajectory_2d

# Create and save animation
anim = animate_trajectory_2d(
    'sphere',
    trajectory,
    save_path='optimization.gif',
    fps=15,
    interval=100
)
```

### Optimization Trajectory
Visualize the path taken by your optimizer.

```python
from bananabench.visualization import plot_trajectory_2d
import numpy as np

# Trajectory must be an array of points (N, 2)
trajectory = np.array([
    [5.0, 5.0], [3.0, 3.0], [1.0, 1.0], [0.1, 0.1], [0.0, 0.0]
])

fig = plot_trajectory_2d('sphere', trajectory)
plt.show()
```

### Convergence Plot
Track the cost function value over iterations.

```python
from bananabench.visualization import plot_convergence

# Single run history
history = [100, 50, 20, 10, 5, 2, 1, 0.1, 0.01]
plot_convergence(history, function_name='sphere', known_minimum=0.0, log_scale=True)
plt.show()

# Compare 'best' vs 'current' cost
history_dict = {
    'best': [10, 5, 2, 1, 0.5],
    'current': [10, 7, 3, 2, 1],
    'iterations': range(5)
}
plot_convergence(history_dict, function_name='ackley', known_minimum=0.0)
plt.show()
```

### Search Heatmap
Visualize the density of search points to check exploration/exploitation balance.

```python
from bananabench.visualization import plot_search_heatmap
import numpy as np

# visited_points is every point your optimizer evaluated, shape (n_points, 2)
visited_points = np.random.uniform(-5, 5, size=(500, 2))

# Visualizing where the algorithm spent the most time
plot_search_heatmap('rastrigin', visited_points, bins=30, cmap='hot')
plt.show()
```

### Algorithm Comparison
Compare multiple algorithms on various metrics using bar charts.

```python
from bananabench.visualization import plot_algorithm_comparison

results = {
    'SimulatedAnnealing': {'sphere': {'error': 0.001, 'time': 1.2}},
    'GeneticAlgorithm': {'sphere': {'error': 0.01, 'time': 0.8}}
}

# Compare by Error
plot_algorithm_comparison(results, metric='error')
plt.show()

# Compare by Time
plot_algorithm_comparison(results, metric='time')
plt.show()
```

### Benchmark Summary Dashboard
Generate a comprehensive 4-panel summary of your benchmark results.

```python
from bananabench.visualization import plot_benchmark_summary

# results is a list of dictionaries from BenchmarkRunner
fig = plot_benchmark_summary(results)
plt.savefig('summary.png', dpi=300)
plt.show()
```

## Advanced Features

### Multi-Format Plot Export
Save plots in multiple formats simultaneously (PNG, SVG, PDF, EPS).

```python
from bananabench.visualization import save_plot

fig = plot_function_2d('ackley')
save_plot(fig, 'ackley_plot', formats=['png', 'svg', 'pdf'], dpi=300)
```

### Batch Plotting
Generate multiple plots for many functions efficiently.

```python
from bananabench import batch_plot_functions

batch_plot_functions(
    function_names=['sphere', 'ackley', 'rastrigin'],
    plot_types=['2d', '3d'],
    output_dir='plots',
    formats=['png', 'svg'],
    cmap='viridis'
)
```

### Available Colormaps
The package supports standard matplotlib colormaps (e.g., 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'coolwarm', 'turbo').
