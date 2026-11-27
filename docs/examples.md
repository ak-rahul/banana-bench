# Examples

Runnable scripts in the repository's
[`examples/`](https://github.com/ak-rahul/banana-bench/tree/main/examples) folder. Each one is
self-contained: install the package, then run it with `python examples/<name>.py`. Some need an
optional extra, noted at the top of the script.

| Script | What it shows |
|--------|---------------|
| [`01_quickstart.py`](https://github.com/ak-rahul/banana-bench/blob/main/examples/01_quickstart.py) | Evaluating functions and reading their metadata |
| [`02_custom_optimizer_benchmarking.py`](https://github.com/ak-rahul/banana-bench/blob/main/examples/02_custom_optimizer_benchmarking.py) | Benchmarking your own optimizer with `BenchmarkRunner` |
| [`03_robustness_wrappers.py`](https://github.com/ak-rahul/banana-bench/blob/main/examples/03_robustness_wrappers.py) | Noise, shift and rotation wrappers for robustness testing |
| [`04_gradient_based_optimization.py`](https://github.com/ak-rahul/banana-bench/blob/main/examples/04_gradient_based_optimization.py) | Finite-difference gradients with SciPy's L-BFGS-B |
| [`05_constrained_optimization.py`](https://github.com/ak-rahul/banana-bench/blob/main/examples/05_constrained_optimization.py) | The constrained G-suite with SciPy's SLSQP |
| [`06_visualization_gallery.py`](https://github.com/ak-rahul/banana-bench/blob/main/examples/06_visualization_gallery.py) | Every plotting function in one place |
| [`07_multiobjective_optimization.py`](https://github.com/ak-rahul/banana-bench/blob/main/examples/07_multiobjective_optimization.py) | The ZDT suite with `MOBenchmarkRunner`, IGD and hypervolume |
| [`08_engineering_design.py`](https://github.com/ak-rahul/banana-bench/blob/main/examples/08_engineering_design.py) | Classic engineering design problems |
| [`09_cec_style_composition.py`](https://github.com/ak-rahul/banana-bench/blob/main/examples/09_cec_style_composition.py) | CEC-style shift, rotation and composition on your own function |
| [`10_optuna_integration.py`](https://github.com/ak-rahul/banana-bench/blob/main/examples/10_optuna_integration.py) | Using a benchmark function as an Optuna objective |
| [`11_budgeted_benchmarking.py`](https://github.com/ak-rahul/banana-bench/blob/main/examples/11_budgeted_benchmarking.py) | Equal evaluation budgets, and verification catching a misreported result |
| [`12_performance_profiles.py`](https://github.com/ak-rahul/banana-bench/blob/main/examples/12_performance_profiles.py) | ERT, performance and data profiles, runtime ECDFs and significance tests |
| [`13_many_objective_dtlz.py`](https://github.com/ak-rahul/banana-bench/blob/main/examples/13_many_objective_dtlz.py) | DTLZ with three and five objectives, exact and Monte Carlo hypervolume |
| [`14_center_bias.py`](https://github.com/ak-rahul/banana-bench/blob/main/examples/14_center_bias.py) | Telling a center-biased optimizer from an unbiased one |
