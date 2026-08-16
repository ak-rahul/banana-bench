"""
banana-bench: Comprehensive Benchmark Functions for Optimization

A collection of 55+ standard mathematical benchmark functions for evaluating
optimization algorithms.

References
----------
.. [1] Adorio, E. P., & Diliman, U. (2005). MVF-multivariate test functions
       library in C for unconstrained global optimization.
.. [2] Surjanovic, S., & Bingham, D. (2013). Virtual Library of Simulation
       Experiments: Test Functions and Datasets.
       http://www.sfu.ca/~ssurjano
.. [3] Jamil, M., & Yang, X. S. (2013). A literature survey of benchmark
       functions for global optimization problems. International Journal of
       Mathematical Modelling and Numerical Optimisation, 4(2), 150-194.
       https://doi.org/10.1504/IJMMNO.2013.055204

License: MIT
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("banana-bench")
except PackageNotFoundError:  # pragma: no cover - only hit when running from source, uninstalled
    __version__ = "0.0.0+unknown"

__author__ = "AK Rahul"
__license__ = "MIT"

import warnings

from . import cec, g_suite, gradients, multiobjective, plotly_viz
from .benchmarking import BenchmarkRunner, MOBenchmarkRunner, quick_benchmark
from .functions import (
    ackley,
    beale,
    bohachevsky1,
    bohachevsky2,
    booth,
    box_betts,
    branin,
    branin2,
    camel3,
    camel6,
    chichinadze,
    colville,
    corana,
    easom,
    eggholder,
    exp2,
    fraudenstein_roth,
    freudenstein_roth,
    gear,
    goldstein_price,
    griewank,
    himmelblau,
    holzman1,
    holzman2,
    hosaki,
    hyperellipsoid,
    katsuura,
    kowalik,
    langerman,
    lennard_jones,
    leon,
    levy,
    matyas,
    maxmod,
    mccormick,
    michalewicz,
    multimod,
    rastrigin,
    rastrigin2,
    rosenbrock,
    rosenbrock_ext1,
    rosenbrock_ext2,
    schaffer1,
    schaffer2,
    schwefel1_2,
    schwefel2_21,
    schwefel2_22,
    schwefel2_26,
    schwefel3_2,
    sphere,
    sphere2,
    step,
    step2,
    stretched_v,
    sum_squares,
    trecanni,
    trefethen4,
    watson,
    xor,
    zettl,
    zimmerman,
)
from .metadata import (
    BENCHMARK_SUITE,
    get_all_functions,
    get_bounds,
    get_function_info,
    get_function_list,
)
from .utils import (
    calculate_distance_to_optimum,
    check_bounds,
    clip_to_bounds,
    generate_grid_points,
    generate_random_point,
    get_bounds_center,
    get_bounds_range,
    normalize_bounds,
    scale_from_unit,
    scale_to_unit,
)

# .visualization always imports successfully (it defers its plt.Figure-typed
# annotations via `from __future__ import annotations` precisely so this
# import can't fail just because matplotlib is missing) — so check its own
# MATPLOTLIB_AVAILABLE flag directly instead of an ImportError side effect.
from .visualization import (
    COLORMAPS,
    animate_trajectory_2d,
    batch_plot_functions,
    plot_algorithm_comparison,
    plot_benchmark_summary,
    plot_convergence,
    plot_function_2d,
    plot_function_3d,
    plot_search_heatmap,
    plot_trajectory_2d,
    save_plot,
)
from .visualization import MATPLOTLIB_AVAILABLE as __visualization_available__
from .wrappers import BenchmarkWrapper, NoisyFunction, RotatedFunction, ShiftedFunction

if not __visualization_available__:
    warnings.warn(
        "matplotlib not installed. Install with: pip install 'banana-bench[viz]'",
        ImportWarning,
        stacklevel=2,
    )

__all__ = [
    "__version__",
    "ackley",
    "rastrigin",
    "rastrigin2",
    "griewank",
    "levy",
    "michalewicz",
    "schwefel2_26",
    "sphere",
    "sphere2",
    "rosenbrock",
    "rosenbrock_ext1",
    "rosenbrock_ext2",
    "sum_squares",
    "hyperellipsoid",
    "schwefel1_2",
    "schwefel2_21",
    "schwefel2_22",
    "schwefel3_2",
    "step",
    "step2",
    "maxmod",
    "multimod",
    "katsuura",
    "beale",
    "booth",
    "matyas",
    "himmelblau",
    "easom",
    "goldstein_price",
    "branin",
    "branin2",
    "camel3",
    "camel6",
    "bohachevsky1",
    "bohachevsky2",
    "schaffer1",
    "schaffer2",
    "leon",
    "trecanni",
    "mccormick",
    "eggholder",
    "chichinadze",
    "hosaki",
    "zettl",
    "holzman1",
    "holzman2",
    "langerman",
    "stretched_v",
    "trefethen4",
    "box_betts",
    "colville",
    "corana",
    "kowalik",
    "exp2",
    "gear",
    "watson",
    "xor",
    "zimmerman",
    "lennard_jones",
    "fraudenstein_roth",
    "freudenstein_roth",
    "BENCHMARK_SUITE",
    "get_all_functions",
    "get_function_info",
    "get_bounds",
    "get_function_list",
    "normalize_bounds",
    "generate_random_point",
    "check_bounds",
    "scale_to_unit",
    "scale_from_unit",
    "clip_to_bounds",
    "get_bounds_range",
    "get_bounds_center",
    "generate_grid_points",
    "calculate_distance_to_optimum",
    "BenchmarkRunner",
    "MOBenchmarkRunner",
    "quick_benchmark",
    "BenchmarkWrapper",
    "NoisyFunction",
    "ShiftedFunction",
    "RotatedFunction",
    "gradients",
    "plotly_viz",
    "g_suite",
    "cec",
    "multiobjective",
]

if __visualization_available__:
    __all__.extend(
        [
            "plot_function_2d",
            "plot_function_3d",
            "plot_convergence",
            "plot_trajectory_2d",
            "plot_algorithm_comparison",
            "plot_benchmark_summary",
            "plot_search_heatmap",
            "save_plot",
            "batch_plot_functions",
            "animate_trajectory_2d",
            "COLORMAPS",
        ]
    )


def get_version():
    """Get package version."""
    return __version__


def list_functions():
    """Get list of all available benchmark functions."""
    return get_function_list()


__all__.extend(["get_version", "list_functions"])
