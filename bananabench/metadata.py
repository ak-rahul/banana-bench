"""
Benchmark Function Metadata

Complete metadata for all optimization benchmark functions including bounds,
dimensions, known minima, and optimal points based on MVF documentation.

References
----------
[1] Adorio, E. P. (2005). MVF - Multivariate Test Functions Library in C for
    Unconstrained Global Optimization. University of the Philippines Diliman.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import functions as _f

# Complete metadata for all benchmark functions
BENCHMARK_SUITE: Dict[str, Any] = {
    # High-dimensional multimodal functions
    "ackley": {
        "function": _f.ackley,
        "properties": ["continuous", "differentiable", "non-separable", "scalable", "multimodal"],
        "bounds": [(-30, 30)],
        "default_dim": 10,
        "known_minimum": 0.0,
        "optimal_point": [0.0],
    },
    "rastrigin": {
        "function": _f.rastrigin,
        "properties": ["continuous", "differentiable", "separable", "scalable", "multimodal"],
        "bounds": [(-5.12, 5.12)],
        "default_dim": 10,
        "known_minimum": 0.0,
        "optimal_point": [0.0],
    },
    "rastrigin2": {
        "function": _f.rastrigin2,
        "properties": ["continuous", "differentiable", "separable", "multimodal"],
        "bounds": [(-5.12, 5.12)],
        "default_dim": 2,
        "known_minimum": -2.0,
        "optimal_point": [0.0, 0.0],
    },
    "griewank": {
        "function": _f.griewank,
        "properties": ["continuous", "differentiable", "non-separable", "scalable", "multimodal"],
        "bounds": [(-600, 600)],
        "default_dim": 10,
        "known_minimum": 0.0,
        "optimal_point": [0.0],
    },
    "levy": {
        "function": _f.levy,
        "properties": ["continuous", "differentiable", "non-separable", "scalable", "multimodal"],
        "bounds": [(-10, 10)],
        "default_dim": 10,
        "known_minimum": 0.0,
        "optimal_point": [1.0],
    },
    "michalewicz": {
        "function": _f.michalewicz,
        "properties": ["continuous", "differentiable", "non-separable", "scalable", "multimodal"],
        "bounds": [(0, np.pi)],
        "default_dim": 10,
        "known_minimum": -9.66,  # Approximate for d=10
        "optimal_point": None,  # Multiple minima
    },
    "schwefel2_26": {
        "function": _f.schwefel2_26,
        "properties": ["continuous", "differentiable", "separable", "scalable", "multimodal"],
        "bounds": [(-500, 500)],
        "default_dim": 10,
        "known_minimum": -4189.829,  # -418.9829 * 10
        "optimal_point": [420.9687],
    },
    # High-dimensional unimodal functions
    "sphere": {
        "function": _f.sphere,
        "properties": [
            "continuous",
            "differentiable",
            "separable",
            "scalable",
            "unimodal",
            "convex",
        ],
        "bounds": [(-100, 100)],
        "default_dim": 10,
        "known_minimum": 0.0,
        "optimal_point": [0.0],
    },
    "sphere2": {
        "function": _f.sphere2,
        "properties": [
            "continuous",
            "differentiable",
            "non-separable",
            "scalable",
            "unimodal",
            "convex",
        ],
        "bounds": [(-100, 100)],
        "default_dim": 10,
        "known_minimum": 0.0,
        "optimal_point": [0.0],
    },
    "rosenbrock": {
        "function": _f.rosenbrock,
        "properties": [
            "continuous",
            "differentiable",
            "non-separable",
            "scalable",
            "unimodal",
            "non-convex",
        ],
        "bounds": [(-10, 10)],
        "default_dim": 10,
        "known_minimum": 0.0,
        "optimal_point": [1.0],
    },
    "rosenbrock_ext1": {
        "function": _f.rosenbrock_ext1,
        "properties": [
            "continuous",
            "differentiable",
            "non-separable",
            "scalable",
            "unimodal",
            "non-convex",
        ],
        "bounds": [(-10, 10)],
        "default_dim": 10,
        "known_minimum": 0.0,
        "optimal_point": [1.0],
    },
    "rosenbrock_ext2": {
        "function": _f.rosenbrock_ext2,
        "properties": [
            "continuous",
            "differentiable",
            "non-separable",
            "scalable",
            "unimodal",
            "non-convex",
        ],
        "bounds": [(-10, 10)],
        "default_dim": 10,
        "known_minimum": 0.0,
        "optimal_point": [1.0],
    },
    "sum_squares": {
        "function": _f.sum_squares,
        "properties": [
            "continuous",
            "differentiable",
            "separable",
            "scalable",
            "unimodal",
            "convex",
        ],
        "bounds": [(-10, 10)],
        "default_dim": 10,
        "known_minimum": 0.0,
        "optimal_point": [0.0],
    },
    "hyperellipsoid": {
        "function": _f.hyperellipsoid,
        "properties": [
            "continuous",
            "differentiable",
            "separable",
            "scalable",
            "unimodal",
            "convex",
        ],
        "bounds": [(-10, 10)],
        "default_dim": 10,
        "known_minimum": 0.0,
        "optimal_point": [0.0],
    },
    "schwefel1_2": {
        "function": _f.schwefel1_2,
        "properties": [
            "continuous",
            "differentiable",
            "non-separable",
            "scalable",
            "unimodal",
            "convex",
        ],
        "bounds": [(-100, 100)],
        "default_dim": 10,
        "known_minimum": 0.0,
        "optimal_point": [0.0],
    },
    "schwefel2_21": {
        "function": _f.schwefel2_21,
        "properties": [
            "continuous",
            "non-differentiable",
            "separable",
            "scalable",
            "unimodal",
            "convex",
        ],
        "bounds": [(-100, 100)],
        "default_dim": 10,
        "known_minimum": 0.0,
        "optimal_point": [0.0],
    },
    "schwefel2_22": {
        "function": _f.schwefel2_22,
        "properties": ["continuous", "non-differentiable", "non-separable", "scalable", "unimodal"],
        "bounds": [(-10, 10)],
        "default_dim": 10,
        "known_minimum": 0.0,
        "optimal_point": [0.0],
    },
    "schwefel3_2": {
        "function": _f.schwefel3_2,
        "properties": ["continuous", "differentiable", "non-separable", "unimodal", "convex"],
        "bounds": [(-10, 10)],
        "default_dim": 10,
        "known_minimum": 0.0,
        "optimal_point": [1.0],
    },
    "step": {
        "function": _f.step,
        "properties": ["discontinuous", "non-differentiable", "separable", "scalable", "unimodal"],
        "bounds": [(-100, 100)],
        "default_dim": 10,
        "known_minimum": 2.5,  # 0.25 * n for n=10; (floor(x)+0.5)**2 bottoms out at 0.25 per dim
        "optimal_point": [0.0],
    },
    "step2": {
        "function": _f.step2,
        "properties": ["discontinuous", "non-differentiable", "separable", "scalable", "unimodal"],
        "bounds": [(-5.12, 5.12)],
        "default_dim": 5,
        "known_minimum": 30.0,  # 6*n for n=5
        "optimal_point": [0.0],
    },
    "maxmod": {
        "function": _f.maxmod,
        "properties": [
            "continuous",
            "non-differentiable",
            "separable",
            "scalable",
            "unimodal",
            "convex",
        ],
        "bounds": [(-10, 10)],
        "default_dim": 10,
        "known_minimum": 0.0,
        "optimal_point": [0.0],
    },
    "multimod": {
        "function": _f.multimod,
        "properties": ["continuous", "non-differentiable", "non-separable", "scalable"],
        "bounds": [(-10, 10)],
        "default_dim": 10,
        "known_minimum": 0.0,
        "optimal_point": [0.0],
    },
    "katsuura": {
        "function": _f.katsuura,
        "properties": [
            "continuous",
            "non-differentiable",
            "non-separable",
            "scalable",
            "multimodal",
        ],
        "bounds": [(-1000, 1000)],
        "default_dim": 10,
        "known_minimum": 1.0,
        "optimal_point": [0.0],
    },
    # 2D functions
    "beale": {
        "function": _f.beale,
        "properties": ["continuous", "differentiable", "non-separable", "multimodal"],
        "bounds": [(-4.5, 4.5), (-4.5, 4.5)],
        "default_dim": 2,
        "known_minimum": 0.0,
        "optimal_point": [3.0, 0.5],
    },
    "booth": {
        "function": _f.booth,
        "properties": ["continuous", "differentiable", "non-separable", "unimodal", "convex"],
        "bounds": [(-10, 10), (-10, 10)],
        "default_dim": 2,
        "known_minimum": 0.0,
        "optimal_point": [1.0, 3.0],
    },
    "matyas": {
        "function": _f.matyas,
        "properties": ["continuous", "differentiable", "non-separable", "unimodal", "convex"],
        "bounds": [(-10, 10), (-10, 10)],
        "default_dim": 2,
        "known_minimum": 0.0,
        "optimal_point": [0.0, 0.0],
    },
    "himmelblau": {
        "function": _f.himmelblau,
        "properties": ["continuous", "differentiable", "non-separable", "multimodal"],
        "bounds": [(-6, 6), (-6, 6)],
        "default_dim": 2,
        "known_minimum": 0.0,
        "optimal_point": [3.0, 2.0],
    },
    "easom": {
        "function": _f.easom,
        "properties": ["continuous", "differentiable", "non-separable", "multimodal"],
        "bounds": [(-100, 100), (-100, 100)],
        "default_dim": 2,
        "known_minimum": -1.0,
        "optimal_point": [np.pi, np.pi],
    },
    "goldstein_price": {
        "function": _f.goldstein_price,
        "properties": ["continuous", "differentiable", "non-separable", "multimodal"],
        "bounds": [(-2, 2), (-2, 2)],
        "default_dim": 2,
        "known_minimum": 3.0,
        "optimal_point": [0.0, -1.0],
    },
    "branin": {
        "function": _f.branin,
        "properties": ["continuous", "differentiable", "non-separable", "multimodal"],
        "bounds": [(-5, 10), (0, 15)],
        "default_dim": 2,
        "known_minimum": 0.397887,
        "optimal_point": [(-np.pi, 12.275), (np.pi, 2.275), (9.42478, 2.475)],  # Multiple minima
    },
    "branin2": {
        "function": _f.branin2,
        "properties": ["continuous", "differentiable", "non-separable", "multimodal"],
        "bounds": [(-10, 10), (-10, 10)],
        "default_dim": 2,
        "known_minimum": 0.0,
        "optimal_point": [0.402357, 0.287408],
    },
    "camel3": {
        "function": _f.camel3,
        "properties": ["continuous", "differentiable", "non-separable", "multimodal"],
        "bounds": [(-5, 5), (-5, 5)],
        "default_dim": 2,
        "known_minimum": 0.0,
        "optimal_point": [0.0, 0.0],
    },
    "camel6": {
        "function": _f.camel6,
        "properties": ["continuous", "differentiable", "non-separable", "multimodal"],
        "bounds": [(-5, 5), (-5, 5)],
        "default_dim": 2,
        "known_minimum": -1.0316285,
        "optimal_point": [(0.08983, -0.7126), (-0.08983, 0.7126)],  # Two minima
    },
    "bohachevsky1": {
        "function": _f.bohachevsky1,
        "properties": ["continuous", "differentiable", "separable", "multimodal"],
        "bounds": [(-50, 50), (-50, 50)],
        "default_dim": 2,
        "known_minimum": 0.0,
        "optimal_point": [0.0, 0.0],
    },
    "bohachevsky2": {
        "function": _f.bohachevsky2,
        "properties": ["continuous", "differentiable", "non-separable", "multimodal"],
        "bounds": [(-50, 50), (-50, 50)],
        "default_dim": 2,
        "known_minimum": 0.0,
        "optimal_point": [0.0, 0.0],
    },
    "schaffer1": {
        "function": _f.schaffer1,
        "properties": ["continuous", "differentiable", "non-separable", "multimodal"],
        "bounds": [(-100, 100), (-100, 100)],
        "default_dim": 2,
        "known_minimum": 0.0,
        "optimal_point": [0.0, 0.0],
    },
    "schaffer2": {
        "function": _f.schaffer2,
        "properties": ["continuous", "differentiable", "non-separable", "unimodal"],
        "bounds": [(-100, 100), (-100, 100)],
        "default_dim": 2,
        "known_minimum": 0.0,
        "optimal_point": [0.0, 0.0],
    },
    "leon": {
        "function": _f.leon,
        "properties": ["continuous", "differentiable", "non-separable", "unimodal", "non-convex"],
        "bounds": [(-10, 10), (-10, 10)],
        "default_dim": 2,
        "known_minimum": 0.0,
        "optimal_point": [1.0, 1.0],
    },
    "trecanni": {
        "function": _f.trecanni,
        "properties": ["continuous", "differentiable", "separable", "multimodal", "non-convex"],
        "bounds": [(-5, 5), (-5, 5)],
        "default_dim": 2,
        "known_minimum": 0.0,
        "optimal_point": [(0.0, 0.0), (-2.0, 0.0)],  # Two minima
    },
    "mccormick": {
        "function": _f.mccormick,
        "properties": ["continuous", "differentiable", "non-separable", "multimodal"],
        "bounds": [(-1.5, 4), (-3, 4)],
        "default_dim": 2,
        "known_minimum": -1.9133,
        "optimal_point": [-0.54719, -1.54719],
    },
    "eggholder": {
        "function": _f.eggholder,
        "properties": ["continuous", "differentiable", "non-separable", "multimodal"],
        "bounds": [(-512, 512), (-512, 512)],
        "default_dim": 2,
        "known_minimum": -959.6407,  # Approximate
        "optimal_point": [512, 404.2319],  # Approximate
    },
    "chichinadze": {
        "function": _f.chichinadze,
        "properties": ["continuous", "differentiable", "separable", "multimodal"],
        "bounds": [(-30, 30), (-10, 10)],
        "default_dim": 2,
        "known_minimum": -43.868648,
        "optimal_point": [5.90133, 0.5],
    },
    "hosaki": {
        "function": _f.hosaki,
        "properties": ["continuous", "differentiable", "non-separable", "multimodal"],
        "bounds": [(0, 5), (0, 6)],
        "default_dim": 2,
        "known_minimum": -2.3458,
        "optimal_point": [4.0, 2.0],
    },
    "zettl": {
        "function": _f.zettl,
        "properties": ["continuous", "differentiable", "non-separable", "unimodal", "non-convex"],
        "bounds": [(-10, 10), (-10, 10)],
        "default_dim": 2,
        "known_minimum": -0.003791,
        "optimal_point": [-0.02990, 0.0],
    },
    # 3D functions
    "holzman1": {
        "function": _f.holzman1,
        "properties": ["continuous", "differentiable", "non-separable", "multimodal"],
        "bounds": [(0.1, 100), (0, 25.6), (0, 5)],
        "default_dim": 3,
        "known_minimum": 0.0,
        "optimal_point": [50, 25, 1.5],
    },
    "holzman2": {
        "function": _f.holzman2,
        "properties": [
            "continuous",
            "differentiable",
            "separable",
            "scalable",
            "unimodal",
            "convex",
        ],
        "bounds": [(-10, 10)],
        "default_dim": 3,
        "known_minimum": 0.0,
        "optimal_point": [0.0],
    },
    "langerman": {
        "function": _f.langerman,
        "properties": ["continuous", "differentiable", "non-separable", "multimodal"],
        "bounds": [(0, 10)],
        "default_dim": 10,
        "known_minimum": -1.4,
        "optimal_point": None,  # Complex
    },
    "stretched_v": {
        "function": _f.stretched_v,
        "properties": ["continuous", "differentiable", "non-separable", "scalable", "multimodal"],
        "bounds": [(-10, 10)],
        "default_dim": 3,
        "known_minimum": 0.0,
        "optimal_point": None,
    },
    "trefethen4": {
        "function": _f.trefethen4,
        "properties": ["continuous", "differentiable", "non-separable", "multimodal"],
        "bounds": [(-6.5, 6.5), (-4.5, 4.5)],
        "default_dim": 2,
        "known_minimum": -3.30686865,
        "optimal_point": [-0.0244031, 0.2106124],
    },
    "box_betts": {
        "function": _f.box_betts,
        "properties": ["continuous", "differentiable", "non-separable", "unimodal"],
        "bounds": [(0.9, 1.2), (9, 11.2), (0.9, 1.2)],
        "default_dim": 3,
        "known_minimum": 0.0,
        "optimal_point": [1.0, 10.0, 1.0],
    },
    # 4D functions
    "colville": {
        "function": _f.colville,
        "properties": ["continuous", "differentiable", "non-separable", "unimodal", "non-convex"],
        "bounds": [(-10, 10)],
        "default_dim": 4,
        "known_minimum": 0.0,
        "optimal_point": [1.0],
    },
    "corana": {
        "function": _f.corana,
        "properties": ["discontinuous", "non-differentiable", "separable", "multimodal"],
        "bounds": [(-100, 100)],
        "default_dim": 4,
        "known_minimum": 0.0,
        "optimal_point": [0.0],
    },
    "kowalik": {
        "function": _f.kowalik,
        "properties": ["continuous", "differentiable", "non-separable", "unimodal"],
        "bounds": [(-5, 5)],
        "default_dim": 4,
        "known_minimum": 0.00030748610,
        "optimal_point": [0.192833, 0.190836, 0.123117, 0.135766],
    },
    "exp2": {
        "function": _f.exp2,
        "properties": ["continuous", "differentiable", "non-separable", "multimodal"],
        "bounds": [(0, 20)],
        "default_dim": 2,
        "known_minimum": 0.0,
        "optimal_point": [1.0, 10.0],
    },
    "gear": {
        "function": _f.gear,
        "properties": ["discontinuous", "non-differentiable", "non-separable", "multimodal"],
        "bounds": [(12, 60)],
        "default_dim": 4,
        "known_minimum": 2.7e-12,
        "optimal_point": [16, 19, 43, 49],
    },
    # Functions previously missing from BENCHMARK_SUITE
    "watson": {
        "function": _f.watson,
        "properties": ["continuous", "differentiable", "non-separable", "multimodal"],
        "bounds": [(-10, 10)],
        "default_dim": 6,
        "known_minimum": 0.002288,
        "optimal_point": None,
    },
    "xor": {
        "function": _f.xor,
        "properties": ["continuous", "differentiable", "non-separable", "multimodal"],
        "bounds": [(-1, 1)],
        "default_dim": 9,
        "known_minimum": 0.0,
        "optimal_point": None,
    },
    "zimmerman": {
        "function": _f.zimmerman,
        "properties": ["discontinuous", "non-differentiable", "non-separable", "multimodal"],
        "bounds": [(0, 100), (0, 100)],
        "default_dim": 2,
        "known_minimum": 0.0,
        "optimal_point": [7.0, 2.0],
    },
    "lennard_jones": {
        "function": _f.lennard_jones,
        "properties": ["continuous", "differentiable", "non-separable", "scalable", "multimodal"],
        "bounds": [(-2, 2)],
        "default_dim": 6,
        "known_minimum": -1.0,
        "optimal_point": None,
    },
    "hansen": {
        "function": _f.hansen,
        "properties": ["continuous", "differentiable", "non-separable", "multimodal", "non-convex"],
        "bounds": [(-10, 10)],
        "default_dim": 2,
        "known_minimum": -176.541793,
        "optimal_point": [-7.589893, -7.708314],
    },
    "hartman3": {
        "function": _f.hartman3,
        "properties": ["continuous", "differentiable", "non-separable", "multimodal", "non-convex"],
        "bounds": [(0, 1)],
        "default_dim": 3,
        "known_minimum": -3.862782,
        "optimal_point": [0.114614, 0.555649, 0.852547],
    },
    "hartman6": {
        "function": _f.hartman6,
        "properties": ["continuous", "differentiable", "non-separable", "multimodal", "non-convex"],
        "bounds": [(0, 1)],
        "default_dim": 6,
        "known_minimum": -3.322368,
        "optimal_point": [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573],
    },
    "neumaier_perm": {
        "function": _f.neumaier_perm,
        "properties": ["continuous", "differentiable", "non-separable", "multimodal", "non-convex"],
        "bounds": [(-4, 4)],
        "default_dim": 4,
        "known_minimum": 0.0,
        "optimal_point": [1.0, 2.0, 3.0, 4.0],
    },
    "neumaier_perm0": {
        "function": _f.neumaier_perm0,
        "properties": ["continuous", "differentiable", "non-separable", "multimodal", "non-convex"],
        "bounds": [(-1, 1)],
        "default_dim": 4,
        "known_minimum": 0.0,
        "optimal_point": [1.0, 0.5, 1.0 / 3.0, 0.25],
    },
    "neumaier_powersum": {
        "function": _f.neumaier_powersum,
        "properties": ["continuous", "differentiable", "non-separable"],
        "bounds": [(0, 4)],
        "default_dim": 4,
        "known_minimum": 0.0,
        "optimal_point": [1.0, 2.0, 2.0, 3.0],
    },
    "neumaier_trid": {
        "function": _f.neumaier_trid,
        "properties": ["continuous", "differentiable", "non-separable", "unimodal", "convex"],
        "bounds": [(-36, 36)],
        "default_dim": 6,
        "known_minimum": -50.0,
        "optimal_point": [6.0, 10.0, 12.0, 12.0, 10.0, 6.0],
    },
    "paviani": {
        "function": _f.paviani,
        "properties": ["continuous", "differentiable", "non-separable", "multimodal", "non-convex"],
        "bounds": [(2.0001, 9.9999)],
        "default_dim": 10,
        "known_minimum": -45.778470,
        "optimal_point": [9.350266] * 10,
    },
    "plateau": {
        "function": _f.plateau,
        "properties": ["discontinuous", "non-differentiable", "separable", "non-convex"],
        "bounds": [(-5.12, 5.12)],
        "default_dim": 5,
        "known_minimum": 30.0,
        "optimal_point": [0.0, 0.0, 0.0, 0.0, 0.0],
    },
    "powell": {
        "function": _f.powell,
        "properties": [
            "continuous",
            "differentiable",
            "non-separable",
            "scalable",
            "unimodal",
            "convex",
        ],
        "bounds": [(-4, 5)],
        "default_dim": 4,
        "known_minimum": 0.0,
        "optimal_point": [0.0],
    },
    "shekel2": {
        "function": _f.shekel2,
        "properties": ["continuous", "differentiable", "non-separable", "multimodal", "non-convex"],
        "bounds": [(-65.536, 65.536)],
        "default_dim": 2,
        "known_minimum": 0.998004,
        "optimal_point": [-31.97833, -31.97833],
    },
    "shekel4_5": {
        "function": _f.shekel4_5,
        "properties": ["continuous", "differentiable", "non-separable", "multimodal", "non-convex"],
        "bounds": [(0, 10)],
        "default_dim": 4,
        "known_minimum": -10.1532,
        "optimal_point": [4.00004, 4.00013, 4.00004, 4.00013],
    },
    "shekel4_7": {
        "function": _f.shekel4_7,
        "properties": ["continuous", "differentiable", "non-separable", "multimodal", "non-convex"],
        "bounds": [(0, 10)],
        "default_dim": 4,
        "known_minimum": -10.4029,
        "optimal_point": [4.00057, 4.00069, 3.99949, 3.99961],
    },
    "shekel4_10": {
        "function": _f.shekel4_10,
        "properties": ["continuous", "differentiable", "non-separable", "multimodal", "non-convex"],
        "bounds": [(0, 10)],
        "default_dim": 4,
        "known_minimum": -10.5364,
        "optimal_point": [4.00075, 4.00059, 3.99966, 3.99951],
    },
    "shubert": {
        "function": _f.shubert,
        "properties": ["continuous", "differentiable", "separable", "multimodal", "non-convex"],
        "bounds": [(-10, 10)],
        "default_dim": 2,
        "known_minimum": -24.062499,
        "optimal_point": [5.791794, 5.791794],
    },
}


def get_function_info(name):
    """
    Get metadata for a specific function.

    Parameters
    ----------
    name : str
        Function name

    Returns
    -------
    dict
        Dictionary with function metadata including:
        - function: callable function
        - bounds: list of (min, max) tuples
        - default_dim: recommended dimension
        - known_minimum: known global minimum value
        - optimal_point: point(s) where minimum occurs

    Raises
    ------
    ValueError
        If function name not found in benchmark suite
    """
    if name not in BENCHMARK_SUITE:
        available = ", ".join(sorted(BENCHMARK_SUITE.keys()))
        raise ValueError(
            f"Function '{name}' not found in benchmark suite. Available functions: {available}"
        )
    return BENCHMARK_SUITE[name]


def get_all_functions():
    """
    Get list of all available function names.

    Returns
    -------
    list
        Sorted list of function names
    """
    return sorted(BENCHMARK_SUITE.keys())


def get_bounds(name, dim=None):
    """
    Get bounds for a function with specified dimension.

    Parameters
    ----------
    name : str
        Function name
    dim : int, optional
        Dimension (uses default_dim if None)

    Returns
    -------
    list
        List of (min, max) tuples for each dimension
    """
    info = get_function_info(name)
    if dim is None:
        dim = info["default_dim"]

    # If function has specific bounds per dimension, return them
    if len(info["bounds"]) > 1:
        return info["bounds"]

    # Otherwise replicate single bound for all dimensions
    return info["bounds"] * dim


def get_function_list():
    """
    Get formatted list of all functions with metadata.

    Returns
    -------
    str
        Formatted string listing all functions
    """
    lines = ["Available Benchmark Functions:", "=" * 70]
    for name in get_all_functions():
        info = BENCHMARK_SUITE[name]
        lines.append(
            f"{name:20s} | dim={info['default_dim']:2d} | min={info['known_minimum']:10.4f}"
        )
    lines.append("=" * 70)
    return "\n".join(lines)
