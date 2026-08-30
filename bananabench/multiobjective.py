"""
Multi-objective ZDT test suite.

Implements the five real-valued ZDT problems (ZDT1-ZDT4, ZDT6) from
Zitzler, Deb & Thiele (2000), "Comparison of Multiobjective Evolutionary
Algorithms: Empirical Results," Evolutionary Computation, 8(2), 173-195 --
the standard bi-objective benchmark suite for multi-objective optimizers.
ZDT5 is deliberately not included: it is binary-encoded (a 30-bit plus ten
5-bit segments genome), which does not fit the ``x: np.ndarray -> np.ndarray``
convention every other function in this package follows.

Each function takes a real-valued decision vector and returns a
2-element objective vector ``[f1, f2]`` (both minimized), rather than the
single float scalar functions in ``functions.py`` return.

``ZDT_SUITE`` mirrors ``metadata.BENCHMARK_SUITE`` in spirit but is kept as
its own registry rather than folded into it: ``BENCHMARK_SUITE`` entries
assume a scalar ``known_minimum``/``optimal_point``, which doesn't apply to
a Pareto *front*. Use ``get_pareto_front`` for the true front (needed by
quality indicators like hypervolume/IGD) and ``get_mo_function_list`` for
the registry's keys.

Every problem's Pareto-optimal front is derived directly from its own
formula rather than transcribed from the paper: each ZDT problem's ``g``
term is minimized at ``x[1:] = 0`` (verifiable from the formula -- e.g.
ZDT4's ``x_i**2 - 10*cos(4*pi*x_i)`` term is minimized at ``x_i = 0`` for
every ``i``), so sweeping ``x1`` across its bounds with ``x[1:] = 0`` and
keeping the non-dominated subset of the resulting curve reconstructs the
true front. This also correctly handles ZDT3 (disconnected front) and
ZDT6 (front doesn't span all of f1's domain) without needing to hand-copy
interval boundaries from the source paper.
"""

from typing import Callable, Dict, List, Optional, Tuple, TypedDict

import numpy as np


class _ZDTEntry(TypedDict):
    function: Callable[[np.ndarray], np.ndarray]
    n_objectives: int
    default_dim: int
    bounds: Callable[[int], List[Tuple[float, float]]]
    properties: List[str]


def _check_min_dim(x: np.ndarray, name: str) -> None:
    if x.size < 2:
        raise ValueError(f"{name} requires at least 2 dimensions.")


def zdt1(x: np.ndarray) -> np.ndarray:
    """
    ZDT1: convex Pareto front.
    Dimension: scalable (default 30), bounds [0, 1]^n.
    Objectives: minimize f1(x), f2(x).
    """
    x = np.asarray(x, dtype=float)
    _check_min_dim(x, "zdt1")
    n = x.size
    f1 = x[0]
    g = 1.0 + 9.0 * np.sum(x[1:]) / (n - 1)
    h = 1.0 - np.sqrt(f1 / g)
    f2 = g * h
    return np.array([f1, f2])


def zdt2(x: np.ndarray) -> np.ndarray:
    """
    ZDT2: non-convex Pareto front.
    Dimension: scalable (default 30), bounds [0, 1]^n.
    Objectives: minimize f1(x), f2(x).
    """
    x = np.asarray(x, dtype=float)
    _check_min_dim(x, "zdt2")
    n = x.size
    f1 = x[0]
    g = 1.0 + 9.0 * np.sum(x[1:]) / (n - 1)
    h = 1.0 - (f1 / g) ** 2
    f2 = g * h
    return np.array([f1, f2])


def zdt3(x: np.ndarray) -> np.ndarray:
    """
    ZDT3: disconnected Pareto front (5 discrete segments).
    Dimension: scalable (default 30), bounds [0, 1]^n.
    Objectives: minimize f1(x), f2(x).
    """
    x = np.asarray(x, dtype=float)
    _check_min_dim(x, "zdt3")
    n = x.size
    f1 = x[0]
    g = 1.0 + 9.0 * np.sum(x[1:]) / (n - 1)
    h = 1.0 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1)
    f2 = g * h
    return np.array([f1, f2])


def zdt4(x: np.ndarray) -> np.ndarray:
    """
    ZDT4: many local Pareto fronts (multimodal g).
    Dimension: scalable (default 10), bounds x1 in [0, 1], x2..xn in [-5, 5].
    Objectives: minimize f1(x), f2(x).
    """
    x = np.asarray(x, dtype=float)
    _check_min_dim(x, "zdt4")
    n = x.size
    f1 = x[0]
    g = 1.0 + 10.0 * (n - 1) + np.sum(x[1:] ** 2 - 10.0 * np.cos(4 * np.pi * x[1:]))
    h = 1.0 - np.sqrt(f1 / g)
    f2 = g * h
    return np.array([f1, f2])


def zdt6(x: np.ndarray) -> np.ndarray:
    """
    ZDT6: non-convex, non-uniformly spaced Pareto front.
    Dimension: scalable (default 10), bounds [0, 1]^n.
    Objectives: minimize f1(x), f2(x).
    """
    x = np.asarray(x, dtype=float)
    _check_min_dim(x, "zdt6")
    n = x.size
    f1 = 1.0 - np.exp(-4.0 * x[0]) * np.sin(6 * np.pi * x[0]) ** 6
    g = 1.0 + 9.0 * (np.sum(x[1:]) / (n - 1)) ** 0.25
    h = 1.0 - (f1 / g) ** 2
    f2 = g * h
    return np.array([f1, f2])


def _zdt4_bounds(dim: int) -> List[Tuple[float, float]]:
    return [(0.0, 1.0)] + [(-5.0, 5.0)] * (dim - 1)


def _uniform_unit_bounds(dim: int) -> List[Tuple[float, float]]:
    return [(0.0, 1.0)] * dim


ZDT_SUITE: Dict[str, _ZDTEntry] = {
    "zdt1": {
        "function": zdt1,
        "n_objectives": 2,
        "default_dim": 30,
        "bounds": _uniform_unit_bounds,
        "properties": [
            "multi-objective",
            "continuous",
            "differentiable",
            "scalable",
            "convex-front",
        ],
    },
    "zdt2": {
        "function": zdt2,
        "n_objectives": 2,
        "default_dim": 30,
        "bounds": _uniform_unit_bounds,
        "properties": [
            "multi-objective",
            "continuous",
            "differentiable",
            "scalable",
            "non-convex-front",
        ],
    },
    "zdt3": {
        "function": zdt3,
        "n_objectives": 2,
        "default_dim": 30,
        "bounds": _uniform_unit_bounds,
        "properties": [
            "multi-objective",
            "continuous",
            "differentiable",
            "scalable",
            "disconnected-front",
        ],
    },
    "zdt4": {
        "function": zdt4,
        "n_objectives": 2,
        "default_dim": 10,
        "bounds": _zdt4_bounds,
        "properties": [
            "multi-objective",
            "continuous",
            "differentiable",
            "scalable",
            "multimodal",
            "convex-front",
        ],
    },
    "zdt6": {
        "function": zdt6,
        "n_objectives": 2,
        "default_dim": 10,
        "bounds": _uniform_unit_bounds,
        "properties": [
            "multi-objective",
            "continuous",
            "differentiable",
            "scalable",
            "non-convex-front",
        ],
    },
}


def get_mo_function_list() -> List[str]:
    """Return the names of every function in ``ZDT_SUITE``."""
    return list(ZDT_SUITE.keys())


def get_mo_bounds(name: str, dim: Optional[int] = None) -> List[Tuple[float, float]]:
    """Return normalized ``[(min, max), ...]`` bounds for a ZDT problem at a given dimension."""
    info = ZDT_SUITE[name]
    resolved_dim = dim if dim is not None else info["default_dim"]
    return info["bounds"](resolved_dim)


def non_dominated_front(points: np.ndarray) -> np.ndarray:
    """
    Filter ``points`` (shape ``(n, n_objectives)``, all objectives minimized)
    down to its Pareto-non-dominated subset.
    """
    points = np.asarray(points, dtype=float)
    n = points.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominates_i = np.all(points <= points[i], axis=1) & np.any(points < points[i], axis=1)
        dominates_i[i] = False
        if np.any(dominates_i):
            keep[i] = False
    return points[keep]


def get_pareto_front(name: str, n_points: int = 200) -> np.ndarray:
    """
    Return a sample of the true Pareto-optimal front for a ZDT problem, sorted
    by the first objective.

    Constructed by evaluating the function itself along its minimal-``g``
    manifold (``x[1:] = 0``, which every ZDT problem's ``g`` term is
    minimized at) and keeping only the non-dominated points -- see the
    module docstring for why this is done analytically rather than
    transcribed from the source paper.
    """
    info = ZDT_SUITE[name]
    func = info["function"]
    dim = info["default_dim"]
    bounds = get_mo_bounds(name, dim)
    x1_lo, x1_hi = bounds[0]

    x1_values = np.linspace(x1_lo, x1_hi, n_points)
    rest = np.zeros(dim - 1)
    raw_front = np.array([func(np.concatenate(([x1], rest))) for x1 in x1_values])

    front = non_dominated_front(raw_front)
    return front[np.argsort(front[:, 0])]


def igd(front: np.ndarray, reference_front: np.ndarray) -> float:
    """
    Inverted Generational Distance: mean distance from each point of the true
    ``reference_front`` to its nearest point in the approximation ``front``.
    Lower is better; 0.0 means every reference point was reached exactly.
    """
    front = np.asarray(front, dtype=float)
    reference_front = np.asarray(reference_front, dtype=float)
    if front.size == 0:
        return float("inf")
    dists = np.linalg.norm(reference_front[:, None, :] - front[None, :, :], axis=2)
    return float(np.mean(dists.min(axis=1)))


def hypervolume(front: np.ndarray, reference_point: np.ndarray) -> float:
    """
    2D hypervolume indicator: the area dominated by ``front`` and bounded by
    ``reference_point`` (both objectives minimized). Higher is better.

    Only 2-objective fronts are supported for now -- every ZDT problem is
    bi-objective. General n-objective hypervolume (needed once DTLZ/WFG are
    added) is a materially harder algorithm (WFG/HSO-style slicing) and is
    deliberately left out of this first slice rather than half-implemented.
    """
    front = np.asarray(front, dtype=float)
    reference_point = np.asarray(reference_point, dtype=float)
    if front.shape[1] != 2:
        raise NotImplementedError("hypervolume() currently supports 2-objective fronts only.")

    dominating = front[np.all(front < reference_point, axis=1)]
    if dominating.size == 0:
        return 0.0
    front = non_dominated_front(dominating)
    order = np.argsort(front[:, 0])
    front = front[order]

    hv = 0.0
    prev_f2 = reference_point[1]
    for f1, f2 in front:
        hv += (reference_point[0] - f1) * (prev_f2 - f2)
        prev_f2 = f2
    return hv
