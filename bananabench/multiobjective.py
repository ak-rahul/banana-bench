"""
Multi-objective test suites: ZDT (bi-objective) and DTLZ (any number of objectives).

**ZDT.** The five real-valued ZDT problems (ZDT1-ZDT4, ZDT6) from Zitzler, Deb &
Thiele (2000), "Comparison of Multiobjective Evolutionary Algorithms: Empirical
Results," Evolutionary Computation, 8(2), 173-195 -- the standard bi-objective
benchmark suite. ZDT5 is deliberately not included: it is binary-encoded (a 30-bit
plus ten 5-bit segments genome), which does not fit the ``x: np.ndarray ->
np.ndarray`` convention every other function in this package follows.

**DTLZ.** DTLZ1-DTLZ7 from Deb, Thiele, Laumanns & Zitzler (2005), "Scalable Test
Problems for Evolutionary Multiobjective Optimization," in *Evolutionary
Multiobjective Optimization*, Springer, pp. 105-145 (conference version: Proc. IEEE
CEC 2002, pp. 825-830). Each takes ``n_objectives`` (M, default 3) and has
``n = M + k - 1`` variables, with the paper's ``k = 5`` (DTLZ1), ``10`` (DTLZ2-6)
and ``20`` (DTLZ7). Every formula was cross-checked against two independent
implementations: pymoo (``pymoo/problems/many/dtlz.py``; agreement to <= 2.4e-15
relative error at random points for M = 2, 3, 4) and Platypus (DTLZ1-4, DTLZ7).

Each function returns an objective vector (all minimized) rather than the single
float the scalar functions in ``functions.py`` return. The suites have their own
registries (``ZDT_SUITE``, ``DTLZ_SUITE``, and ``MO_SUITE`` for both) rather than
entries in ``metadata.BENCHMARK_SUITE``, which assumes a scalar
``known_minimum``/``optimal_point`` that doesn't apply to a Pareto *front*.

True Pareto fronts (``get_pareto_front``) are derived from each problem's own
structure rather than transcribed:

- ZDT: every ``g`` term is minimized at ``x[1:] = 0``, so sweeping ``x1`` with
  ``x[1:] = 0`` and keeping the non-dominated subset reconstructs the front,
  including ZDT3's disconnected front and ZDT6's partial one.
- DTLZ1-4: Das-Dennis points (Das & Dennis 1998) mapped onto the known front
  (the ``sum(f) = 0.5`` simplex, or the unit sphere).
- DTLZ5/6: the degenerate curve traced by ``theta_1`` at ``g = 0``.
- DTLZ7: the non-dominated subset of a grid on the ``g = 1`` manifold.

``hypervolume`` is exact for any number of objectives (a sweep for 2, slicing for
3, the WFG algorithm of While, Bradstreet & Barone 2012 for more), with a seeded
Monte Carlo estimate for large fronts; it was cross-checked against pymoo's
independent implementation for 2-5 objectives.
"""

import functools
import itertools
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict

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


# ---------------------------------------------------------------------------
# DTLZ suite (scalable number of objectives)
# ---------------------------------------------------------------------------


def _dtlz_split(x: np.ndarray, n_objectives: int, name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Split ``x`` into position variables ``x[:M-1]`` and distance variables ``x[M-1:]``."""
    if n_objectives < 2:
        raise ValueError(f"{name} needs at least 2 objectives, got {n_objectives}.")
    if x.size < n_objectives:
        raise ValueError(
            f"{name} with {n_objectives} objectives needs at least {n_objectives} variables "
            f"(k = n - M + 1 >= 1), got {x.size}."
        )
    return x[: n_objectives - 1], x[n_objectives - 1 :]


def _g_rastrigin(xm: np.ndarray) -> float:
    """DTLZ1/DTLZ3 distance function: multimodal, minimized (g = 0) at xm = 0.5."""
    return float(100.0 * (xm.size + np.sum((xm - 0.5) ** 2 - np.cos(20.0 * np.pi * (xm - 0.5)))))


def _g_sphere(xm: np.ndarray) -> float:
    """DTLZ2/DTLZ4/DTLZ5 distance function, minimized (g = 0) at xm = 0.5."""
    return float(np.sum((xm - 0.5) ** 2))


def _sphere_objectives(angles: np.ndarray, g: float) -> np.ndarray:
    """f_i = (1 + g) * prod_{j < M-1-i} cos(angle_j) * [sin(angle_{M-1-i}) if i > 0]."""
    m = angles.size + 1
    f = np.full(m, 1.0 + g)
    for i in range(m):
        f[i] *= np.prod(np.cos(angles[: m - 1 - i]))
        if i > 0:
            f[i] *= np.sin(angles[m - 1 - i])
    return f


def _dtlz56_angles(xp: np.ndarray, g: float) -> np.ndarray:
    """DTLZ5/DTLZ6 angle mapping: theta_1 = x_1 * pi/2 and, for i >= 2,
    theta_i = pi / (4 (1 + g)) * (1 + 2 g x_i), which collapses the front to a curve."""
    angles = np.pi / (4.0 * (1.0 + g)) * (1.0 + 2.0 * g * xp)
    angles[0] = xp[0] * np.pi / 2.0
    return angles


def dtlz1(x: np.ndarray, n_objectives: int = 3) -> np.ndarray:
    """
    DTLZ1: linear Pareto front sum(f) = 0.5, with a highly multimodal g
    (11^k - 1 local fronts).
    Dimension: n = M + k - 1, default k = 5. Bounds [0, 1]^n.
    Pareto-optimal: x_M = 0.5 (all distance variables).
    """
    x = np.asarray(x, dtype=float)
    xp, xm = _dtlz_split(x, n_objectives, "dtlz1")
    g = _g_rastrigin(xm)
    m = n_objectives
    f = np.full(m, 0.5 * (1.0 + g))
    for i in range(m):
        f[i] *= np.prod(xp[: m - 1 - i])
        if i > 0:
            f[i] *= 1.0 - xp[m - 1 - i]
    return f


def dtlz2(x: np.ndarray, n_objectives: int = 3) -> np.ndarray:
    """
    DTLZ2: spherical Pareto front sum(f^2) = 1 in the positive orthant.
    Dimension: n = M + k - 1, default k = 10. Bounds [0, 1]^n.
    Pareto-optimal: x_M = 0.5.
    """
    x = np.asarray(x, dtype=float)
    xp, xm = _dtlz_split(x, n_objectives, "dtlz2")
    return _sphere_objectives(xp * np.pi / 2.0, _g_sphere(xm))


def dtlz3(x: np.ndarray, n_objectives: int = 3) -> np.ndarray:
    """
    DTLZ3: DTLZ2's spherical front with DTLZ1's multimodal g (3^k - 1 local fronts).
    Dimension: n = M + k - 1, default k = 10. Bounds [0, 1]^n.
    Pareto-optimal: x_M = 0.5.
    """
    x = np.asarray(x, dtype=float)
    xp, xm = _dtlz_split(x, n_objectives, "dtlz3")
    return _sphere_objectives(xp * np.pi / 2.0, _g_rastrigin(xm))


def dtlz4(x: np.ndarray, n_objectives: int = 3, alpha: float = 100.0) -> np.ndarray:
    """
    DTLZ4: DTLZ2's spherical front with a biased density of solutions
    (position variables mapped through x^alpha, alpha = 100).
    Dimension: n = M + k - 1, default k = 10. Bounds [0, 1]^n.
    Pareto-optimal: x_M = 0.5.
    """
    x = np.asarray(x, dtype=float)
    xp, xm = _dtlz_split(x, n_objectives, "dtlz4")
    return _sphere_objectives(xp**alpha * np.pi / 2.0, _g_sphere(xm))


def dtlz5(x: np.ndarray, n_objectives: int = 3) -> np.ndarray:
    """
    DTLZ5: degenerate Pareto front (a curve on the unit sphere) for M >= 3.
    Dimension: n = M + k - 1, default k = 10. Bounds [0, 1]^n.
    Pareto-optimal: x_M = 0.5.
    """
    x = np.asarray(x, dtype=float)
    xp, xm = _dtlz_split(x, n_objectives, "dtlz5")
    g = _g_sphere(xm)
    return _sphere_objectives(_dtlz56_angles(xp, g), g)


def dtlz6(x: np.ndarray, n_objectives: int = 3) -> np.ndarray:
    """
    DTLZ6: DTLZ5's degenerate curve with a harder g = sum(x_M^0.1).
    Dimension: n = M + k - 1, default k = 10. Bounds [0, 1]^n.
    Pareto-optimal: x_M = 0.
    """
    x = np.asarray(x, dtype=float)
    xp, xm = _dtlz_split(x, n_objectives, "dtlz6")
    g = float(np.sum(xm**0.1))
    return _sphere_objectives(_dtlz56_angles(xp, g), g)


def dtlz7(x: np.ndarray, n_objectives: int = 3) -> np.ndarray:
    """
    DTLZ7: disconnected Pareto front (2^(M-1) regions).
    f_i = x_i for i < M; g = 1 + 9/k * sum(x_M); h = M - sum_{i<M} f_i/(1+g) (1 + sin(3 pi f_i));
    f_M = (1 + g) h.
    Dimension: n = M + k - 1, default k = 20. Bounds [0, 1]^n.
    Pareto-optimal: x_M = 0 (g = 1).
    """
    x = np.asarray(x, dtype=float)
    xp, xm = _dtlz_split(x, n_objectives, "dtlz7")
    g = 1.0 + 9.0 / xm.size * float(np.sum(xm))
    h = n_objectives - float(np.sum(xp / (1.0 + g) * (1.0 + np.sin(3.0 * np.pi * xp))))
    return np.concatenate([xp, [(1.0 + g) * h]])


class _DTLZEntry(TypedDict):
    function: Callable[..., np.ndarray]
    n_objectives: int
    k: int
    default_dim: int
    bounds: Callable[[int], List[Tuple[float, float]]]
    optimal_distance_value: float
    properties: List[str]


def _dtlz_entry(func, k, optimal_distance_value, properties) -> _DTLZEntry:
    return {
        "function": func,
        "n_objectives": 3,
        "k": k,
        "default_dim": 3 + k - 1,
        "bounds": _uniform_unit_bounds,
        "optimal_distance_value": optimal_distance_value,
        "properties": ["multi-objective", "scalable", "scalable-objectives", *properties],
    }


DTLZ_SUITE: Dict[str, _DTLZEntry] = {
    "dtlz1": _dtlz_entry(dtlz1, 5, 0.5, ["continuous", "multimodal", "linear-front"]),
    "dtlz2": _dtlz_entry(dtlz2, 10, 0.5, ["continuous", "differentiable", "concave-front"]),
    "dtlz3": _dtlz_entry(dtlz3, 10, 0.5, ["continuous", "multimodal", "concave-front"]),
    "dtlz4": _dtlz_entry(
        dtlz4, 10, 0.5, ["continuous", "differentiable", "concave-front", "biased"]
    ),
    "dtlz5": _dtlz_entry(dtlz5, 10, 0.5, ["continuous", "differentiable", "degenerate-front"]),
    "dtlz6": _dtlz_entry(dtlz6, 10, 0.0, ["continuous", "degenerate-front"]),
    "dtlz7": _dtlz_entry(dtlz7, 20, 0.0, ["continuous", "disconnected-front"]),
}

#: Every multi-objective problem: the ZDT suite (bi-objective) plus DTLZ
#: (any number of objectives, default 3).
MO_SUITE: Dict[str, Any] = {**ZDT_SUITE, **DTLZ_SUITE}


def get_mo_default_dim(name: str, n_objectives: Optional[int] = None) -> int:
    """Default dimension of a multi-objective problem (``M + k - 1`` for DTLZ)."""
    info = MO_SUITE[name]
    if name in DTLZ_SUITE and n_objectives is not None:
        return int(n_objectives + info["k"] - 1)
    return int(info["default_dim"])


def get_mo_n_objectives(name: str, n_objectives: Optional[int] = None) -> int:
    """Number of objectives: fixed at 2 for ZDT, ``n_objectives`` (default 3) for DTLZ."""
    if name in DTLZ_SUITE:
        return int(n_objectives) if n_objectives is not None else 3
    if n_objectives not in (None, 2):
        raise ValueError(f"{name} is bi-objective; n_objectives must be 2, got {n_objectives}.")
    return 2


def get_mo_function(
    name: str, n_objectives: Optional[int] = None
) -> Callable[[np.ndarray], np.ndarray]:
    """The objective function, with ``n_objectives`` bound for DTLZ problems."""
    func = MO_SUITE[name]["function"]
    if name in DTLZ_SUITE:
        return functools.partial(func, n_objectives=get_mo_n_objectives(name, n_objectives))
    get_mo_n_objectives(name, n_objectives)  # validates
    return func


def das_dennis(n_objectives: int, n_partitions: int) -> np.ndarray:
    """
    Das–Dennis uniformly spaced points on the unit simplex.

    Every vector whose components are non-negative multiples of
    ``1 / n_partitions`` and sum to 1: ``C(n_partitions + M - 1, M - 1)`` points.
    The standard way to place reference points/weights for many-objective
    problems (Das & Dennis 1998, SIAM J. Optim. 8(3):631-657; used by NSGA-III,
    Deb & Jain 2014).
    """
    if n_objectives < 1 or n_partitions < 0:
        raise ValueError("n_objectives must be >= 1 and n_partitions >= 0.")
    if n_objectives == 1:
        return np.ones((1, 1))
    points = []
    # Stars and bars: choose positions of M-1 bars among H + M - 1 slots.
    for bars in itertools.combinations(range(n_partitions + n_objectives - 1), n_objectives - 1):
        edges = (-1, *bars, n_partitions + n_objectives - 1)
        points.append([edges[i + 1] - edges[i] - 1 for i in range(n_objectives)])
    return np.asarray(points, dtype=float) / max(n_partitions, 1)


def _partitions_for(n_objectives: int, n_points: int) -> int:
    """Smallest Das–Dennis partition count giving at least ``n_points`` points."""
    h = 1
    while math.comb(h + n_objectives - 1, n_objectives - 1) < n_points:
        h += 1
    return h


def _dtlz_pareto_front(name: str, n_points: int, n_objectives: int) -> np.ndarray:
    m = n_objectives
    if name == "dtlz1":
        return 0.5 * das_dennis(m, _partitions_for(m, n_points))
    if name in ("dtlz2", "dtlz3", "dtlz4"):
        w = das_dennis(m, _partitions_for(m, n_points))
        return w / np.linalg.norm(w, axis=1, keepdims=True)
    if name in ("dtlz5", "dtlz6"):
        # At g = 0 every theta_i (i >= 2) equals pi/4: the front is the curve traced by theta_1.
        theta1 = np.linspace(0.0, np.pi / 2.0, n_points)
        angles = np.column_stack([theta1, np.full((n_points, m - 2), np.pi / 4.0)])
        return np.array([_sphere_objectives(a, 0.0) for a in angles])
    if name == "dtlz7":
        # Sample the g = 1 manifold on a grid, then keep the non-dominated points;
        # this recovers the 2^(M-1) disconnected regions without transcribing them.
        per_axis = max(2, int(np.ceil((8 * n_points) ** (1.0 / (m - 1)))))
        axis = np.linspace(0.0, 1.0, per_axis)
        grid = np.array(list(itertools.product(axis, repeat=m - 1)))
        last = 2.0 * m - np.sum(grid * (1.0 + np.sin(3.0 * np.pi * grid)), axis=1)
        return non_dominated_front(np.column_stack([grid, last]))
    raise KeyError(name)


def get_mo_function_list() -> List[str]:
    """Return the names of every multi-objective problem (ZDT, then DTLZ)."""
    return list(MO_SUITE.keys())


def get_mo_bounds(
    name: str, dim: Optional[int] = None, n_objectives: Optional[int] = None
) -> List[Tuple[float, float]]:
    """Return ``[(min, max), ...]`` bounds for a multi-objective problem at a given dimension."""
    info = MO_SUITE[name]
    resolved_dim = dim if dim is not None else get_mo_default_dim(name, n_objectives)
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


def get_pareto_front(
    name: str, n_points: int = 200, n_objectives: Optional[int] = None
) -> np.ndarray:
    """
    Return a sample of the true Pareto-optimal front, sorted by the first objective.

    - ZDT: built by evaluating the function along its minimal-``g`` manifold
      (``x[1:] = 0``) and keeping the non-dominated points (see the module
      docstring for why this is derived rather than transcribed).
    - DTLZ1–4: Das–Dennis reference points mapped onto the known front (the
      ``sum(f) = 0.5`` simplex for DTLZ1, the unit sphere for DTLZ2–4), about
      ``n_points`` of them.
    - DTLZ5/6: ``n_points`` points along the degenerate curve.
    - DTLZ7: the non-dominated subset of a grid on the ``g = 1`` manifold.
    """
    if name in DTLZ_SUITE:
        front = _dtlz_pareto_front(name, n_points, get_mo_n_objectives(name, n_objectives))
        return front[np.lexsort(front.T[::-1])]

    get_mo_n_objectives(name, n_objectives)  # validates
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


# ---------------------------------------------------------------------------
# Hypervolume
# ---------------------------------------------------------------------------

# method="auto" is always exact for 2-3 objectives (polynomial sweeps); for 4+ it uses
# WFG up to these sizes and Monte Carlo beyond them (WFG's cost grows quickly).
_HV_EXACT_MAX_OBJECTIVES = 5
_HV_EXACT_MAX_POINTS = 100


def _hv_2d(points: np.ndarray, ref: np.ndarray) -> float:
    """Exact 2-objective sweep over non-dominated points (O(n log n))."""
    points = points[np.argsort(points[:, 0])]
    hv = 0.0
    prev_f2 = ref[1]
    for f1, f2 in points:
        if f2 < prev_f2:
            hv += (ref[0] - f1) * (prev_f2 - f2)
            prev_f2 = f2
    return float(hv)


def _hv_3d(points: np.ndarray, ref: np.ndarray) -> float:
    """Exact 3-objective slicing: sweep f3 upward, multiplying each slab's height by the
    2-D hypervolume of every point at or below it (O(n^2 log n))."""
    points = points[np.argsort(points[:, 2])]
    total = 0.0
    for i in range(points.shape[0]):
        upper = points[i + 1, 2] if i + 1 < points.shape[0] else ref[2]
        height = upper - points[i, 2]
        if height > 0:
            total += _hv_2d(points[: i + 1, :2], ref[:2]) * height
    return float(total)


def _wfg(points: np.ndarray, ref: np.ndarray) -> float:
    """
    WFG exact hypervolume (While, Bradstreet & Barone 2012):
    HV(S) = sum_k [ inclhv(p_k) - HV(nondom(limitset(S, k))) ], where
    limitset(S, k) = { max(p_k, p_j) : j > k } (componentwise "worse" point).
    Points are sorted by the last objective, which keeps limit sets small.
    """
    n, m = points.shape
    if n == 0:
        return 0.0
    if n == 1:
        return float(np.prod(ref - points[0]))
    if m == 2:
        return _hv_2d(points, ref)
    if m == 3:
        return _hv_3d(points, ref)
    points = points[np.argsort(points[:, -1])]
    total = 0.0
    for k in range(n):
        p = points[k]
        incl = float(np.prod(ref - p))
        rest = points[k + 1 :]
        if rest.shape[0] == 0:
            total += incl
            continue
        limit = non_dominated_front(np.maximum(rest, p))
        total += incl - _wfg(limit, ref)
    return total


def _hv_monte_carlo(
    points: np.ndarray, ref: np.ndarray, n_samples: int, seed: Optional[int]
) -> Tuple[float, float]:
    """Uniform sampling in the box [ideal, ref]: returns (estimate, standard error)."""
    ideal = points.min(axis=0)
    box_volume = float(np.prod(ref - ideal))
    rng = np.random.default_rng(seed)
    hits = 0
    remaining = n_samples
    while remaining > 0:
        chunk = min(remaining, 20_000)
        samples = rng.uniform(ideal, ref, size=(chunk, points.shape[1]))
        dominated = np.zeros(chunk, dtype=bool)
        for p in points:
            dominated |= np.all(samples >= p, axis=1)
        hits += int(dominated.sum())
        remaining -= chunk
    fraction = hits / n_samples
    return box_volume * fraction, box_volume * np.sqrt(fraction * (1 - fraction) / n_samples)


def hypervolume(
    front: np.ndarray,
    reference_point: np.ndarray,
    method: str = "auto",
    n_samples: int = 100_000,
    seed: Optional[int] = None,
) -> float:
    """
    Hypervolume indicator: the volume dominated by ``front`` and bounded by
    ``reference_point`` (all objectives minimized). Higher is better.

    Parameters
    ----------
    front : array_like, shape (n_points, n_objectives)
    reference_point : array_like, shape (n_objectives,)
        Points that don't strictly dominate it contribute nothing.
    method : {"auto", "exact", "monte_carlo"}, default="auto"
        ``"exact"`` uses a sweep for 2 objectives, slicing along the third
        objective for 3, and the WFG algorithm (While, Bradstreet & Barone 2012,
        "A Fast Way of Calculating Exact Hypervolumes", IEEE TEC 16(1):86-95)
        for more, with the 3-objective sweep as its base case. ``"monte_carlo"``
        estimates it by uniform sampling, with standard error
        ``V * sqrt(p (1 - p) / n_samples)`` for box volume ``V`` and hit rate
        ``p``. ``"auto"`` is exact for 2 and 3 objectives, and for 4–5 objectives
        up to 100 non-dominated points; Monte Carlo otherwise.
    n_samples : int, default=100000
        Samples for the Monte Carlo estimate.
    seed : int, optional
        Seed for the Monte Carlo estimate (reproducible when given).

    Returns
    -------
    float
    """
    front = np.asarray(front, dtype=float)
    reference_point = np.asarray(reference_point, dtype=float)
    if front.ndim != 2 or front.shape[1] != reference_point.size:
        raise ValueError(
            "front must have shape (n_points, n_objectives) matching reference_point's length."
        )
    if method not in ("auto", "exact", "monte_carlo"):
        raise ValueError(f"Unknown method {method!r}; use 'auto', 'exact' or 'monte_carlo'.")

    dominating = front[np.all(front < reference_point, axis=1)]
    if dominating.size == 0:
        return 0.0
    points = np.unique(non_dominated_front(dominating), axis=0)
    m = points.shape[1]

    if method == "auto":
        exact = m <= 3 or (m <= _HV_EXACT_MAX_OBJECTIVES and len(points) <= _HV_EXACT_MAX_POINTS)
        method = "exact" if exact else "monte_carlo"
    if method == "exact":
        return _wfg(points, reference_point)
    return float(_hv_monte_carlo(points, reference_point, n_samples, seed)[0])
