"""
CEC 2017 & 2020 Benchmark Suites Architecture.

This module provides the framework for building the complex, shifted,
rotated, and composite functions typical of CEC competitions.
"""

from typing import Callable, List, Optional

import numpy as np


class CECFunction:
    """
    Base class for constructing complex CEC-style benchmark functions
    with built-in support for shifting and rotating the search space.
    """

    def __init__(
        self,
        base_func: Callable[[np.ndarray], float],
        shift_vector: Optional[np.ndarray] = None,
        rotation_matrix: Optional[np.ndarray] = None,
        bias: float = 0.0,
    ):
        self.base_func = base_func
        self.shift_vector = shift_vector
        self.rotation_matrix = rotation_matrix
        self.bias = bias

    def __call__(self, x: np.ndarray) -> float:
        x_mapped = np.asarray(x, dtype=float).copy()

        # 1. Shift
        if self.shift_vector is not None:
            x_mapped = x_mapped - self.shift_vector

        # 2. Rotate
        if self.rotation_matrix is not None:
            x_mapped = np.dot(self.rotation_matrix, x_mapped)

        # 3. Evaluate and Bias
        return self.base_func(x_mapped) + self.bias


class CECCompositionFunction:
    """
    Evaluates a composition of multiple CEC functions using weighted aggregation,
    a hallmark of the CEC 2017 and 2020 suites.
    """

    def __init__(
        self, functions: List[Callable[[np.ndarray], float]], sigma: np.ndarray, biases: np.ndarray
    ):
        self.functions = functions
        self.sigma = sigma
        self.biases = biases

    def __call__(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        weights = np.zeros(len(self.functions))

        # Calculate raw weights based on distance to each function's optimum
        for i, func in enumerate(self.functions):
            # A component's optimum is its CECFunction.shift_vector, if it has one.
            # getattr's default only fires when the attribute is absent -- but every
            # CECFunction always sets self.shift_vector, defaulting to None rather
            # than omitting it -- so an explicit None check is needed too, or a
            # component built without a shift (the common case for "no-op shift")
            # crashes below on `x - None`.
            optimum = getattr(func, "shift_vector", None)
            if optimum is None:
                optimum = np.zeros_like(x)
            dist_sq = np.sum((x - optimum) ** 2)

            if dist_sq == 0:
                weights[i] = 1e99  # Exact match
            else:
                weights[i] = np.exp(-dist_sq / (2 * self.sigma[i] ** 2)) / np.sqrt(dist_sq)

        # Normalize weights
        w_sum = np.sum(weights)
        if w_sum > 0:
            weights = weights / w_sum
        else:
            weights = np.ones(len(self.functions)) / len(self.functions)

        # Aggregate
        result = 0.0
        for i, func in enumerate(self.functions):
            result += weights[i] * (func(x) + self.biases[i])

        return result
