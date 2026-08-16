"""
Gradient estimation wrappers for benchmark functions.

This module provides utilities to estimate gradients and Hessians
of black-box benchmark functions using finite differences.
"""

from typing import Callable

import numpy as np


def approximate_gradient(
    func: Callable[[np.ndarray], float],
    x: np.ndarray,
    method: str = "central",
    epsilon: float = 1e-8,
) -> np.ndarray:
    """
    Approximate the gradient of a function at point x using finite differences.

    Parameters
    ----------
    func : Callable
        The objective function.
    x : np.ndarray
        The point at which to evaluate the gradient.
    method : str
        The finite difference method: 'forward', 'backward', or 'central' (default).
    epsilon : float
        The step size for finite differences.

    Returns
    -------
    np.ndarray
        The estimated gradient vector.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    grad = np.zeros(n)

    if method == "central":
        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += epsilon
            x_minus = x.copy()
            x_minus[i] -= epsilon
            grad[i] = (func(x_plus) - func(x_minus)) / (2 * epsilon)

    elif method == "forward":
        fx = func(x)
        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += epsilon
            grad[i] = (func(x_plus) - fx) / epsilon

    elif method == "backward":
        fx = func(x)
        for i in range(n):
            x_minus = x.copy()
            x_minus[i] -= epsilon
            grad[i] = (fx - func(x_minus)) / epsilon

    else:
        raise ValueError(f"Unknown finite difference method: {method}")

    return grad


class GradientWrapper:
    """
    A wrapper class that equips a black-box function with a gradient method.
    Useful for optimization algorithms in SciPy that require a Jacobian.
    """

    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        method: str = "central",
        epsilon: float = 1e-8,
    ):
        self.func = func
        self.method = method
        self.epsilon = epsilon
        self.evaluations = 0
        self.grad_evaluations = 0

    def __call__(self, x: np.ndarray) -> float:
        self.evaluations += 1
        return self.func(x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        self.grad_evaluations += 1
        return approximate_gradient(self.func, x, method=self.method, epsilon=self.epsilon)

    def fun_and_grad(self, x: np.ndarray):
        """Return both function value and gradient, typical for scipy.optimize."""
        return self.__call__(x), self.gradient(x)
