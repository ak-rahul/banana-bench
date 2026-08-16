"""
Function wrappers for optimization benchmarks.

This module provides wrappers to modify the behavior of benchmark functions,
such as adding noise, shifting the optimum, or rotating the landscape.
"""

from typing import Callable, Optional

import numpy as np


class BenchmarkWrapper:
    """Base class for benchmark function wrappers."""

    def __init__(self, func: Callable[[np.ndarray], float]):
        self.func = func
        self.__doc__ = func.__doc__
        self.__name__ = func.__name__

    def __call__(self, x: np.ndarray) -> float:
        return self.func(x)


class NoisyFunction(BenchmarkWrapper):
    """
    Adds noise to the function evaluation.

    f_noisy(x) = f(x) + noise
    """

    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        noise_type: str = "gaussian",
        scale: float = 0.1,
        seed: Optional[int] = None,
    ):
        super().__init__(func)
        self.noise_type = noise_type
        self.scale = scale
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: np.ndarray) -> float:
        val = self.func(x)
        if self.noise_type == "gaussian":
            noise = self.rng.normal(0, self.scale)
        elif self.noise_type == "uniform":
            noise = self.rng.uniform(-self.scale, self.scale)
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")
        return val + noise


class ShiftedFunction(BenchmarkWrapper):
    """
    Shifts the function in the input space.

    f_shifted(x) = f(x - shift)

    This moves the global minimum by 'shift'.
    """

    def __init__(self, func: Callable[[np.ndarray], float], shift: np.ndarray):
        super().__init__(func)
        self.shift = np.asarray(shift, dtype=float)

    def __call__(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return self.func(x - self.shift)


class RotatedFunction(BenchmarkWrapper):
    """
    Rotates the function in the input space.

    f_rotated(x) = f(M @ x)

    where M is a rotation matrix.
    """

    def __init__(self, func: Callable[[np.ndarray], float], matrix: np.ndarray):
        super().__init__(func)
        self.matrix = np.asarray(matrix, dtype=float)

    def __call__(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        # Apply rotation x_rot = M @ x
        # Note: Depending on definition, might need x @ M.T
        # Standard convention: y = Mx
        return self.func(self.matrix @ x)
