"""
Testing optimizer robustness with function wrappers.

NoisyFunction / ShiftedFunction / RotatedFunction let you perturb a benchmark
function's behavior without touching the optimizer under test — noisy
evaluations, a relocated optimum, and a rotated landscape are three classic
ways to check an algorithm isn't secretly relying on properties (determinism,
axis-alignment, optimum-at-the-origin) that a real objective won't have.
Needs only NumPy.
"""

import numpy as np

from bananabench import NoisyFunction, RotatedFunction, ShiftedFunction, rastrigin, sphere
from bananabench.wrappers import BenchmarkWrapper


def main():
    x = np.zeros(2)

    # 1. Noisy evaluations: same input, different output each call.
    noisy_sphere = NoisyFunction(sphere, noise_type="gaussian", scale=0.5, seed=1)
    print("NoisyFunction - repeated calls at the same point differ:")
    print(f"  {noisy_sphere(x):.4f}, {noisy_sphere(x):.4f}, {noisy_sphere(x):.4f}")

    # 2. Shifted optimum: sphere's minimum moves from (0,0) to (3,-2).
    shift = np.array([3.0, -2.0])
    shifted_sphere = ShiftedFunction(sphere, shift=shift)
    print(f"\nShiftedFunction - new optimum at {shift}:")
    print(f"  f(shift)  = {shifted_sphere(shift):.6f}  (should be ~0)")
    print(f"  f(origin) = {shifted_sphere(x):.6f}  (no longer the minimum)")

    # 3. Rotated landscape: a 45-degree rotation of rastrigin.
    theta = np.pi / 4
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated_rastrigin = RotatedFunction(rastrigin, matrix=rotation)
    probe = np.array([1.0, 0.0])
    print("\nRotatedFunction - same value under the corresponding rotated input:")
    print(f"  rotated_rastrigin({probe}) = {rotated_rastrigin(probe):.6f}")
    print(f"  rastrigin(rotation @ {probe}) = {rastrigin(rotation @ probe):.6f}")

    # 4. Writing your own wrapper: subclass BenchmarkWrapper for anything the
    #    built-ins don't cover — here, snapping the input to integers first.
    class DiscretizedFunction(BenchmarkWrapper):
        """Rounds input to the nearest integer before evaluating."""

        def __call__(self, x):
            return self.func(np.round(x))

    discrete_sphere = DiscretizedFunction(sphere)
    print("\nCustom BenchmarkWrapper - DiscretizedFunction:")
    print(f"  discrete_sphere([1.2, 2.8]) = {discrete_sphere([1.2, 2.8])}")
    print(f"  (evaluated sphere([1, 3]) = {sphere(np.array([1.0, 3.0]))})")


if __name__ == "__main__":
    main()
