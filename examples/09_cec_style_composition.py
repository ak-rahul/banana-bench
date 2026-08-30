"""
CEC-style composition on a user's own function, not just the bundled suite.

`cec.CECFunction`/`CECCompositionFunction` already let you build a CEC
2017/2020-style shift/rotate/compose problem out of *any* callable that takes
an ndarray and returns a float -- including a research function you wrote
yourself, not just `bananabench`'s bundled suite. This is a lighter-weight
alternative to waiting for real CEC2013/2014/2017 data-file suites to ship
(still gated on ROADMAP.md's unresolved shift/rotation data-file strategy):
you get the same shift/rotate/compose *mechanics* today, on any function.
"""

import numpy as np

from bananabench import sphere
from bananabench.cec import CECCompositionFunction, CECFunction


def my_research_function(x: np.ndarray) -> float:
    """A stand-in for a user's own objective function -- not part of
    bananabench at all. Simple asymmetric bowl, minimum at the origin."""
    x = np.asarray(x, dtype=float)
    return float(np.sum(x**2) + 0.5 * np.sum(x))


def main():
    # 1. Shift + rotate + bias a user's own function, exactly like you would
    #    any bundled benchmark.
    shift = np.array([3.0, -2.0])
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])  # 90-degree rotation
    shifted_research_func = CECFunction(
        my_research_function, shift_vector=shift, rotation_matrix=rotation, bias=10.0
    )
    print("CECFunction wrapping a user-defined function:")
    print(f"  f(shift)      = {shifted_research_func(shift):.4f}   (should equal bias: 10.0)")
    print(f"  f([0, 0])     = {shifted_research_func(np.zeros(2)):.4f}")

    # 2. Compose that user function with a bundled benchmark (sphere) into a
    #    single CEC-style composition -- the framework doesn't care whether
    #    each component came from the package or from the user.
    component_a = CECFunction(my_research_function, shift_vector=np.zeros(2))
    component_b = CECFunction(sphere, shift_vector=np.array([50.0, 50.0]))
    composition = CECCompositionFunction(
        [component_a, component_b],
        sigma=np.array([2.0, 2.0]),
        biases=np.array([0.0, 100.0]),
    )

    print("\nCECCompositionFunction mixing a user function with a bundled one:")
    near_a = np.array([0.01, 0.0])
    near_b = np.array([50.01, 50.0])
    print(f"  near component_a's optimum: {composition(near_a):.4f}  (dominated by component_a)")
    print(f"  near component_b's optimum: {composition(near_b):.4f}  (dominated by component_b)")


if __name__ == "__main__":
    main()
