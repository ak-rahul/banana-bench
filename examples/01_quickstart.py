"""
Quickstart: evaluating functions and reading their metadata.

Covers the two things you'll do in almost every session: calling a benchmark
function directly, and looking up its bounds/dimension/known minimum through
the metadata registry instead of hardcoding them. Needs only NumPy.
"""

import numpy as np

from bananabench import ackley, get_all_functions, get_bounds, get_function_info


def main():
    print(f"{len(get_all_functions())} benchmark functions available:")
    print(get_all_functions()[:10], "...")

    # Call a function directly once you know its dimension.
    x = np.zeros(5)
    print(f"\nackley(zeros(5)) = {ackley(x):.6f}")

    # Or look everything up by name instead of hardcoding bounds/dimension.
    info = get_function_info("ackley")
    print(f"\nackley metadata:")
    print(f"  default dimension : {info['default_dim']}")
    print(f"  known minimum     : {info['known_minimum']}")
    print(f"  optimal point     : {info['optimal_point']}")
    print(f"  properties        : {info['properties']}")

    # get_bounds() handles both "one bound replicated across dimensions" (like
    # ackley's) and "one explicit (min, max) per dimension" (like branin's) —
    # don't replicate info['bounds'] by hand, it isn't safe for the latter case.
    bounds = get_bounds("ackley")
    print(f"  bounds (10D)      : {bounds[:2]} ... ({len(bounds)} total)")

    x_at_min = np.full(info["default_dim"], info["optimal_point"][0])
    print(f"\nackley at its optimal point = {ackley(x_at_min):.2e}")


if __name__ == "__main__":
    main()
