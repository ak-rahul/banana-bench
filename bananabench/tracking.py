"""
Evaluation tracking and budgets.

``TrackedFunction`` wraps any objective and records what a benchmark needs to
compare optimizers fairly: how many evaluations were spent, the best value and
point seen so far, and an anytime "best-so-far" trace. With ``max_evals`` it also
enforces an evaluation budget by raising :class:`BudgetExhausted` instead of
performing evaluation ``max_evals + 1``.

Unlike the classes in :mod:`bananabench.wrappers`, which deliberately perturb a
function (noise, shift, rotation), a ``TrackedFunction`` is purely observational:
it returns exactly ``func(x)``. It composes with them, but the order matters:
``TrackedFunction(NoisyFunction(f))`` counts and records the *noisy* values.

Counting evaluations and comparing algorithms at equal budgets follows standard
benchmarking practice; see Hansen, Auger, Ros, Mersmann, Tušar & Brockhoff (2021),
"COCO: A Platform for Comparing Continuous Optimizers in a Black-Box Setting",
Optimization Methods and Software 36(1):114-144, and Moré & Wild (2009),
"Benchmarking Derivative-Free Optimization Algorithms", SIAM J. Optim.
20(1):172-191. The trace records the first evaluation, which data profiles need as
f(x0), and every strict improvement after it, so memory grows with the number of
improvements rather than the number of evaluations.

License: MIT
"""

from typing import Any, Callable, List, Optional, Tuple

import numpy as np

__all__ = ["BudgetExhausted", "TrackedFunction"]


class BudgetExhausted(Exception):
    """Raised by :class:`TrackedFunction` when a call would exceed ``max_evals``.

    The evaluation that would exceed the budget is *not* performed, so a tracker
    with ``max_evals=n`` never evaluates its function more than ``n`` times.
    """

    def __init__(self, max_evals: int) -> None:
        super().__init__(f"Evaluation budget of {max_evals} exhausted.")
        self.max_evals = max_evals


class TrackedFunction:
    """
    Count evaluations, enforce a budget, and record the best-so-far trace.

    Parameters
    ----------
    func : callable
        Objective ``f(x) -> float`` to track.
    max_evals : int, optional
        Evaluation budget. Calling the tracker once it has performed
        ``max_evals`` evaluations raises :class:`BudgetExhausted`. ``None``
        (default) means unlimited.
    record_trace : bool, default=True
        Record ``(evaluation_index, best_so_far)`` at the first evaluation and
        at every strict improvement.

    Attributes
    ----------
    n_evals : int
        Number of evaluations performed so far.
    best_f : float
        Best finite value seen (``inf`` until a finite value is seen).
    best_x : ndarray or None
        Copy of the input that produced ``best_f``.
    first_f : float or None
        Value returned by the first evaluation (``f(x0)`` for data profiles).

    Examples
    --------
    >>> import numpy as np
    >>> from bananabench import sphere
    >>> from bananabench.tracking import TrackedFunction
    >>> f = TrackedFunction(sphere, max_evals=100)
    >>> float(f(np.array([1.0, 2.0])))
    5.0
    >>> f.n_evals, f.best_f
    (1, 5.0)

    Notes
    -----
    NaN and infinite values count as evaluations but never as improvements.
    The tracker copies ``x`` before storing it, so optimizers that later modify
    their arrays in place can't change ``best_x``.
    """

    def __init__(
        self,
        func: Callable[[np.ndarray], Any],
        max_evals: Optional[int] = None,
        record_trace: bool = True,
    ) -> None:
        if max_evals is not None and max_evals < 1:
            raise ValueError(f"max_evals must be a positive integer or None, got {max_evals!r}.")
        self.func = func
        self.max_evals = max_evals
        self.record_trace = record_trace
        self.reset()

    def reset(self) -> None:
        """Forget all evaluations, the best point and the trace."""
        self.n_evals = 0
        self.best_f = float("inf")
        self.best_x: Optional[np.ndarray] = None
        self.first_f: Optional[float] = None
        self._trace: List[Tuple[int, float]] = []

    @property
    def budget_left(self) -> Optional[int]:
        """Evaluations remaining, or ``None`` when there is no budget."""
        if self.max_evals is None:
            return None
        return self.max_evals - self.n_evals

    @property
    def trace(self) -> np.ndarray:
        """Best-so-far trace as an array of shape ``(k, 2)``.

        Column 0 is the 1-based evaluation index, column 1 the best finite value
        seen up to and including that evaluation. The first row is the first
        evaluation (its value may be non-finite); every later row is a strict
        improvement.
        """
        if not self._trace:
            return np.empty((0, 2))
        return np.array(self._trace, dtype=float)

    def __call__(self, x: Any) -> Any:
        if self.max_evals is not None and self.n_evals >= self.max_evals:
            raise BudgetExhausted(self.max_evals)

        value = self.func(x)
        self.n_evals += 1

        scalar = float(value)
        if self.first_f is None:
            self.first_f = scalar
        improved = np.isfinite(scalar) and scalar < self.best_f
        if improved:
            self.best_f = scalar
            self.best_x = np.array(x, dtype=float, copy=True)
        if self.record_trace and (improved or self.n_evals == 1):
            self._trace.append((self.n_evals, self.best_f if improved else scalar))
        return value

    def __repr__(self) -> str:
        name = getattr(self.func, "__name__", type(self.func).__name__)
        budget = "unlimited" if self.max_evals is None else str(self.max_evals)
        return (
            f"TrackedFunction({name}, n_evals={self.n_evals}, max_evals={budget}, "
            f"best_f={self.best_f!r})"
        )
