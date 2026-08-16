"""Shared pytest configuration for the banana-bench test suite."""

try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend so plotting tests never open a GUI window.
except ImportError:
    pass
