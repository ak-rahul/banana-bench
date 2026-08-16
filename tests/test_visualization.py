"""
Tests for bananabench.visualization (optional matplotlib-based module).

All tests are skipped when matplotlib is not installed. tests/conftest.py
forces the non-interactive "Agg" backend so nothing tries to open a GUI
window during the run.
"""

import os
import shutil

import numpy as np
import pytest

try:
    import matplotlib.pyplot as plt

    from bananabench.visualization import (
        COLORMAPS,
        animate_trajectory_2d,
        batch_plot_functions,
        plot_algorithm_comparison,
        plot_benchmark_summary,
        plot_convergence,
        plot_function_2d,
        plot_function_3d,
        plot_search_heatmap,
        plot_trajectory_2d,
        save_plot,
    )

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

pytestmark = pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not installed")


def test_module_imports_cleanly_without_matplotlib(monkeypatch):
    """Regression test: visualization.py must not raise NameError (only the
    intended ImportError) when matplotlib is unavailable. Before
    `from __future__ import annotations` was added, importing this module
    with matplotlib absent crashed the *entire* bananabench package, not just
    matplotlib-dependent functions — see CHANGELOG.md.
    """
    import importlib
    import sys

    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    monkeypatch.setitem(sys.modules, "mpl_toolkits.mplot3d", None)

    from bananabench import visualization

    try:
        reloaded = importlib.reload(visualization)
        assert reloaded.MATPLOTLIB_AVAILABLE is False
        with pytest.raises(ImportError, match="matplotlib is required"):
            reloaded.plot_function_2d("sphere")
    finally:
        # monkeypatch's own teardown runs *after* this function returns (including
        # after this finally block), so sys.modules["matplotlib"] is still faked
        # to None here. Undo it explicitly first, or this reload "restores" the
        # module while matplotlib still looks absent, permanently corrupting
        # MATPLOTLIB_AVAILABLE = False for every later test in the session.
        monkeypatch.undo()
        importlib.reload(visualization)


class TestFunctionPlots:
    def test_plot_function_2d(self):
        fig = plot_function_2d("sphere", resolution=20, show_optimum=True)
        assert fig is not None
        plt.close(fig)

    def test_plot_function_3d(self):
        fig = plot_function_3d("ackley", resolution=10)
        assert fig is not None
        plt.close(fig)


class TestConvergenceAndTrajectory:
    def test_plot_convergence(self):
        history = [10, 5, 2, 1, 0.5, 0.1]
        fig = plot_convergence(history, function_name="sphere", known_minimum=0.0)
        assert fig is not None
        plt.close(fig)

    def test_plot_trajectory_2d(self):
        trajectory = np.array([[5, 5], [3, 3], [1, 1], [0, 0]])
        fig = plot_trajectory_2d("sphere", trajectory)
        assert fig is not None
        plt.close(fig)

    def test_animate_trajectory_2d_returns_animation(self):
        from matplotlib.animation import FuncAnimation

        trajectory = np.array([[5.0, 5.0], [3.0, 3.0], [1.0, 1.0], [0.0, 0.0]])
        anim = animate_trajectory_2d("sphere", trajectory, resolution=10)
        assert isinstance(anim, FuncAnimation)
        plt.close("all")

    def test_animate_trajectory_2d_rejects_non_2d_trajectory(self):
        trajectory = np.array([[5.0, 5.0, 1.0], [3.0, 3.0, 1.0]])
        with pytest.raises(ValueError):
            animate_trajectory_2d("sphere", trajectory)


class TestComparisonAndSummary:
    def test_plot_algorithm_comparison(self):
        results = {
            "Algo1": {"sphere": {"error": 0.01, "time": 1.0}},
            "Algo2": {"sphere": {"error": 0.05, "time": 1.5}},
        }
        fig = plot_algorithm_comparison(results, metric="error")
        assert fig is not None
        plt.close(fig)

    def test_plot_benchmark_summary(self):
        results = [
            {"function": "sphere", "error": 0.01, "time": 1.0},
            {"function": "ackley", "error": 0.05, "time": 1.5},
        ]
        fig = plot_benchmark_summary(results)
        assert fig is not None
        plt.close(fig)


class TestSearchHeatmap:
    def test_returns_figure(self):
        points = np.random.uniform(-5, 5, (50, 2))
        fig = plot_search_heatmap("sphere", points, bins=10)
        assert fig is not None
        plt.close(fig)


class TestSavePlot:
    def test_saves_single_format(self, tmp_path):
        fig = plot_function_2d("sphere")
        output = tmp_path / "test_plot"
        save_plot(fig, str(output), formats=["png"])
        assert (tmp_path / "test_plot.png").exists()
        plt.close(fig)

    def test_saves_multiple_formats(self, tmp_path):
        fig = plot_function_2d("sphere")
        output = tmp_path / "multi_plot"
        save_plot(fig, str(output), formats=["png", "svg"])
        assert (tmp_path / "multi_plot.png").exists()
        assert (tmp_path / "multi_plot.svg").exists()
        plt.close(fig)


class TestBatchPlotting:
    def test_generates_plots_for_every_function(self, tmp_path):
        output_dir = tmp_path / "batch_plots"
        results = batch_plot_functions(
            function_names=["sphere", "ackley"],
            plot_types=["2d"],
            output_dir=str(output_dir),
            formats=["png"],
        )
        assert set(results) == {"sphere", "ackley"}
        assert all(len(files) == 1 for files in results.values())
        for files in results.values():
            for filepath in files:
                assert os.path.exists(filepath)


class TestColormaps:
    def test_colormaps_constant_is_nonempty_dict(self):
        assert isinstance(COLORMAPS, dict)
        assert len(COLORMAPS) > 0

    @pytest.mark.parametrize("cmap", ["viridis", "plasma", "inferno", "coolwarm"])
    def test_plot_function_3d_accepts_each_colormap(self, cmap):
        fig = plot_function_3d("sphere", cmap=cmap, resolution=10)
        assert fig is not None
        plt.close(fig)
