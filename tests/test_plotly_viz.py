"""
Tests for bananabench.plotly_viz: interactive Plotly surface/contour plots.

Optional dependency: uses pytest.importorskip so these tests skip cleanly
when plotly isn't installed, and monkeypatch to exercise the "plotly missing"
error path deterministically regardless of what's actually installed.
"""

import numpy as np
import pytest

from bananabench import sphere


def test_raises_import_error_without_plotly(monkeypatch):
    from bananabench import plotly_viz

    monkeypatch.setattr(plotly_viz, "PLOTLY_AVAILABLE", False)
    with pytest.raises(ImportError):
        plotly_viz.plot_surface_interactive(sphere, ((-1, 1), (-1, 1)), resolution=5, show=False)
    with pytest.raises(ImportError):
        plotly_viz.plot_contour_interactive(sphere, ((-1, 1), (-1, 1)), resolution=5, show=False)


def test_surface_and_contour_when_plotly_available():
    pytest.importorskip("plotly")
    from bananabench import plotly_viz

    assert plotly_viz.PLOTLY_AVAILABLE is True

    fig = plotly_viz.plot_surface_interactive(sphere, ((-1, 1), (-1, 1)), resolution=5, show=False)
    assert fig.data[0].type == "surface"

    fig = plotly_viz.plot_contour_interactive(sphere, ((-1, 1), (-1, 1)), resolution=5, show=False)
    assert fig.data[0].type == "contour"


def test_surface_values_match_function():
    pytest.importorskip("plotly")
    from bananabench import plotly_viz

    fig = plotly_viz.plot_surface_interactive(sphere, ((-2, 2), (-2, 2)), resolution=5, show=False)
    z = np.array(fig.data[0].z)
    assert np.isclose(np.min(z), 0.0, atol=1e-6)  # sphere's minimum is inside these bounds


def test_log_scale_option_produces_finite_output():
    pytest.importorskip("plotly")
    from bananabench import plotly_viz

    fig = plotly_viz.plot_surface_interactive(
        sphere, ((-2, 2), (-2, 2)), resolution=5, log_scale=True, show=False
    )
    z = np.array(fig.data[0].z)
    assert np.all(np.isfinite(z))
