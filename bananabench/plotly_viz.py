"""
Interactive Plotly visualizations for optimization benchmark functions.

This module provides 3D surface plots and contour plots using Plotly,
allowing for interactive exploration of the fitness landscapes.
"""

from typing import Callable, Tuple

import numpy as np

try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def _generate_grid(
    func: Callable[[np.ndarray], float],
    bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    resolution: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate X, Y, Z grid for plotting."""
    x = np.linspace(bounds[0][0], bounds[0][1], resolution)
    y = np.linspace(bounds[1][0], bounds[1][1], resolution)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))

    return X, Y, Z


def plot_surface_interactive(
    func: Callable[[np.ndarray], float],
    bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    resolution: int = 100,
    title: str = "Function Surface",
    log_scale: bool = False,
    show: bool = True,
):
    """
    Create an interactive 3D surface plot using Plotly.

    Parameters
    ----------
    func : Callable
        The 2D function to plot.
    bounds : tuple
        Bounds for x and y axes: ((xmin, xmax), (ymin, ymax)).
    resolution : int
        Number of grid points per axis.
    title : str
        Plot title.
    log_scale : bool
        If True, plot log10(Z - min(Z) + 1) for better visualization of flat areas.
    show : bool
        If True, displays the figure immediately.

    Returns
    -------
    plotly.graph_objects.Figure
        The generated Plotly figure.
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is required for interactive visualizations. Run `pip install plotly`."
        )

    X, Y, Z = _generate_grid(func, bounds, resolution)

    if log_scale:
        Z_min = np.min(Z)
        Z_plot = np.log10(Z - Z_min + 1)
        z_title = "log10(f(x,y) - min + 1)"
    else:
        Z_plot = Z
        z_title = "f(x,y)"

    fig = go.Figure(data=[go.Surface(z=Z_plot, x=X, y=Y, colorscale="Viridis")])

    fig.update_layout(
        title=title,
        autosize=True,
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title=z_title,
        ),
        margin=dict(l=65, r=50, b=65, t=90),
    )

    if show:
        fig.show()

    return fig


def plot_contour_interactive(
    func: Callable[[np.ndarray], float],
    bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    resolution: int = 100,
    title: str = "Function Contour",
    log_scale: bool = False,
    show: bool = True,
):
    """
    Create an interactive 2D contour plot using Plotly.
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is required for interactive visualizations. Run `pip install plotly`."
        )

    X, Y, Z = _generate_grid(func, bounds, resolution)

    if log_scale:
        Z_min = np.min(Z)
        Z_plot = np.log10(Z - Z_min + 1)
    else:
        Z_plot = Z

    fig = go.Figure(
        data=[
            go.Contour(
                z=Z_plot, x=X[0], y=Y[:, 0], colorscale="Viridis", contours=dict(showlines=False)
            )
        ]
    )

    fig.update_layout(title=title, xaxis_title="x", yaxis_title="y", autosize=True)

    if show:
        fig.show()

    return fig
