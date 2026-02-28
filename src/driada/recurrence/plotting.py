"""Visualization for recurrence analysis."""

import matplotlib.pyplot as plt


def plot_recurrence(rg, ax=None, markersize=0.5, color='black', **kwargs):
    """Plot a recurrence matrix as a spy plot.

    Uses matplotlib's spy() for efficient rendering of sparse binary matrices.

    Parameters
    ----------
    rg : RecurrenceGraph or Network
        Object with ``.adj`` sparse adjacency matrix.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, creates new figure.
    markersize : float, default=0.5
        Size of recurrence dots.
    color : str, default='black'
        Color of recurrence dots.
    **kwargs
        Additional keyword arguments passed to ``ax.spy()``.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        Figure (None if ax was provided).
    ax : matplotlib.axes.Axes
        Axes with the plot.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    ax.spy(rg.adj, markersize=markersize, color=color, **kwargs)
    ax.set_xlabel('Time index')
    ax.set_ylabel('Time index')
    ax.set_title('Recurrence Plot')

    return fig, ax
