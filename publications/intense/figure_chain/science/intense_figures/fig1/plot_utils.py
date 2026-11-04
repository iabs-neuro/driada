"""Plotting utilities for INTENSE figures."""

import numpy as np
import matplotlib.pyplot as plt


def plot_shadowed_groups(ax, xvals, binary_series, color='gray', alpha=0.3,
                         label='shadowed', ymin=0.0, ymax=1.0):
    """Plot shadowed regions for groups of consecutive 1s in a binary series.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    xvals : array-like
        X-axis values corresponding to binary_series
    binary_series : array-like
        Binary array (0s and 1s) indicating regions to shadow
    color : str, default 'gray'
        Color for shadowed regions
    alpha : float, default 0.3
        Transparency for shadowed regions
    label : str, default 'shadowed'
        Label for legend (only applied to first region)
    ymin : float, default 0.0
        Lower y extent in axes coordinates (0–1).
    ymax : float, default 1.0
        Upper y extent in axes coordinates (0–1).

    Returns
    -------
    ax : matplotlib.axes.Axes
        Modified axes
    """
    x = np.arange(len(binary_series))

    # Find and shadow groups of 1s
    i = 0
    n = len(binary_series)
    labelled = False
    while i < n:
        if binary_series[i] == 1:
            x_min = xvals[i]  # Start of group
            while i < n and binary_series[i] == 1:
                i += 1
            x_max = xvals[i - 1]  # End of group
            # Shadow the region
            ax.axvspan(x_min, x_max + 1, ymin=ymin, ymax=ymax, alpha=alpha,
                       color=color, label=label if not labelled else "")
            labelled = True
        else:
            i += 1

    return ax
