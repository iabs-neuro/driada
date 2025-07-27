"""Plotting utilities for DRIADA.

Provides functions for creating publication-quality figures with
consistent styling.
"""

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from typing import Optional, Tuple, Dict, Any
import numpy as np


def make_beautiful(
    ax,
    spine_width: float = 4,
    tick_width: float = 4,
    tick_length: float = 8,
    tick_pad: float = 15,
    tick_labelsize: int = 26,
    label_size: int = 30,
    title_size: int = 30,
    legend_fontsize: int = 18,
    dpi: Optional[int] = None
):
    """Apply publication-quality styling to a matplotlib axis.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to style
    spine_width : float, optional
        Width of visible spines (default: 4)
    tick_width : float, optional
        Width of tick marks (default: 4)
    tick_length : float, optional
        Length of tick marks (default: 8)
    tick_pad : float, optional
        Padding between ticks and labels (default: 15)
    tick_labelsize : int, optional
        Font size for tick labels (default: 26)
    label_size : int, optional
        Font size for axis labels (default: 30)
    title_size : int, optional
        Font size for title (default: 30)
    legend_fontsize : int, optional
        Font size for legend (default: 18)
    dpi : int, optional
        DPI for the figure. If provided, sets the figure's DPI
        
    Returns
    -------
    matplotlib.axes.Axes
        The styled axis
    """
    # Style spines
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(spine_width)
        
    for axis in ['top', 'right']:
        ax.spines[axis].set_linewidth(0.0)
        
    # Style ticks
    ax.tick_params(
        width=tick_width,
        direction='in',
        length=tick_length,
        pad=tick_pad
    )
    ax.tick_params(axis='x', which='major', labelsize=tick_labelsize)
    ax.tick_params(axis='y', which='major', labelsize=tick_labelsize)
    
    # Style labels
    ax.xaxis.label.set_size(label_size)
    ax.yaxis.label.set_size(label_size)
    
    # Set axis-specific properties without modifying global params
    ax.title.set_size(title_size)
    if ax.legend_:
        for text in ax.legend_.get_texts():
            text.set_fontsize(legend_fontsize)
    
    # Set DPI if provided
    if dpi is not None and hasattr(ax, 'figure'):
        ax.figure.set_dpi(dpi)
    
    return ax


def create_default_figure(
    figsize: Tuple[float, float] = (16, 12),
    nrows: int = 1,
    ncols: int = 1,
    sharex: bool = False,
    sharey: bool = False,
    squeeze: bool = True,
    **style_kwargs
) -> Tuple[plt.Figure, Any]:
    """Create a figure with default publication-quality styling.
    
    Parameters
    ----------
    figsize : tuple of float, optional
        Figure size as (width, height) in inches (default: (16, 12))
    nrows : int, optional
        Number of rows in subplot grid (default: 1)
    ncols : int, optional
        Number of columns in subplot grid (default: 1)
    sharex : bool, optional
        Whether to share x-axis among subplots (default: False)
    sharey : bool, optional
        Whether to share y-axis among subplots (default: False)
    squeeze : bool, optional
        If True, extra dimensions are squeezed out from returned axes (default: True)
    **style_kwargs : dict
        Additional keyword arguments passed to make_beautiful()
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    axes : matplotlib.axes.Axes or array of Axes
        The styled axis/axes. If nrows=ncols=1 and squeeze=True, returns single Axes.
        Otherwise returns array of Axes.
    """
    fig, axes = plt.subplots(
        nrows=nrows, 
        ncols=ncols, 
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        squeeze=squeeze
    )
    
    # Apply styling to all axes
    if isinstance(axes, np.ndarray):
        for ax in axes.flat:
            make_beautiful(ax, **style_kwargs)
    else:
        axes = make_beautiful(axes, **style_kwargs)
    
    return fig, axes


def plot_mat(
    mat: np.ndarray,
    figsize: Tuple[float, float] = (12, 12),
    ax: Optional[plt.Axes] = None,
    with_cbar: bool = True,
    cmap: str = 'viridis',
    aspect: str = 'auto',
    **imshow_kwargs
) -> Tuple[Optional[plt.Figure], plt.Axes]:
    """Plot a matrix as an image with optional colorbar.
    
    Parameters
    ----------
    mat : np.ndarray
        2D array to plot
    figsize : tuple of float, optional
        Figure size if creating new figure (default: (12, 12))
    ax : matplotlib.axes.Axes, optional
        Existing axis to plot on. If None, creates new figure
    with_cbar : bool, optional
        Whether to add a colorbar (default: True)
    cmap : str, optional
        Colormap name (default: 'viridis')
    aspect : str, optional
        Aspect ratio setting (default: 'auto')
    **imshow_kwargs : dict
        Additional keyword arguments passed to ax.imshow()
        
    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The figure (None if ax was provided)
    ax : matplotlib.axes.Axes
        The axis with the plot
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Set default imshow parameters
    imshow_params = {
        'cmap': cmap,
        'aspect': aspect
    }
    imshow_params.update(imshow_kwargs)
    
    im = ax.imshow(mat, **imshow_params)
    
    if with_cbar:
        cbar = ax.figure.colorbar(im, ax=ax)
    
    return fig, ax
