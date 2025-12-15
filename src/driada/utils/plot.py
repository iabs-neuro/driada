"""Plotting utilities for DRIADA.

Provides functions for creating publication-quality figures with
consistent styling.
"""

import matplotlib.pyplot as plt
from typing import Optional, Tuple, Any
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
    dpi: Optional[int] = None,
    lowercase_labels: bool = True,
    legend_frameon: bool = False,
    legend_loc: str = 'auto',
    legend_offset: float = 0.15,
    legend_ncol: Optional[int] = None,
    tight_layout: bool = True,
    remove_origin_tick: bool = False,
    panel_size: Optional[Tuple[float, float]] = None,
    panel_units: str = 'cm',
    reference_size: Tuple[float, float] = (8.0, 8.0),
):
    """Apply publication-quality styling to a matplotlib axis with optional auto-scaling.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to style.
    spine_width : float, optional
        Width of visible spines (default: 4).
    tick_width : float, optional
        Width of tick marks (default: 4).
    tick_length : float, optional
        Length of tick marks (default: 8).
    tick_pad : float, optional
        Padding between ticks and labels (default: 15).
    tick_labelsize : int, optional
        Font size for tick labels (default: 26).
    label_size : int, optional
        Font size for axis labels (default: 30).
    title_size : int, optional
        Font size for title (default: 30).
    legend_fontsize : int, optional
        Font size for legend (default: 18).
    dpi : int, optional
        DPI for the figure. If provided, sets the figure's DPI.
    lowercase_labels : bool, optional
        Whether to convert all labels and legend text to lowercase (default: True).
        This includes axis labels, title, tick labels, and legend entries.
    legend_frameon : bool, optional
        Whether to draw frame around legend (default: False).
    legend_loc : str, optional
        Legend location (default: 'auto'). Can be:
        - 'auto': Use matplotlib's automatic placement
        - 'above': Place legend above the plot, spanning full x-axis width
        - 'below': Place legend below the plot, spanning full x-axis width
        - Any valid matplotlib location string (e.g., 'upper right', 'center left')
    legend_offset : float, optional
        Vertical offset for 'above' and 'below' legend positions (default: 0.15).
        Positive values move the legend further from the plot.
    legend_ncol : int, optional
        Number of columns for legend entries (default: None, auto-determined).
        For 'above' and 'below', defaults to number of legend entries (single row).
    tight_layout : bool, optional
        Whether to remove extra margins on both axes (default: True).
    remove_origin_tick : bool, optional
        Whether to remove tick labels at the origin (0,0) to avoid overlap (default: False).
    panel_size : tuple of float, optional
        Physical size (width, height) of the panel. If provided, all size-related
        parameters (fonts, line widths) are automatically scaled based on panel
        area relative to reference_size. This maintains consistent visual density
        across panels of different sizes (default: None, no scaling).
    panel_units : {'cm', 'inches'}, default 'cm'
        Units for panel_size and reference_size (default: 'cm').
    reference_size : tuple of float, default (8.0, 8.0)
        Reference panel size for scaling calculations. Size parameters
        (spine_width, tick_width, etc.) are assumed to be appropriate for
        this reference size (default: (8.0, 8.0) cm).

    Returns
    -------
    matplotlib.axes.Axes
        The styled axis.
        
    Notes
    -----
    This function applies a consistent publication-quality style to matplotlib
    axes by:
    - Hiding top and right spines
    - Setting spine and tick widths
    - Configuring font sizes for all text elements
    - Setting figure DPI if requested
    
    The function modifies the axis in-place and returns it for convenience.
    
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> _ = ax.plot([1, 2, 3], [1, 4, 9])
    >>> _ = make_beautiful(ax)  # Apply styling
    >>> plt.show()
    
    >>> # With custom styling
    >>> fig, ax = plt.subplots()
    >>> _ = ax.plot([1, 2, 3], [1, 4, 9])
    >>> _ = make_beautiful(ax, spine_width=2, tick_labelsize=14)
    
    See Also
    --------
    ~driada.utils.plot.create_default_figure :
        Create figure with default styling applied.
    ~driada.utils.publication.PanelLayout :
        Layout manager for multi-panel figures with precise dimensions.
    ~driada.utils.publication.StylePreset :
        Style presets with automatic scaling.
    """
    # Calculate scale factor if panel_size is provided
    scale = 1.0
    if panel_size is not None:
        # Import here to avoid circular dependency
        from .publication.layout import to_inches

        # Convert sizes to inches for comparison
        ref_size_inches = to_inches(reference_size, panel_units)
        panel_size_inches = to_inches(panel_size, panel_units)

        # Calculate areas
        ref_area = ref_size_inches[0] * ref_size_inches[1]
        panel_area = panel_size_inches[0] * panel_size_inches[1]

        # Scale factor is sqrt of area ratio (preserves visual density)
        scale = np.sqrt(panel_area / ref_area)

    # Apply scale factor to all size parameters
    spine_width = spine_width * scale
    tick_width = tick_width * scale
    tick_length = tick_length * scale
    tick_pad = tick_pad * scale
    tick_labelsize = int(tick_labelsize * scale)
    label_size = int(label_size * scale)
    title_size = int(title_size * scale)
    legend_fontsize = int(legend_fontsize * scale)

    # Style spines
    for axis in ["bottom", "left"]:
        ax.spines[axis].set_linewidth(spine_width)

    for axis in ["top", "right"]:
        ax.spines[axis].set_linewidth(0.0)

    # Style ticks
    ax.tick_params(width=tick_width, direction="out", length=tick_length, pad=tick_pad)
    ax.tick_params(axis="x", which="major", labelsize=tick_labelsize)
    ax.tick_params(axis="y", which="major", labelsize=tick_labelsize)

    # Style labels
    ax.xaxis.label.set_size(label_size)
    ax.yaxis.label.set_size(label_size)

    # Convert labels to lowercase if requested
    if lowercase_labels:
        # Axis labels
        if ax.get_xlabel():
            ax.set_xlabel(ax.get_xlabel().lower())
        if ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel().lower())

        # Title
        if ax.get_title():
            ax.set_title(ax.get_title().lower())

        # Tick labels - use formatter to avoid warnings
        from matplotlib.ticker import FuncFormatter

        def lowercase_formatter(x, pos):
            """Format tick labels to lowercase."""
            # Round to 6 decimal places to avoid floating-point precision issues
            x_rounded = round(x, 6)
            # Format with appropriate precision
            if x_rounded == int(x_rounded):
                # Display as integer if it's a whole number
                return f"{int(x_rounded)}".lower()
            else:
                # Use general format that removes trailing zeros
                return f"{x_rounded:g}".lower()

        ax.xaxis.set_major_formatter(FuncFormatter(lowercase_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(lowercase_formatter))

    # Set axis-specific properties without modifying global params
    ax.title.set_size(title_size)
    if ax.legend_:
        # Determine number of columns for legend
        if legend_ncol is None:
            if legend_loc in ['above', 'below']:
                # Default to single row for above/below
                legend_ncol = len(ax.legend_.get_texts())
            else:
                # Default to single column for other positions
                legend_ncol = 1

        # Handle legend positioning
        if legend_loc == 'above':
            # Place legend above the plot
            ax.legend(bbox_to_anchor=(0.5, 1 + legend_offset), loc='lower center',
                     ncol=legend_ncol,
                     frameon=legend_frameon,
                     fontsize=legend_fontsize,
                     borderaxespad=0.0)
        elif legend_loc == 'below':
            # Place legend below the plot
            ax.legend(bbox_to_anchor=(0.5, -legend_offset), loc='upper center',
                     ncol=legend_ncol,
                     frameon=legend_frameon,
                     fontsize=legend_fontsize,
                     borderaxespad=0.0)
        elif legend_loc == 'auto':
            # Use matplotlib's automatic placement
            ax.legend(frameon=legend_frameon, fontsize=legend_fontsize, ncol=legend_ncol)
        else:
            # Use specified matplotlib location
            ax.legend(loc=legend_loc, frameon=legend_frameon, fontsize=legend_fontsize, ncol=legend_ncol)

        # Apply lowercase to legend text if needed
        if lowercase_labels and ax.legend_:
            for text in ax.legend_.get_texts():
                text.set_text(text.get_text().lower())

    # Remove extra margins on x and y axes
    if tight_layout:
        ax.margins(x=0, y=0)
        ax.autoscale(enable=True, axis='both', tight=True)

    # Remove tick at origin (axis intersection) if requested
    if remove_origin_tick:
        # Get current ticks
        xticks = list(ax.get_xticks())
        yticks = list(ax.get_yticks())

        # Remove 0 from both axes if present
        if 0.0 in xticks:
            xticks.remove(0.0)
            ax.set_xticks(xticks)
        if 0.0 in yticks:
            yticks.remove(0.0)
            ax.set_yticks(yticks)

    # Set DPI if provided
    if dpi is not None and hasattr(ax, "figure"):
        ax.figure.set_dpi(dpi)

    return ax


def create_default_figure(
    figsize: Tuple[float, float] = (16, 12),
    nrows: int = 1,
    ncols: int = 1,
    sharex: bool = False,
    sharey: bool = False,
    squeeze: bool = True,
    **style_kwargs,
) -> Tuple[plt.Figure, Any]:
    """Create a figure with default publication-quality styling.

    Parameters
    ----------
    figsize : tuple of float, optional
        Figure size as (width, height) in inches (default: (16, 12)).
    nrows : int, optional
        Number of rows in subplot grid (default: 1).
    ncols : int, optional
        Number of columns in subplot grid (default: 1).
    sharex : bool, optional
        Whether to share x-axis among subplots (default: False).
    sharey : bool, optional
        Whether to share y-axis among subplots (default: False).
    squeeze : bool, optional
        If True, extra dimensions are squeezed out from returned axes (default: True).
    **style_kwargs : dict
        Additional keyword arguments passed to make_beautiful().

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    axes : matplotlib.axes.Axes or array of Axes
        The styled axis/axes. If nrows=ncols=1 and squeeze=True, returns single Axes.
        Otherwise returns array of Axes.
        
    Notes
    -----
    This is a convenience function that combines matplotlib's subplots() with
    automatic application of publication-quality styling via make_beautiful().
    All axes in the figure receive the same styling.
    
    Examples
    --------
    >>> # Single subplot with default styling
    >>> fig, ax = create_default_figure()
    >>> _ = ax.plot([1, 2, 3], [1, 4, 9])
    >>> plt.show()
    
    >>> # Multiple subplots with custom styling
    >>> fig, axes = create_default_figure(nrows=2, ncols=2, figsize=(20, 16),
    ...                                   spine_width=2, tick_labelsize=14)
    >>> for ax in axes.flat:
    ...     _ = ax.plot(np.random.randn(100))
    
    See Also
    --------
    ~driada.utils.plot.make_beautiful :
        Apply styling to existing axes.
    matplotlib.pyplot.subplots :
        Base function for creating subplots.
    """
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        squeeze=squeeze,
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
    cmap: str = "viridis",
    aspect: str = "auto",
    **imshow_kwargs,
) -> Tuple[Optional[plt.Figure], plt.Axes]:
    """Plot a matrix as an image with optional colorbar.

    Parameters
    ----------
    mat : np.ndarray
        2D array to plot.
    figsize : tuple of float, optional
        Figure size if creating new figure (default: (12, 12)).
    ax : matplotlib.axes.Axes, optional
        Existing axis to plot on. If None, creates new figure.
    with_cbar : bool, optional
        Whether to add a colorbar (default: True).
    cmap : str, optional
        Colormap name (default: 'viridis').
    aspect : str, optional
        Aspect ratio setting (default: 'auto').
    **imshow_kwargs : dict
        Additional keyword arguments passed to ax.imshow().

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The figure (None if ax was provided).
    ax : matplotlib.axes.Axes
        The axis with the plot.
        
    Raises
    ------
    ValueError
        If mat is not a 2D array.
        
    Notes
    -----
    This function is a convenience wrapper around matplotlib's imshow for
    visualizing 2D matrices. It handles figure creation and colorbar addition
    automatically.
    
    The function returns both figure and axis to allow further customization.
    If an existing axis is provided, the figure return value will be None.
    
    Examples
    --------
    >>> # Plot a random matrix
    >>> mat = np.random.randn(10, 10)
    >>> fig, ax = plot_mat(mat)
    >>> _ = ax.set_title('Random Matrix')
    >>> plt.show()
    
    >>> # Plot on existing axis without colorbar
    >>> fig, ax = plt.subplots()
    >>> _, ax = plot_mat(mat, ax=ax, with_cbar=False, cmap='coolwarm')
    
    See Also
    --------
    matplotlib.pyplot.imshow : Base function for displaying images.
    matplotlib.pyplot.colorbar : Function for adding colorbars.    """
    # Validate input
    if mat.ndim != 2:
        raise ValueError(f"mat must be a 2D array, got {mat.ndim}D array")
    
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Set default imshow parameters
    imshow_params = {"cmap": cmap, "aspect": aspect}
    imshow_params.update(imshow_kwargs)

    im = ax.imshow(mat, **imshow_params)

    if with_cbar:
        cbar = ax.figure.colorbar(im, ax=ax)

    return fig, ax
