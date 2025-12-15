"""Utilities for integrating external content and adding panel labels.

This module provides tools for:
- Adding external plots (from R, MATLAB, etc.) as images to panels
- Adding panel labels (A, B, C, ...) with precise positioning
"""

import matplotlib.pyplot as plt
from matplotlib.image import imread
from typing import Literal, Optional, Tuple
from pathlib import Path


class ExternalPanel:
    """Utilities for adding external plot images to panels.

    This class provides static methods for displaying images from external
    plotting tools (R, MATLAB, etc.) in matplotlib axes.
    """

    @staticmethod
    def add_image_panel(
        ax: plt.Axes,
        image_path: str,
        aspect: str = 'equal',
        hide_axes: bool = True
    ):
        """Add an external image to a matplotlib axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to add the image to
        image_path : str
            Path to the image file
        aspect : str, default 'equal'
            Aspect ratio for the image display
        hide_axes : bool, default True
            Whether to hide axis ticks and labels

        Examples
        --------
        >>> fig, ax = plt.subplots()
        >>> ExternalPanel.add_image_panel(ax, 'external_plot.png')
        >>> plt.show()
        """
        # Read and display image
        img = imread(image_path)
        ax.imshow(img, aspect=aspect)

        # Hide axes if requested
        if hide_axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

    @staticmethod
    def create_placeholder(
        ax: plt.Axes,
        text: str = 'External Plot',
        fontsize: int = 14,
        color: str = 'gray'
    ):
        """Create a placeholder for external content.

        Useful during figure development when external plots are not yet ready.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to add the placeholder to
        text : str, default 'External Plot'
            Placeholder text to display
        fontsize : int, default 14
            Font size for placeholder text
        color : str, default 'gray'
            Color for placeholder elements

        Examples
        --------
        >>> fig, ax = plt.subplots()
        >>> ExternalPanel.create_placeholder(ax, 'R plot goes here')
        >>> plt.show()
        """
        # Draw placeholder box
        ax.add_patch(plt.Rectangle((0.1, 0.1), 0.8, 0.8,
                                   fill=False, edgecolor=color,
                                   linestyle='--', linewidth=2,
                                   transform=ax.transAxes))

        # Add placeholder text
        ax.text(0.5, 0.5, text,
               horizontalalignment='center',
               verticalalignment='center',
               fontsize=fontsize,
               color=color,
               transform=ax.transAxes)

        # Hide axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)


class PanelLabeler:
    """Utilities for adding panel labels (A, B, C, ...) to multi-panel figures.

    Parameters
    ----------
    fontsize_pt : float, default 12
        Font size for labels in points
    location : {'top_left', 'top_right', 'bottom_left', 'bottom_right'}, default 'top_left'
        Location for panel labels
    offset : tuple of float, default (-0.1, 1.05)
        Offset for label position in axes coordinates (x, y)
    fontweight : str, default 'bold'
        Font weight for labels
    fontfamily : str, default 'sans-serif'
        Font family for labels

    Examples
    --------
    >>> labeler = PanelLabeler(fontsize_pt=10, location='top_left')
    >>> fig, axes = plt.subplots(2, 2)
    >>> for idx, (ax, label) in enumerate(zip(axes.flat, ['A', 'B', 'C', 'D'])):
    ...     labeler.add_label(ax, label, dpi=300)
    >>> plt.show()
    """

    def __init__(
        self,
        fontsize_pt: float = 12,
        location: Literal['top_left', 'top_right', 'bottom_left', 'bottom_right'] = 'top_left',
        offset: Optional[Tuple[float, float]] = None,
        fontweight: str = 'bold',
        fontfamily: str = 'sans-serif'
    ):
        self.fontsize_pt = fontsize_pt
        self.location = location
        self.fontweight = fontweight
        self.fontfamily = fontfamily

        # Set default offset based on location if not provided
        if offset is None:
            offset_map = {
                'top_left': (-0.1, 1.05),
                'top_right': (1.05, 1.05),
                'bottom_left': (-0.1, -0.1),
                'bottom_right': (1.05, -0.1)
            }
            self.offset = offset_map[location]
        else:
            self.offset = offset

        # Set alignment based on location
        alignment_map = {
            'top_left': ('right', 'bottom'),
            'top_right': ('left', 'bottom'),
            'bottom_left': ('right', 'top'),
            'bottom_right': ('left', 'top')
        }
        self.ha, self.va = alignment_map[location]

    def add_label(
        self,
        ax: plt.Axes,
        label: str,
        dpi: int = 300,
        **text_kwargs
    ):
        """Add a panel label to an axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to label
        label : str
            Label text (e.g., 'A', 'B', 'C')
        dpi : int, default 300
            DPI of the figure (used for size calculations)
        **text_kwargs
            Additional keyword arguments passed to ax.text()

        Examples
        --------
        >>> fig, ax = plt.subplots()
        >>> labeler = PanelLabeler()
        >>> labeler.add_label(ax, 'A', dpi=300)
        """
        # Merge user kwargs with defaults
        text_params = {
            'transform': ax.transAxes,
            'fontsize': self.fontsize_pt,
            'fontweight': self.fontweight,
            'fontfamily': self.fontfamily,
            'horizontalalignment': self.ha,
            'verticalalignment': self.va
        }
        text_params.update(text_kwargs)

        # Add label
        ax.text(self.offset[0], self.offset[1], label, **text_params)

    def add_labels_to_dict(
        self,
        axes_dict: dict,
        dpi: int = 300,
        **text_kwargs
    ):
        """Add labels to all axes in a dictionary.

        Convenience method for labeling all panels in a PanelLayout figure.

        Parameters
        ----------
        axes_dict : dict of matplotlib.axes.Axes
            Dictionary mapping panel names to axes
        dpi : int, default 300
            DPI of the figure
        **text_kwargs
            Additional keyword arguments passed to ax.text()

        Examples
        --------
        >>> layout = PanelLayout(units='cm', dpi=300)
        >>> layout.add_panel('A', size=(8, 6))
        >>> layout.add_panel('B', size=(8, 6))
        >>> layout.set_grid(rows=1, cols=2)
        >>> fig, axes = layout.create_figure()
        >>> labeler = PanelLabeler()
        >>> labeler.add_labels_to_dict(axes, dpi=layout.dpi)
        """
        for name, ax in axes_dict.items():
            self.add_label(ax, name, dpi=dpi, **text_kwargs)


def format_panel_label(
    index: int,
    style: Literal['upper', 'lower', 'number'] = 'upper'
) -> str:
    """Format panel label from index.

    Parameters
    ----------
    index : int
        Zero-based panel index
    style : {'upper', 'lower', 'number'}, default 'upper'
        Label style:
        - 'upper': Uppercase letters (A, B, C, ...)
        - 'lower': Lowercase letters (a, b, c, ...)
        - 'number': Numbers (1, 2, 3, ...)

    Returns
    -------
    str
        Formatted panel label

    Examples
    --------
    >>> format_panel_label(0, 'upper')
    'A'
    >>> format_panel_label(1, 'lower')
    'b'
    >>> format_panel_label(2, 'number')
    '3'
    """
    if style == 'upper':
        return chr(65 + index)  # A, B, C, ...
    elif style == 'lower':
        return chr(97 + index)  # a, b, c, ...
    elif style == 'number':
        return str(index + 1)  # 1, 2, 3, ...
    else:
        raise ValueError(f"Unknown style: {style}. Must be 'upper', 'lower', or 'number'")
