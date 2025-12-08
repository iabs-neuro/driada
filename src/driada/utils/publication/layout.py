"""Layout management for publication-ready multi-panel figures.

This module provides classes and utilities for creating publication-quality
multi-panel figures with precise physical dimensions and consistent styling.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional, Tuple, Dict, Literal, Union
from dataclasses import dataclass
import numpy as np


# Unit conversion constants
CM_PER_INCH = 2.54


def to_inches(value: Union[float, Tuple[float, ...]], from_units: str) -> Union[float, Tuple[float, ...]]:
    """Convert from user units to inches (matplotlib's native unit).

    Parameters
    ----------
    value : float or tuple of float
        Value(s) to convert
    from_units : {'cm', 'inches'}
        Source units

    Returns
    -------
    float or tuple of float
        Value(s) in inches

    Raises
    ------
    ValueError
        If units are not 'cm' or 'inches'
    """
    if from_units == 'cm':
        if isinstance(value, tuple):
            return tuple(v / CM_PER_INCH for v in value)
        return value / CM_PER_INCH
    elif from_units == 'inches':
        return value
    else:
        raise ValueError(f"Unknown units: {from_units}. Must be 'cm' or 'inches'")


def from_inches(value: Union[float, Tuple[float, ...]], to_units: str) -> Union[float, Tuple[float, ...]]:
    """Convert from inches to user units.

    Parameters
    ----------
    value : float or tuple of float
        Value(s) in inches
    to_units : {'cm', 'inches'}
        Target units

    Returns
    -------
    float or tuple of float
        Value(s) in target units

    Raises
    ------
    ValueError
        If units are not 'cm' or 'inches'
    """
    if to_units == 'cm':
        if isinstance(value, tuple):
            return tuple(v * CM_PER_INCH for v in value)
        return value * CM_PER_INCH
    elif to_units == 'inches':
        return value
    else:
        raise ValueError(f"Unknown units: {to_units}. Must be 'cm' or 'inches'")


@dataclass
class PanelSpec:
    """Specification for a single panel in a multi-panel figure.

    Parameters
    ----------
    name : str
        Identifier for the panel (e.g., 'A', 'B', 'C')
    size : tuple of float
        (width, height) in user's preferred units
    position : tuple of int, optional
        (row, col) grid position. If None, panels are arranged sequentially
    row_span : int, optional
        Number of rows this panel spans (default: 1)
    col_span : int, optional
        Number of columns this panel spans (default: 1)
    """
    name: str
    size: Tuple[float, float]
    position: Optional[Tuple[int, int]] = None
    row_span: int = 1
    col_span: int = 1


class PanelLayout:
    """Manages layout and creation of publication-ready multi-panel figures.

    This class handles precise physical dimensions for each subplot and generates
    matplotlib figures with correct sizing and spacing.

    Parameters
    ----------
    units : {'cm', 'inches'}, default 'cm'
        Physical units for panel dimensions and spacing
    dpi : int, default 300
        Dots per inch for the figure (300 for publication, 150 for draft, 72 for screen)
    spacing : dict, optional
        Spacing between panels:
        - 'wspace': horizontal spacing in physical units
        - 'hspace': vertical spacing in physical units
        Default: {'wspace': 0, 'hspace': 0}

    Examples
    --------
    >>> # Simple 2-panel layout
    >>> layout = PanelLayout(units='cm', dpi=300)
    >>> layout.add_panel('A', size=(8, 6))
    >>> layout.add_panel('B', size=(8, 6))
    >>> layout.set_grid(rows=1, cols=2)
    >>> fig, axes = layout.create_figure()

    >>> # Complex layout with custom positioning
    >>> layout = PanelLayout(units='cm', dpi=300, spacing={'wspace': 1.5, 'hspace': 1.0})
    >>> layout.add_panel('A', size=(5, 5), position=(0, 0))
    >>> layout.add_panel('B', size=(10, 5), position=(0, 1))
    >>> layout.add_panel('C', size=(15, 6), position=(1, 0), col_span=2)
    >>> fig, axes = layout.create_figure()
    """

    def __init__(
        self,
        units: Literal['cm', 'inches'] = 'cm',
        dpi: int = 300,
        spacing: Optional[Dict[str, float]] = None
    ):
        self.units = units
        self.dpi = dpi
        self.spacing = spacing or {'wspace': 0, 'hspace': 0}

        self.panels: Dict[str, PanelSpec] = {}
        self.grid_shape: Optional[Tuple[int, int]] = None  # (rows, cols)

    def add_panel(
        self,
        name: str,
        size: Tuple[float, float],
        position: Optional[Tuple[int, int]] = None,
        row_span: int = 1,
        col_span: int = 1,
        **kwargs  # For backward compatibility with 'rowspan' and 'colspan'
    ):
        """Add a panel to the layout.

        Parameters
        ----------
        name : str
            Identifier for the panel (e.g., 'A', 'B', 'C')
        size : tuple of float
            (width, height) in units specified by self.units
        position : tuple of int, optional
            (row, col) grid position. If None, position determined by grid or sequential order
        row_span : int, default 1
            Number of rows this panel spans
        col_span : int, default 1
            Number of columns this panel spans
        **kwargs
            For backward compatibility: 'rowspan', 'colspan'
        """
        # Handle backward compatibility
        if 'rowspan' in kwargs:
            row_span = kwargs['rowspan']
        if 'colspan' in kwargs:
            col_span = kwargs['colspan']

        panel = PanelSpec(
            name=name,
            size=size,
            position=position,
            row_span=row_span,
            col_span=col_span
        )
        self.panels[name] = panel

    def set_grid(self, rows: int, cols: int):
        """Set the grid shape for automatic panel positioning.

        Parameters
        ----------
        rows : int
            Number of rows in the grid
        cols : int
            Number of columns in the grid
        """
        self.grid_shape = (rows, cols)

    def get_panel_size(self, name: str) -> Tuple[float, float]:
        """Get the size of a panel in the layout's units.

        Parameters
        ----------
        name : str
            Panel identifier

        Returns
        -------
        tuple of float
            (width, height) in self.units
        """
        if name not in self.panels:
            raise ValueError(f"Panel '{name}' not found in layout")
        return self.panels[name].size

    def _calculate_figure_size(self) -> Tuple[float, float]:
        """Calculate total figure size in inches based on panels and spacing.

        Returns
        -------
        tuple of float
            (fig_width_inches, fig_height_inches)
        """
        if not self.panels:
            raise ValueError("No panels added to layout")

        # Determine grid shape if not set
        if self.grid_shape is None:
            # Arrange panels in a row by default
            self.grid_shape = (1, len(self.panels))

        rows, cols = self.grid_shape

        # Calculate width and height ratios for grid
        # For simplicity, use max width/height for each grid cell
        col_widths = [0.0] * cols
        row_heights = [0.0] * rows

        for panel in self.panels.values():
            if panel.position is not None:
                row, col = panel.position
                # Update column width (take max if multiple panels in same column)
                col_widths[col] = max(col_widths[col], panel.size[0] / panel.col_span)
                # Update row height (take max if multiple panels in same row)
                row_heights[row] = max(row_heights[row], panel.size[1] / panel.row_span)

        # If any columns/rows are still 0, fill with average
        avg_width = np.mean([w for w in col_widths if w > 0]) if any(w > 0 for w in col_widths) else 8
        avg_height = np.mean([h for h in row_heights if h > 0]) if any(h > 0 for h in row_heights) else 6
        col_widths = [w if w > 0 else avg_width for w in col_widths]
        row_heights = [h if h > 0 else avg_height for h in row_heights]

        # Calculate total size
        total_width = sum(col_widths)
        total_height = sum(row_heights)

        # Add spacing
        wspace = self.spacing.get('wspace', 0)
        hspace = self.spacing.get('hspace', 0)

        total_width += wspace * (cols - 1)
        total_height += hspace * (rows - 1)

        # Convert to inches
        width_inches, height_inches = to_inches((total_width, total_height), self.units)

        return width_inches, height_inches

    def create_figure(self, style=None) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
        """Create a matplotlib figure with all panels.

        Parameters
        ----------
        style : StylePreset, optional
            Style preset to apply to all panels. If None, no styling is applied.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure
        axes : dict of matplotlib.axes.Axes
            Dictionary mapping panel names to their axes

        Examples
        --------
        >>> layout = PanelLayout(units='cm', dpi=300)
        >>> layout.add_panel('A', size=(8, 6), position=(0, 0))
        >>> layout.add_panel('B', size=(8, 6), position=(0, 1))
        >>> layout.set_grid(rows=1, cols=2)
        >>> fig, axes = layout.create_figure()
        >>> axes['A'].plot([1, 2, 3], [1, 4, 9])
        """
        # Calculate figure size
        fig_width_inches, fig_height_inches = self._calculate_figure_size()

        # Create figure
        fig = plt.figure(figsize=(fig_width_inches, fig_height_inches), dpi=self.dpi)

        # Determine grid shape
        rows, cols = self.grid_shape if self.grid_shape is not None else (1, len(self.panels))

        # Calculate column widths and row heights in user units
        col_widths = [0.0] * cols
        row_heights = [0.0] * rows

        for panel in self.panels.values():
            if panel.position is not None:
                row, col = panel.position
                # Only use non-spanning panels to determine column widths/heights
                if panel.col_span == 1:
                    col_widths[col] = max(col_widths[col], panel.size[0])
                if panel.row_span == 1:
                    row_heights[row] = max(row_heights[row], panel.size[1])

        # Fill any zero widths/heights
        avg_width = np.mean([w for w in col_widths if w > 0]) if any(w > 0 for w in col_widths) else 8
        avg_height = np.mean([h for h in row_heights if h > 0]) if any(h > 0 for h in row_heights) else 6
        col_widths = [w if w > 0 else avg_width for w in col_widths]
        row_heights = [h if h > 0 else avg_height for h in row_heights]

        # Get spacing in user units
        wspace = self.spacing.get('wspace', 0)
        hspace = self.spacing.get('hspace', 0)

        # Calculate cumulative positions (left edges and bottom edges) in user units
        col_positions = [0.0]  # left edge of each column
        for i in range(cols):
            col_positions.append(col_positions[-1] + col_widths[i] + (wspace if i < cols-1 else 0))

        row_positions = [0.0]  # bottom edge of each row (from bottom)
        for i in range(rows):
            row_positions.append(row_positions[-1] + row_heights[rows-1-i] + (hspace if i < rows-1 else 0))

        # Convert to inches
        col_positions_inches = [to_inches(p, self.units) for p in col_positions]
        row_positions_inches = [to_inches(p, self.units) for p in row_positions]
        col_widths_inches = [to_inches(w, self.units) for w in col_widths]
        row_heights_inches = [to_inches(h, self.units) for h in row_heights]

        # Create axes manually with precise positioning
        axes = {}

        if self.grid_shape is not None and all(p.position is None for p in self.panels.values()):
            # Simple grid - automatic positioning
            for idx, (name, panel) in enumerate(self.panels.items()):
                row = idx // cols
                col = idx % cols

                left = col_positions_inches[col] / fig_width_inches
                bottom = row_positions_inches[rows-1-row] / fig_height_inches
                width = col_widths_inches[col] / fig_width_inches
                height = row_heights_inches[row] / fig_height_inches

                ax = fig.add_axes([left, bottom, width, height])
                axes[name] = ax
        else:
            # Custom positioning
            for name, panel in self.panels.items():
                if panel.position is None:
                    raise ValueError(f"Panel '{name}' has no position specified")

                row, col = panel.position

                # Calculate position and size in inches
                left_inches = col_positions_inches[col]
                bottom_inches = row_positions_inches[rows-1-row]

                # Handle spanning
                if panel.col_span > 1:
                    width_inches = col_positions_inches[col + panel.col_span] - col_positions_inches[col]
                else:
                    width_inches = col_widths_inches[col]

                if panel.row_span > 1:
                    height_inches = row_positions_inches[rows-row] - row_positions_inches[rows-1-row]
                else:
                    height_inches = row_heights_inches[row]

                # Convert to figure fractions
                left = left_inches / fig_width_inches
                bottom = bottom_inches / fig_height_inches
                width = width_inches / fig_width_inches
                height = height_inches / fig_height_inches

                ax = fig.add_axes([left, bottom, width, height])
                axes[name] = ax

        # Apply styling if provided
        if style is not None:
            for name, ax in axes.items():
                panel_size = self.panels[name].size
                style.apply_to_axes(ax, panel_size, self.units)

        return fig, axes
