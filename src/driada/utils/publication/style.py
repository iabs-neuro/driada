"""Style presets for publication-ready figures with auto-scaling.

This module provides StylePreset class that automatically scales fonts and line widths
based on panel physical size to maintain consistent visual density across panels.
"""

import numpy as np
from typing import Tuple, Optional, Literal
from dataclasses import dataclass


@dataclass
class StylePreset:
    """Style preset with consistent physical sizing across all panels.

    This class defines styling parameters that maintain the SAME physical size
    (in cm/inches) across all panels regardless of their dimensions. This ensures
    that when printed, all text and lines have identical physical measurements.

    For advanced use cases, area-based scaling can be enabled to maintain visual
    density across panels of different sizes instead.

    Parameters
    ----------
    name : str
        Name of the preset (e.g., 'nature', 'custom')
    reference_size : tuple of float
        (width, height) of reference panel in reference_units.
        Only used when scaling_mode='area'.
    reference_units : {'cm', 'inches'}, default 'cm'
        Units for reference_size
    base_spine_width : float, default 1.5
        Spine width in points (same physical size on all panels)
    base_tick_width : float, default 1.5
        Tick width in points (same physical size on all panels)
    base_tick_length : float, default 6
        Tick length in points (same physical size on all panels)
    base_tick_pad : float, default 8
        Tick padding in points (same physical size on all panels)
    base_tick_labelsize : float, default 8
        Tick label font size in points (same physical size on all panels)
    base_label_size : float, default 10
        Axis label font size in points (same physical size on all panels)
    base_title_size : float, default 10
        Title font size in points (same physical size on all panels)
    base_legend_fontsize : float, default 8
        Legend font size in points (same physical size on all panels)
    scaling_mode : {'fixed', 'area'}, default 'fixed'
        Scaling behavior:
        - 'fixed': Same physical size on all panels (DEFAULT, recommended)
        - 'area': Scale by sqrt(area_ratio) to preserve visual density

    Examples
    --------
    >>> # Create a Nature journal preset with fixed physical sizing
    >>> style = StylePreset.nature_journal()
    >>> # Apply to any panel - fonts will have same physical size
    >>> style.apply_to_axes(ax, (8, 8), 'cm')
    >>> style.apply_to_axes(ax2, (4, 4), 'cm')  # Same font size as above!
    """

    name: str = "default"
    reference_size: Tuple[float, float] = (8.0, 8.0)
    reference_units: Literal["cm", "inches"] = "cm"

    # Base parameters (applied to all panels with same physical size by default)
    base_spine_width: float = 1.5
    base_tick_width: float = 1.5
    base_tick_length: float = 6
    base_tick_pad: float = 8
    base_tick_labelsize: float = 8
    base_label_size: float = 10
    base_title_size: float = 10
    base_legend_fontsize: float = 8

    # Additional style parameters
    legend_frameon: bool = False
    lowercase_labels: bool = False
    tight_layout: bool = True
    scaling_mode: Literal["fixed", "area"] = "fixed"  # DEFAULT: same physical size

    def calculate_scale_factor(
        self, panel_size: Tuple[float, float], panel_units: Literal["cm", "inches"]
    ) -> float:
        """Calculate scale factor based on scaling mode.

        Parameters
        ----------
        panel_size : tuple of float
            (width, height) of the panel
        panel_units : {'cm', 'inches'}
            Units for panel_size

        Returns
        -------
        float
            Scale factor to apply to all visual elements:
            - scaling_mode='fixed': Always returns 1.0 (same physical size on all panels)
            - scaling_mode='area': Returns sqrt(panel_area / reference_area) to preserve
              visual density across different panel sizes
        """
        # Fixed mode: all panels get the same absolute physical sizes
        if self.scaling_mode == "fixed":
            return 1.0

        # Area mode: scale by sqrt of area ratio to preserve visual density
        from .layout import to_inches

        # Convert both to inches for comparison
        ref_size_inches = to_inches(self.reference_size, self.reference_units)
        panel_size_inches = to_inches(panel_size, panel_units)

        # Calculate areas
        ref_area = ref_size_inches[0] * ref_size_inches[1]
        panel_area = panel_size_inches[0] * panel_size_inches[1]

        # Scale factor is sqrt of area ratio
        scale = np.sqrt(panel_area / ref_area)

        return scale

    def apply_to_axes(
        self, ax, panel_size: Tuple[float, float], panel_units: Literal["cm", "inches"] = "cm"
    ) -> None:
        """Apply scaled styling to a matplotlib axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to style
        panel_size : tuple of float
            (width, height) of the panel in panel_units
        panel_units : {'cm', 'inches'}, default 'cm'
            Units for panel_size

        Examples
        --------
        >>> style = StylePreset.nature_journal()
        >>> fig, ax = plt.subplots(figsize=(3, 3))
        >>> style.apply_to_axes(ax, (8, 8), 'cm')
        """
        # Calculate scale factor
        scale = self.calculate_scale_factor(panel_size, panel_units)

        # Apply scaled styling
        spine_width = self.base_spine_width * scale
        tick_width = self.base_tick_width * scale
        tick_length = self.base_tick_length * scale
        tick_pad = self.base_tick_pad * scale
        tick_labelsize = self.base_tick_labelsize * scale
        label_size = self.base_label_size * scale
        title_size = self.base_title_size * scale
        legend_fontsize = self.base_legend_fontsize * scale

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
        ax.title.set_size(title_size)

        # Handle legend if present
        if ax.legend_:
            ax.legend(frameon=self.legend_frameon, fontsize=legend_fontsize)

        # Apply lowercase if requested
        if self.lowercase_labels:
            if ax.get_xlabel():
                ax.set_xlabel(ax.get_xlabel().lower())
            if ax.get_ylabel():
                ax.set_ylabel(ax.get_ylabel().lower())
            if ax.get_title():
                ax.set_title(ax.get_title().lower())

        # Note: tight_layout is better handled at figure save time with bbox_inches='tight'
        # We don't force margins here to allow flexibility in plot content

    @classmethod
    def nature_journal(cls, scaling_mode: Literal["fixed", "area"] = "fixed") -> "StylePreset":
        """Create a preset styled for Nature journal specifications.

        Nature requires figures with precise dimensions and professional appearance.
        By default, uses fixed physical sizing so all panels have identical font
        and line sizes when measured with a ruler on the printed page.

        Parameters
        ----------
        scaling_mode : {'fixed', 'area'}, default 'fixed'
            - 'fixed': Same physical size on all panels (recommended for most cases)
            - 'area': Scale by panel area to preserve visual density

        Returns
        -------
        StylePreset
            Configured preset for Nature journal

        Examples
        --------
        >>> # Default: fixed physical size across all panels
        >>> style = StylePreset.nature_journal()
        >>>
        >>> # Optional: area-based scaling for visual density preservation
        >>> style_area = StylePreset.nature_journal(scaling_mode='area')
        """
        return cls(
            name="nature",
            reference_size=(8.0, 8.0),
            reference_units="cm",
            base_spine_width=1.5,
            base_tick_width=1.5,
            base_tick_length=6,
            base_tick_pad=8,
            base_tick_labelsize=8,
            base_label_size=10,
            base_title_size=10,
            base_legend_fontsize=8,
            legend_frameon=False,
            lowercase_labels=False,
            tight_layout=True,
            scaling_mode=scaling_mode,
        )

    @classmethod
    def fixed_size(cls, **kwargs) -> "StylePreset":
        """Create a preset with FIXED sizes across all panels.

        This is an explicit alias for the default behavior. All panels get the
        same font sizes and line widths regardless of their physical dimensions,
        ensuring consistent absolute physical size when printed.

        Note: This is now the DEFAULT behavior, so StylePreset() and
        StylePreset.nature_journal() already use fixed sizing. This method
        is provided for explicit clarity.

        Parameters
        ----------
        **kwargs : dict
            Override any style parameters (e.g., base_spine_width=2.0)

        Returns
        -------
        StylePreset
            Configured preset with fixed scaling

        Examples
        --------
        >>> # All panels get 10pt fonts and 1.5pt lines regardless of size
        >>> style = StylePreset.fixed_size()
        >>> # Or customize:
        >>> style = StylePreset.fixed_size(base_label_size=12, base_spine_width=2.0)
        """
        defaults = {
            "name": "fixed_size",
            "reference_size": (8.0, 8.0),  # Not used when scaling_mode='fixed'
            "reference_units": "cm",
            "base_spine_width": 1.5,
            "base_tick_width": 1.5,
            "base_tick_length": 6,
            "base_tick_pad": 8,
            "base_tick_labelsize": 8,
            "base_label_size": 10,
            "base_title_size": 10,
            "base_legend_fontsize": 8,
            "legend_frameon": False,
            "lowercase_labels": False,
            "tight_layout": True,
            "scaling_mode": "fixed",  # Explicit fixed mode (also the default)
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def from_make_beautiful(
        cls,
        spine_width: float = 4,
        tick_width: float = 4,
        tick_length: float = 8,
        tick_pad: float = 15,
        tick_labelsize: int = 26,
        label_size: int = 30,
        title_size: int = 30,
        legend_fontsize: int = 18,
        reference_size: Tuple[float, float] = (16.0, 12.0),
        reference_units: Literal["cm", "inches"] = "inches",
    ) -> "StylePreset":
        """Create a preset matching existing make_beautiful() styling.

        This allows converting existing code to use the new framework while
        maintaining the same visual appearance.

        Parameters
        ----------
        spine_width : float, default 4
            Spine width from make_beautiful
        tick_width : float, default 4
            Tick width from make_beautiful
        tick_length : float, default 8
            Tick length from make_beautiful
        tick_pad : float, default 15
            Tick padding from make_beautiful
        tick_labelsize : int, default 26
            Tick label font size from make_beautiful
        label_size : int, default 30
            Axis label font size from make_beautiful
        title_size : int, default 30
            Title font size from make_beautiful
        legend_fontsize : int, default 18
            Legend font size from make_beautiful
        reference_size : tuple of float, default (16.0, 12.0)
            Reference panel size (matches make_beautiful default figsize)
        reference_units : {'cm', 'inches'}, default 'inches'
            Units for reference_size

        Returns
        -------
        StylePreset
            Configured preset matching make_beautiful
        """
        return cls(
            name="make_beautiful",
            reference_size=reference_size,
            reference_units=reference_units,
            base_spine_width=spine_width,
            base_tick_width=tick_width,
            base_tick_length=tick_length,
            base_tick_pad=tick_pad,
            base_tick_labelsize=tick_labelsize,
            base_label_size=label_size,
            base_title_size=title_size,
            base_legend_fontsize=legend_fontsize,
            legend_frameon=False,
            lowercase_labels=True,  # make_beautiful default
            tight_layout=True,
        )

    def copy(self, **kwargs) -> "StylePreset":
        """Create a copy of this preset with optional parameter overrides.

        Parameters
        ----------
        **kwargs
            Parameters to override in the copy

        Returns
        -------
        StylePreset
            New preset with modified parameters

        Examples
        --------
        >>> style = StylePreset.nature_journal()
        >>> larger_fonts = style.copy(base_label_size=12, base_title_size=14)
        >>> area_scaled = style.copy(scaling_mode='area')
        """
        # Get current values as dict
        current = {
            "name": self.name,
            "reference_size": self.reference_size,
            "reference_units": self.reference_units,
            "base_spine_width": self.base_spine_width,
            "base_tick_width": self.base_tick_width,
            "base_tick_length": self.base_tick_length,
            "base_tick_pad": self.base_tick_pad,
            "base_tick_labelsize": self.base_tick_labelsize,
            "base_label_size": self.base_label_size,
            "base_title_size": self.base_title_size,
            "base_legend_fontsize": self.base_legend_fontsize,
            "legend_frameon": self.legend_frameon,
            "lowercase_labels": self.lowercase_labels,
            "tight_layout": self.tight_layout,
            "scaling_mode": self.scaling_mode,
        }

        # Update with overrides
        current.update(kwargs)

        return StylePreset(**current)
