"""Publication-ready figure framework for DRIADA.

This module provides a comprehensive framework for creating publication-quality
multi-panel figures with precise physical dimensions and consistent styling.

Key Features
------------
- Precise physical dimensions (cm or inches) for each subplot
- Configurable DPI for different output targets (300 for publication, 150 for draft)
- Automatic font/line scaling based on panel size
- Flexible layout system (grid or custom positioning)
- Support for external plots (from R, MATLAB, etc.)
- Panel labeling utilities (A, B, C, ...)

Quick Start
-----------
>>> from driada.utils.publication import PanelLayout, StylePreset
>>>
>>> # Create layout
>>> layout = PanelLayout(units='cm', dpi=300, spacing={'wspace': 1.5})
>>> layout.add_panel('A', size=(8, 6))
>>> layout.add_panel('B', size=(8, 6))
>>> layout.set_grid(rows=1, cols=2)
>>>
>>> # Create figure with auto-styled panels
>>> style = StylePreset.nature_journal()
>>> fig, axes = layout.create_figure(style=style)
>>>
>>> # Plot data
>>> axes['A'].plot(x, y)
>>> axes['A'].set_xlabel('Time (s)')
>>>
>>> # Save at specified DPI
>>> fig.savefig('figure.pdf', dpi=layout.dpi)

See Also
--------
~driada.utils.plot.make_beautiful :
    Enhanced with auto-scaling support for single-panel figures.
"""

from .layout import PanelLayout, PanelSpec, to_inches, from_inches
from .style import StylePreset
from .external import ExternalPanel, PanelLabeler, format_panel_label

__all__ = [
    # Layout classes
    'PanelLayout',
    'PanelSpec',

    # Style classes
    'StylePreset',

    # External content utilities
    'ExternalPanel',
    'PanelLabeler',
    'format_panel_label',

    # Unit conversion utilities
    'to_inches',
    'from_inches',
]

__version__ = '1.0.0'
