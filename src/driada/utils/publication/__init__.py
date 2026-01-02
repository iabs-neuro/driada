"""Publication-ready figure framework for DRIADA.

This module provides a comprehensive framework for creating publication-quality
multi-panel figures with precise physical dimensions and consistent styling.

Philosophy: SAME PHYSICAL SIZE ACROSS ALL PANELS
-------------------------------------------------
By default, all text and lines maintain the SAME physical size (in cm/inches)
across all panels regardless of their dimensions. When printed and measured
with a ruler, a 10pt font will be exactly 10pt in every panel, whether the
panel is 4×4 cm or 12×12 cm.

This ensures professional, consistent appearance in publications.

Key Features
------------
- Precise physical dimensions (cm or inches) for each subplot
- Configurable DPI for different output targets (300 for publication, 150 for draft)
- Fixed physical sizing: Same font/line sizes across all panels (DEFAULT)
- Optional area-based scaling for advanced use cases
- Flexible layout system (grid or custom positioning)
- Support for external plots (from R, MATLAB, etc.)
- Panel labeling utilities (A, B, C, ...)
- Seamless integration with driada.utils.plot.make_beautiful()

Quick Start
-----------
>>> from driada.utils.publication import PanelLayout, StylePreset
>>>
>>> # Create layout with different panel sizes
>>> layout = PanelLayout(units='cm', dpi=300, spacing={'wspace': 1.5})
>>> layout.add_panel('A', size=(4, 4))  # Small panel
>>> layout.add_panel('B', size=(8, 8))  # Large panel
>>> layout.set_grid(rows=1, cols=2)
>>>
>>> # Create figure - all panels get SAME physical font/line sizes
>>> style = StylePreset.nature_journal()
>>> fig, axes = layout.create_figure(style=style)
>>>
>>> # Plot data - fonts will be identical physical size in both panels!
>>> axes['A'].plot(x, y)
>>> axes['A'].set_xlabel('Time (s)')  # 10pt font
>>> axes['B'].plot(x, y)
>>> axes['B'].set_xlabel('Time (s)')  # Also 10pt font (same physical size)
>>>
>>> # Save at specified DPI
>>> fig.savefig('figure.pdf', dpi=layout.dpi, bbox_inches='tight')

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
