"""
Backward compatibility module for signals.

This module provides backward compatibility for code that imports from driada.signals.
All functionality has been moved to driada.utils.signals.

.. deprecated:: 0.2.0
   The driada.signals module is deprecated. Import from driada.utils instead.
"""

import warnings

# Show deprecation warning when this module is imported
warnings.warn(
    "The driada.signals module is deprecated and will be removed in a future version. "
    "Import from driada.utils instead:\n"
    "  from driada.utils import filter_signals, adaptive_filter_signals",
    DeprecationWarning,
    stacklevel=2
)

# Import from new location for backward compatibility
from .utils.signals import (
    filter_signals,
    adaptive_filter_signals,
    filter_1d_timeseries
)

# For backward compatibility with old names
filter_neural_signals = filter_signals
adaptive_filter_neural_signals = adaptive_filter_signals

__all__ = [
    'filter_signals',
    'adaptive_filter_signals',
    'filter_1d_timeseries',
    'filter_neural_signals',  # Deprecated name
    'adaptive_filter_neural_signals',  # Deprecated name
]