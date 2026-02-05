"""Utilities for circular feature transformation.

This module provides functions to transform circular/angular data to a
(cos, sin) representation that works correctly with standard MI estimators.

The Problem
-----------
Circular features have discontinuity at boundaries (value jumps from ~2π to ~0).
Standard MI estimators (GCMI, KSG) use Euclidean distance, so points 0.01 and
2π-0.01 are neighbors but treated as maximally far. This results in biased MI
estimates and incorrect selectivity detection.

The Solution
------------
Transform θ ∈ [0, 2π] → (cos θ, sin θ) ∈ ℝ²

This embedding:
1. Preserves topology: Points near boundary wrap correctly on unit circle
2. Maintains distances: Euclidean distance on (cos, sin) approximates circular distance
3. Enables standard estimators: GCMI and KSG work correctly
4. Is reversible: arctan2(sin, cos) recovers original angle exactly
"""

import numpy as np

from .info_base import TimeSeries, MultiTimeSeries


def circular_to_cos_sin(data, period=None, name=None):
    """Transform circular angle data to (cos, sin) MultiTimeSeries.

    Parameters
    ----------
    data : array-like or TimeSeries
        Circular/angular data.
    period : float, optional
        Circular period. If None, auto-detect from data range.
        Common values: 2*pi (radians), 360 (degrees).
    name : str, optional
        Name for the resulting MultiTimeSeries. If the input is a TimeSeries
        with a name and this is None, uses "{ts.name}_2d".

    Returns
    -------
    MultiTimeSeries
        2D MTS with cos and sin components.

    Examples
    --------
    >>> import numpy as np
    >>> angles = np.linspace(0, 2*np.pi, 100)
    >>> mts = circular_to_cos_sin(angles, period=2*np.pi, name="hd_2d")
    >>> mts.n_dim
    2
    """
    if isinstance(data, TimeSeries):
        arr = data.data
        if period is None and data.type_info and data.type_info.is_circular:
            period = data.type_info.circular_period
        if name is None and data.name:
            name = f"{data.name}_2d"
    else:
        arr = np.asarray(data)

    # Auto-detect period if not provided
    if period is None:
        period = detect_circular_period(arr)

    # Normalize to radians
    normalized = normalize_to_radians(arr, period)

    # Compute cos and sin components
    cos_component = np.cos(normalized)
    sin_component = np.sin(normalized)

    # Create component TimeSeries (internal names, not exposed as separate features)
    cos_ts = TimeSeries(cos_component, discrete=False, name="cos")
    sin_ts = TimeSeries(sin_component, discrete=False, name="sin")

    return MultiTimeSeries([cos_ts, sin_ts], name=name)


def detect_circular_period(data):
    """Auto-detect circular period from data range.

    Parameters
    ----------
    data : array-like
        Circular data to analyze.

    Returns
    -------
    float
        Detected period (2*pi for radians, 360 for degrees).

    Examples
    --------
    >>> import numpy as np
    >>> rad_data = np.random.uniform(0, 2*np.pi, 100)
    >>> period = detect_circular_period(rad_data)
    >>> abs(period - 2*np.pi) < 0.1
    True
    """
    data = np.asarray(data)
    data_range = np.nanmax(data) - np.nanmin(data)
    data_max = np.nanmax(data)

    # Check for common circular ranges
    if data_max <= 2 * np.pi + 0.1 and data_range > np.pi:
        return 2 * np.pi  # Radians [0, 2π] or [-π, π]
    elif data_max <= 360 + 1 and data_range > 180:
        return 360.0  # Degrees [0, 360] or [-180, 180]
    else:
        return 2 * np.pi  # Default to radians


def normalize_to_radians(data, period):
    """Normalize circular data to radians [0, 2π).

    Parameters
    ----------
    data : array-like
        Circular data to normalize.
    period : float
        Period of the circular variable.

    Returns
    -------
    ndarray
        Data normalized to radians.

    Examples
    --------
    >>> import numpy as np
    >>> deg_data = np.array([0, 90, 180, 270, 360])
    >>> rad_data = normalize_to_radians(deg_data, 360)
    >>> np.allclose(rad_data, [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    True
    """
    data = np.asarray(data)
    if period is None or abs(period - 2 * np.pi) < 0.1:
        return data  # Already in radians
    else:
        # Convert to radians (e.g., degrees -> radians)
        return data * (2 * np.pi / period)


def cos_sin_to_circular(cos_data, sin_data, period=2 * np.pi):
    """Inverse transformation: recover angle from (cos, sin) components.

    Parameters
    ----------
    cos_data : array-like
        Cosine component.
    sin_data : array-like
        Sine component.
    period : float, optional
        Desired output period. Default is 2*pi (radians).

    Returns
    -------
    ndarray
        Recovered angles in [0, period).

    Examples
    --------
    >>> import numpy as np
    >>> original = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    >>> cos_data = np.cos(original)
    >>> sin_data = np.sin(original)
    >>> recovered = cos_sin_to_circular(cos_data, sin_data)
    >>> np.allclose(recovered, original)
    True
    """
    cos_data = np.asarray(cos_data)
    sin_data = np.asarray(sin_data)

    angles = np.arctan2(sin_data, cos_data)
    angles = angles % (2 * np.pi)  # Normalize to [0, 2π)

    if abs(period - 2 * np.pi) > 0.1:
        angles = angles * (period / (2 * np.pi))  # Convert back to original units

    return angles


def get_circular_2d_name(feature_name):
    """Get the _2d counterpart name for a circular feature.

    Parameters
    ----------
    feature_name : str
        Original circular feature name.

    Returns
    -------
    str
        Name with _2d suffix.

    Examples
    --------
    >>> get_circular_2d_name("headdirection")
    'headdirection_2d'
    """
    return f"{feature_name}_2d"


def is_circular_2d_feature(feature_name):
    """Check if a feature name is a _2d circular transformation.

    Parameters
    ----------
    feature_name : str
        Feature name to check.

    Returns
    -------
    bool
        True if the name ends with '_2d'.

    Examples
    --------
    >>> is_circular_2d_feature("headdirection_2d")
    True
    >>> is_circular_2d_feature("headdirection")
    False
    """
    return feature_name.endswith("_2d")
