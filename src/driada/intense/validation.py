"""
Input validation for INTENSE analysis.

Provides validation functions for time series bunches, metrics, and common
parameters used throughout the INTENSE pipeline.
"""

import numpy as np

from ..information.info_base import TimeSeries, MultiTimeSeries


def validate_time_series_bunches(ts_bunch1, ts_bunch2) -> None:
    """
    Validate time series bunches for INTENSE computations.

    Parameters
    ----------
    ts_bunch1 : list of TimeSeries or MultiTimeSeries
        First set of time series objects (e.g., neural activity).
    ts_bunch2 : list of TimeSeries or MultiTimeSeries
        Second set of time series objects (e.g., behavioral features).

    Raises
    ------
    ValueError
        If bunches are empty, contain wrong types, or have mismatched lengths.
    """
    if len(ts_bunch1) == 0:
        raise ValueError("ts_bunch1 cannot be empty")
    if len(ts_bunch2) == 0:
        raise ValueError("ts_bunch2 cannot be empty")

    # Check lengths match
    lengths1 = [
        len(ts.data) if isinstance(ts, TimeSeries) else ts.data.shape[1] for ts in ts_bunch1
    ]
    lengths2 = [
        len(ts.data) if isinstance(ts, TimeSeries) else ts.data.shape[1] for ts in ts_bunch2
    ]

    if len(set(lengths1)) > 1:
        raise ValueError(f"All time series in ts_bunch1 must have same length, got {set(lengths1)}")
    if len(set(lengths2)) > 1:
        raise ValueError(f"All time series in ts_bunch2 must have same length, got {set(lengths2)}")
    if lengths1[0] != lengths2[0]:
        raise ValueError(f"Time series lengths don't match: {lengths1[0]} vs {lengths2[0]}")


def validate_metric(metric, allow_scipy=True) -> str:
    """
    Validate metric name and check if it's supported.

    Parameters
    ----------
    metric : str
        Metric name to validate. Supported metrics:

        - 'mi': Mutual information (supports multivariate data)
        - 'av': Activity ratio (requires one binary and one continuous variable)
        - 'fast_pearsonr': Fast Pearson correlation implementation
        - 'spearmanr', 'pearsonr', 'kendalltau': scipy.stats correlation functions
        - Any other callable from scipy.stats (if allow_scipy=True)
    allow_scipy : bool, default=True
        Whether to allow scipy.stats correlation functions.

    Returns
    -------
    metric_type : str
        Type of metric:

        - 'mi': Mutual information metric
        - 'special': Special metrics ('av', 'fast_pearsonr')
        - 'scipy': scipy.stats functions

    Raises
    ------
    ValueError
        If metric is not supported or not a callable function in scipy.stats.

    Notes
    -----
    The function validates that scipy.stats attributes are callable to prevent
    accepting non-function attributes like constants or data arrays.
    """
    # Built-in metrics
    if metric == "mi":
        return "mi"

    # Special metrics
    if metric in ["av", "fast_pearsonr"]:
        return "special"

    # Full scipy names
    scipy_correlation_metrics = ["spearmanr", "pearsonr", "kendalltau"]
    if metric in scipy_correlation_metrics:
        return "scipy"

    # Check if it's a scipy function
    if allow_scipy:
        try:
            import scipy.stats

            attr = getattr(scipy.stats, metric, None)
            if attr is not None and callable(attr):
                return "scipy"
        except ImportError:
            pass

    # If we get here, metric is not supported
    raise ValueError(
        f"Unsupported metric: {metric}. Supported metrics include: "
        f"'mi', 'av', 'fast_pearsonr', 'spearmanr', 'pearsonr', 'kendalltau', "
        f"and other scipy.stats functions."
    )


def validate_common_parameters(shift_window=None, ds=None, nsh=None, noise_const=None) -> None:
    """
    Validate common INTENSE parameters.

    Parameters
    ----------
    shift_window : int, optional
        Maximum shift window in frames. Must be non-negative.
    ds : int, optional
        Downsampling factor. Must be positive integer.
    nsh : int, optional
        Number of shuffles for significance testing. Must be positive integer.
    noise_const : float, optional
        Noise constant for numerical stability. Must be non-negative.

    Raises
    ------
    TypeError
        If parameters have incorrect types (non-integer for int params; non-numeric for noise_const).
    ValueError
        If parameters have invalid values (negative shift_window or noise_const; non-positive ds or nsh).

    Notes
    -----
    This function validates parameter types using isinstance checks for numpy
    compatibility (accepts both Python int and numpy integer types).
    """
    if shift_window is not None:
        if not isinstance(shift_window, (int, np.integer)):
            raise TypeError(f"shift_window must be integer, got {type(shift_window).__name__}")
        if shift_window < 0:
            raise ValueError(f"shift_window must be non-negative, got {shift_window}")

    if ds is not None:
        if not isinstance(ds, (int, np.integer)):
            raise TypeError(f"ds must be integer, got {type(ds).__name__}")
        if ds <= 0:
            raise ValueError(f"ds must be positive, got {ds}")

    if nsh is not None:
        if not isinstance(nsh, (int, np.integer)):
            raise TypeError(f"nsh must be integer, got {type(nsh).__name__}")
        if nsh <= 0:
            raise ValueError(f"nsh must be positive, got {nsh}")

    if noise_const is not None:
        if not isinstance(noise_const, (int, float, np.number)):
            raise TypeError(f"noise_const must be numeric, got {type(noise_const).__name__}")
        if noise_const < 0:
            raise ValueError(f"noise_const must be non-negative, got {noise_const}")
