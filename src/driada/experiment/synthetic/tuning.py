"""
Tuning curve functions for synthetic neural data generation.

This module consolidates all tuning curve implementations used for generating
neural responses to various features:

- von_mises_tuning_curve: Circular tuning for head direction cells
- gaussian_place_field: 2D Gaussian for place cells
- sigmoid_tuning_curve: Monotonic response for speed cells
- threshold_response: Binary rate modulation for event-responsive cells
- combine_responses: Combine multiple tuning curves

Also includes utilities for computing derived features:
- compute_speed_from_positions: Instantaneous speed from trajectory
- compute_head_direction_from_positions: Head direction from trajectory
"""

from typing import Dict, List, Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .time_series import discretize_via_roi
from ...utils.data import check_positive, check_nonnegative


# =============================================================================
# Default Tuning Parameters
# =============================================================================

TUNING_DEFAULTS = {
    "head_direction": {"kappa": 4.0},  # von Mises concentration
    "x": {"sigma": 0.25},  # Gaussian width (1D marginal)
    "y": {"sigma": 0.25},  # Gaussian width (1D marginal)
    "position_2d": {"sigma": 0.10},  # True 2D Gaussian place field width
    "speed": {"slope": 12.0},  # Sigmoid slope
    "fbm": {"slope": 8.0, "hurst": 0.7},  # FBM sigmoid slope and Hurst exponent
}


# =============================================================================
# Tuning Curves
# =============================================================================


def von_mises_tuning_curve(angles, preferred_direction, kappa):
    """
    Calculate neural response using Von Mises tuning curve.

    Implements a normalized Von Mises (circular Gaussian) tuning curve,
    commonly used to model head direction cells and other neurons with
    circular selectivity.

    Parameters
    ----------
    angles : ndarray
        Current head directions in radians. Can be any real values
        (automatically handles periodicity).
    preferred_direction : float
        Preferred direction of the neuron in radians.
    kappa : float
        Concentration parameter (inverse width of tuning curve).
        Higher kappa = narrower tuning. Typical values: 2-8.
        kappa=0 gives uniform response, negative kappa inverts tuning.

    Returns
    -------
    response : ndarray
        Neural response (firing rate modulation) normalized to max=1.
        Same shape as input angles. Values in range [exp(-kappa), 1].

    Raises
    ------
    ValueError
        If kappa is NaN or infinity.
    TypeError
        If inputs are not numeric types.

    Notes
    -----
    The response follows: response = exp(kappa * (cos(theta - theta_pref) - 1))
    This is a Von Mises distribution normalized to peak at 1.
    """
    # Input validation
    angles = np.asarray(angles)
    if not np.issubdtype(angles.dtype, np.number):
        raise TypeError("angles must be numeric")
    if not isinstance(preferred_direction, (int, float)):
        raise TypeError("preferred_direction must be numeric")
    if not isinstance(kappa, (int, float)):
        raise TypeError("kappa must be numeric")
    if not np.isfinite(kappa):
        raise ValueError("kappa must be finite")

    # Von Mises distribution normalized to max=1
    response = np.exp(kappa * (np.cos(angles - preferred_direction) - 1))
    return response


def gaussian_place_field(positions, center, sigma=0.1):
    """
    Calculate neural response using 2D Gaussian place field.

    Implements an isotropic 2D Gaussian receptive field commonly used to
    model hippocampal place cells. The response peaks at the field center
    and falls off with a Gaussian profile.

    Parameters
    ----------
    positions : ndarray
        Shape (2, n_timepoints) with x, y coordinates. First row is x,
        second row is y coordinates.
    center : ndarray
        Shape (2,) with place field center coordinates [x, y].
    sigma : float, optional
        Width (standard deviation) of the place field. Must be positive.
        Default is 0.1. Larger values give wider fields.

    Returns
    -------
    response : ndarray
        Neural response (firing rate modulation) in range [0, 1].
        Shape: (n_timepoints,). Maximum value is 1.0 at field center.

    Raises
    ------
    ValueError
        If sigma is not positive, or if positions shape is invalid.
    TypeError
        If inputs are not numeric arrays.

    Notes
    -----
    The response follows a 2D Gaussian:
    response = exp(-((x-cx)^2 + (y-cy)^2) / (2*sigma^2))
    where (cx, cy) is the field center and sigma is the width.
    """
    # Input validation
    positions = np.asarray(positions)
    center = np.asarray(center)

    if positions.ndim != 2 or positions.shape[0] != 2:
        raise ValueError("positions must have shape (2, n_timepoints)")
    if center.shape != (2,):
        raise ValueError("center must have shape (2,)")

    check_positive(sigma=sigma)

    # Calculate squared distance from center
    dx = positions[0, :] - center[0]
    dy = positions[1, :] - center[1]
    dist_sq = dx**2 + dy**2

    # Gaussian response
    response = np.exp(-dist_sq / (2 * sigma**2))

    return response


def sigmoid_tuning_curve(
    x: np.ndarray,
    threshold: float,
    slope: float,
    max_response: float = 1.0,
) -> np.ndarray:
    """
    Sigmoid tuning curve for monotonic response to linear features.

    Used for speed cells that increase firing rate with running speed.

    Parameters
    ----------
    x : array-like
        Input values (e.g., running speed normalized to [0, 1]).
    threshold : float
        Value at which response is 50% of maximum.
    slope : float
        Steepness of the sigmoid. Higher = sharper transition.
    max_response : float, optional
        Maximum response value. Default: 1.0.

    Returns
    -------
    response : ndarray
        Response values in [0, max_response].

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 1, 100)
    >>> response = sigmoid_tuning_curve(x, threshold=0.5, slope=10.0)
    >>> response[50]  # At threshold, response is 50%
    0.5
    """
    x = np.asarray(x)
    return max_response / (1 + np.exp(-slope * (x - threshold)))


def threshold_response(
    feature: np.ndarray,
    discretization: str = "roi",
    threshold: float = 0.5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Binary threshold response for rate modulation.

    Discretizes a continuous feature to binary and returns binary response.
    Used for threshold-based selectivity where neurons fire at high rate
    when feature is "active" (above threshold or in ROI) and low rate otherwise.

    Parameters
    ----------
    feature : ndarray
        Input feature values (typically in [0, 1] range).
    discretization : str, optional
        Method for discretization:
        - "roi": Use ROI-based discretization (picks 15% active window). Default.
        - "binary": Simple threshold at given value.
    threshold : float, optional
        Threshold value for "binary" mode. Default: 0.5.
    seed : int, optional
        Random seed for ROI selection reproducibility.

    Returns
    -------
    response : ndarray
        Binary response (0 or 1) indicating feature active state.

    Examples
    --------
    >>> import numpy as np
    >>> feature = np.array([0.1, 0.3, 0.7, 0.9, 0.2])
    >>> threshold_response(feature, discretization="binary", threshold=0.5)
    array([0., 0., 1., 1., 0.])
    """
    feature = np.asarray(feature)

    if discretization == "roi":
        # Use ROI-based discretization (picks ~15% active window)
        return discretize_via_roi(feature, seed=seed).astype(float)
    elif discretization == "binary":
        # Simple threshold
        return (feature > threshold).astype(float)
    else:
        raise ValueError(
            f"Unknown discretization mode: {discretization}. "
            "Use 'roi' or 'binary'."
        )


# =============================================================================
# Feature Computation Utilities
# =============================================================================


def compute_speed_from_positions(
    positions: np.ndarray,
    fps: float,
    smooth_sigma: float = 3,
) -> np.ndarray:
    """
    Compute instantaneous speed from 2D position trajectory.

    Parameters
    ----------
    positions : ndarray
        Shape (2, n_timepoints) with x, y coordinates.
    fps : float
        Sampling rate in Hz.
    smooth_sigma : float, optional
        Gaussian smoothing sigma in frames. Default: 3.

    Returns
    -------
    speed : ndarray
        Instantaneous speed, smoothed. Shape (n_timepoints,).

    Examples
    --------
    >>> import numpy as np
    >>> positions = np.random.randn(2, 100) * 0.1
    >>> positions = np.cumsum(positions, axis=1)
    >>> speed = compute_speed_from_positions(positions, fps=20)
    >>> speed.shape
    (100,)
    """
    dx = np.diff(positions[0, :])
    dy = np.diff(positions[1, :])
    speed = np.sqrt(dx**2 + dy**2) * fps
    # Pad to match original length
    speed = np.concatenate([[speed[0]], speed])
    # Smooth for more realistic speed signal
    speed = gaussian_filter1d(speed, sigma=smooth_sigma)
    return speed


def compute_head_direction_from_positions(
    positions: np.ndarray,
    smooth_sigma: float = 5,
) -> np.ndarray:
    """
    Compute head direction from 2D position trajectory.

    Assumes animal faces in direction of movement.

    Parameters
    ----------
    positions : ndarray
        Shape (2, n_timepoints) with x, y coordinates.
    smooth_sigma : float, optional
        Gaussian smoothing sigma in frames. Default: 5.

    Returns
    -------
    head_direction : ndarray
        Head direction in radians [0, 2*pi). Shape (n_timepoints,).

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> positions = np.random.randn(2, 100) * 0.1
    >>> positions = np.cumsum(positions, axis=1)
    >>> hd = compute_head_direction_from_positions(positions)
    >>> hd.shape
    (100,)
    >>> 0 <= hd.min() and hd.max() < 2 * np.pi
    True
    """
    # Smooth positions first for cleaner velocity estimate
    x_smooth = gaussian_filter1d(positions[0, :], sigma=smooth_sigma)
    y_smooth = gaussian_filter1d(positions[1, :], sigma=smooth_sigma)

    dx = np.diff(x_smooth)
    dy = np.diff(y_smooth)
    # Compute angle from velocity
    angles = np.arctan2(dy, dx)
    # Pad to match original length
    angles = np.concatenate([[angles[0]], angles])
    # Convert to [0, 2*pi)
    angles = angles % (2 * np.pi)
    # Additional smoothing using circular mean (via sin/cos)
    sin_smooth = gaussian_filter1d(np.sin(angles), sigma=smooth_sigma)
    cos_smooth = gaussian_filter1d(np.cos(angles), sigma=smooth_sigma)
    head_direction = np.arctan2(sin_smooth, cos_smooth) % (2 * np.pi)
    return head_direction


# =============================================================================
# Response Combination
# =============================================================================


def combine_responses(
    responses: List[np.ndarray],
    weights: Optional[List[float]] = None,
    mode: str = "or",
) -> np.ndarray:
    """
    Combine multiple response traces using various combination modes.

    Parameters
    ----------
    responses : list of ndarray
        List of response traces, each in range [0, 1].
    weights : list of float, optional
        Weights for each response. Must sum to 1.0 for weighted modes.
        If None and using weighted mode, defaults to equal weights.
    mode : str, optional
        Combination mode. Options:
        - "or": Element-wise maximum (any feature active). Default.
        - "and": Element-wise minimum (all features active).
        - "weighted_sum": Sum of (response * weight), clipped to [0, 1].
        - "weighted_or": Maximum of (response * weight).

    Returns
    -------
    combined : ndarray
        Combined response trace.

    Examples
    --------
    >>> import numpy as np
    >>> r1 = np.array([0.2, 0.8, 0.1])
    >>> r2 = np.array([0.5, 0.3, 0.9])
    >>> combine_responses([r1, r2], mode="or")
    array([0.5, 0.8, 0.9])
    >>> combine_responses([r1, r2], mode="and")
    array([0.2, 0.3, 0.1])
    >>> combine_responses([r1, r2], weights=[0.7, 0.3], mode="weighted_sum")
    array([0.29, 0.65, 0.34])
    """
    if len(responses) == 0:
        raise ValueError("responses list cannot be empty")
    if len(responses) == 1:
        return responses[0]

    if mode == "or":
        return np.maximum.reduce(responses)
    elif mode == "and":
        return np.minimum.reduce(responses)
    elif mode in ("weighted_sum", "weighted_or"):
        # Handle weights
        if weights is None:
            weights = [1.0 / len(responses)] * len(responses)
        if len(weights) != len(responses):
            raise ValueError(
                f"Number of weights ({len(weights)}) must match "
                f"number of responses ({len(responses)})"
            )
        if mode == "weighted_sum":
            total = np.zeros_like(responses[0])
            for r, w in zip(responses, weights):
                total += r * w
            return np.clip(total, 0, 1)
        else:  # weighted_or
            weighted = [r * w for r, w in zip(responses, weights)]
            return np.maximum.reduce(weighted)
    else:
        raise ValueError(
            f"Unknown combination mode: {mode}. "
            "Use 'or', 'and', 'weighted_sum', or 'weighted_or'."
        )
