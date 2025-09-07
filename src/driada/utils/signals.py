"""
Signal Processing Utilities
===========================

This module contains utility functions for signal generation, analysis,
and filtering. It consolidates functionality from the former signals module.

Functions
---------
brownian : Generate Brownian motion (Wiener process)
approximate_entropy : Calculate approximate entropy of a signal
filter_1d_timeseries : Filter a 1D time series using various methods
filter_signals : Filter multiple signals (2D array)
adaptive_filter_signals : Adaptively filter based on SNR
"""

import numpy as np
from scipy.stats import norm
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import pywt
from typing import Optional, Union, List

try:
    from .data import check_positive, check_nonnegative
except ImportError:
    # For standalone module execution (e.g., doctests)
    from driada.utils.data import check_positive, check_nonnegative


def brownian(
    x0: Union[float, np.ndarray],
    n: int,
    dt: float = 1.0,
    delta: float = 1.0,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Generate an instance of Brownian motion (i.e. the Wiener process).

    The Brownian motion follows:
        X(t) = X(0) + N(0, delta**2 * t; 0, t)

    where N(a,b; t0, t1) is a normally distributed random variable with mean a
    and variance b. The parameters t0 and t1 make explicit the statistical
    independence of N on different time intervals.

    Written as an iteration scheme:
        X(t + dt) = X(t) + N(0, delta**2 * dt; t, t+dt)

    Parameters
    ----------
    x0 : float or np.ndarray
        The initial condition(s) (i.e. position(s)) of the Brownian motion.
        If array, each value is treated as an initial condition.
    n : int
        The number of steps to take.
    dt : float, optional
        The time step. Default is 1.0.
    delta : float, optional
        Determines the "speed" of the Brownian motion. The random variable
        of the position at time t, X(t), has a normal distribution whose mean
        is the position at time t=0 and whose variance is delta**2*t.
        Default is 1.0.
    out : np.ndarray, optional
        If provided, specifies the array in which to put the result.
        If None, a new numpy array is created and returned.

    Returns
    -------
    np.ndarray
        Array of floats with shape `x0.shape + (n,)`.
        Note that the initial value `x0` is not included in the returned array.

    Raises
    ------
    ValueError
        If n <= 0, dt <= 0, or delta < 0

    Examples
    --------
    >>> # Generate single Brownian motion path
    >>> path = brownian(0.0, 1000, dt=0.01)
    >>> path.shape
    (1000,)

    >>> # Generate multiple paths with different initial conditions
    >>> paths = brownian([0.0, 1.0, -1.0], 1000, dt=0.01)
    >>> paths.shape
    (3, 1000)    """
    # Validate inputs
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError(f"n must be a positive integer, got {n}")
    check_positive(dt=dt)
    check_nonnegative(delta=delta)
    
    x0 = np.asarray(x0)

    # For each element of x0, generate a sample of n numbers from a
    # normal distribution.
    r = norm.rvs(size=x0.shape + (n,), scale=delta * np.sqrt(dt))

    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)

    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples.
    np.cumsum(r, axis=-1, out=out)

    # Add the initial condition.
    out += np.expand_dims(x0, axis=-1)

    return out


def approximate_entropy(U: Union[List, np.ndarray], m: int, r: float) -> float:
    """
    Calculate approximate entropy (ApEn) of a signal.

    Approximate entropy is a regularity statistic that quantifies the
    unpredictability of fluctuations in a time series. A time series
    containing many repetitive patterns has a relatively small ApEn;
    a less predictable process has a higher ApEn.

    Parameters
    ----------
    U : array-like
        Input signal/time series. Must have length >= m + 2.
    m : int
        Pattern length. Common values are 1 or 2.
    r : float
        Tolerance threshold for pattern matching. Typically 0.1-0.25
        times the standard deviation of the data.

    Returns
    -------
    float
        The approximate entropy value. Higher values indicate more
        randomness/complexity.

    Raises
    ------
    ValueError
        If length of U < m + 2, m < 1, or r < 0

    Notes
    -----
    The algorithm:
    1. Fix m (pattern length) and r (tolerance)
    2. Look at patterns of length m and m+1
    3. Count pattern matches within tolerance r
    4. Calculate the logarithmic frequency of patterns
    5. Return the difference between m and m+1 pattern frequencies
    
    Complexity is O(N²). For long signals consider downsampling.

    References
    ----------
    Pincus, S. M. (1991). Approximate entropy as a measure of system
    complexity. Proceedings of the National Academy of Sciences, 88(6),
    2297-2301.

    Examples
    --------
    >>> # Regular signal has low entropy
    >>> regular = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    >>> apen = approximate_entropy(regular, m=2, r=0.5)
    >>> apen < 0.1
    True

    >>> # Random signal has high entropy
    >>> import numpy as np
    >>> random_signal = np.random.randn(100)
    >>> apen = approximate_entropy(random_signal, m=2, r=0.2 * np.std(random_signal))
    >>> apen > 0.5  # Typically true for random signals
    True    """

    # Validate inputs
    U = np.asarray(U)
    N = len(U)
    
    if N < m + 2:
        raise ValueError(f"Signal length ({N}) must be >= m + 2 ({m + 2})")
    if m < 1:
        raise ValueError(f"Pattern length m must be >= 1, got {m}")
    check_nonnegative(r=r)
    
    def _maxdist(x_i, x_j):
        """Calculate maximum distance between two patterns.
        
        Parameters
        ----------
        x_i : list
            First pattern of values
        x_j : list
            Second pattern of values
            
        Returns
        -------
        float
            Maximum absolute difference between corresponding elements
        """
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        """Calculate phi(m) - the logarithmic frequency of patterns.
        
        Parameters
        ----------
        m : int
            Pattern length
            
        Returns
        -------
        float
            Average logarithm of matching pattern frequencies
        """
        # Extract all patterns of length m
        patterns = [[U[j] for j in range(i, i + m)] for i in range(N - m + 1)]

        # Count matches for each pattern
        C = [
            len([1 for x_j in patterns if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0)
            for x_i in patterns
        ]

        # Avoid log(0) by adding small epsilon
        phi_sum = 0.0
        for c in C:
            if c > 0:
                phi_sum += np.log(c)
            else:
                # When no patterns match, use a very small probability
                phi_sum += np.log(1e-10)
        
        return phi_sum / (N - m + 1.0)

    # Approximate entropy is the difference between phi(m+1) and phi(m)
    return abs(_phi(m + 1) - _phi(m))


def filter_1d_timeseries(
    data: np.ndarray, method: str = "gaussian", **kwargs
) -> np.ndarray:
    """
    Filter a 1D time series using various denoising methods.

    This is the core filtering function used by TimeSeries.filter() method.
    Supports Gaussian smoothing, Savitzky-Golay filtering, and wavelet denoising.

    Parameters
    ----------
    data : ndarray
        1D time series data
    method : str
        Filtering method: 'gaussian', 'savgol', 'wavelet', or 'none'
    **kwargs : dict
        Method-specific parameters:
        - gaussian: sigma (default: 1.0) - standard deviation for Gaussian kernel
        - savgol: window_length (default: 5), polyorder (default: 2)
        - wavelet: wavelet (default: 'db4'), level (default: auto),
                  mode (default: 'smooth'), threshold_method (default: 'mad')

    Returns
    -------
    ndarray
        Filtered 1D time series

    Raises
    ------
    ValueError
        If unknown method or invalid parameters (e.g., polyorder >= window_length)

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> # Create noisy signal
    >>> t = np.linspace(0, 1, 100)
    >>> data = np.sin(2 * np.pi * 5 * t) + 0.2 * np.random.randn(100)
    
    >>> # Gaussian smoothing for general noise reduction
    >>> filtered = filter_1d_timeseries(data, method='gaussian', sigma=1.5)
    >>> filtered.shape
    (100,)

    >>> # Savitzky-Golay for preserving peaks while smoothing
    >>> filtered = filter_1d_timeseries(data, method='savgol', window_length=5)
    >>> filtered.shape
    (100,)

    >>> # Wavelet denoising for multi-scale noise removal
    >>> filtered = filter_1d_timeseries(data, method='wavelet', wavelet='db4')
    >>> filtered.shape
    (100,)    """
    if method == "none":
        return data.copy()

    # Ensure we have a 1D array
    data = np.asarray(data).ravel()

    if method == "gaussian":
        # Gaussian filter: good for general smoothing
        sigma = kwargs.get("sigma", 1.0)
        return gaussian_filter1d(data, sigma=sigma)

    elif method == "savgol":
        # Savitzky-Golay: preserves features like peaks better than Gaussian
        window_length = kwargs.get("window_length", 5)
        polyorder = kwargs.get("polyorder", 2)

        # Ensure window_length is odd
        if window_length % 2 == 0:
            window_length += 1

        # Check if signal is long enough
        if len(data) <= window_length:
            return data.copy()
        
        # Validate polyorder
        if polyorder >= window_length:
            raise ValueError(f"polyorder ({polyorder}) must be less than window_length ({window_length})")

        return savgol_filter(data, window_length, polyorder)

    elif method == "wavelet":
        # Wavelet denoising: excellent for multi-scale noise removal
        wavelet = kwargs.get("wavelet", "db4")  # Daubechies 4 is a good default
        level = kwargs.get("level", None)
        mode = kwargs.get("mode", "smooth")  # boundary handling
        threshold_method = kwargs.get("threshold_method", "mad")

        n = len(data)

        # Determine decomposition level if not specified
        max_level = pywt.dwt_max_level(n, wavelet)
        if level is None:
            # Automatic level selection: balance between noise removal and signal preservation
            level = min(max_level, max(1, int(np.log2(n)) - 4))
        elif level > max_level:
            level = max_level

        # Perform wavelet decomposition
        coeffs = pywt.wavedec(data, wavelet, mode=mode, level=level)

        # Apply thresholding to detail coefficients (not the approximation)
        if threshold_method == "mad":
            # MAD-based threshold: robust to outliers
            for j in range(1, len(coeffs)):
                detail_coeffs = coeffs[j]
                # Estimate noise level using Median Absolute Deviation
                sigma = np.median(np.abs(detail_coeffs)) / 0.6745
                # Universal threshold (Donoho & Johnstone)
                threshold = sigma * np.sqrt(2 * np.log(n))
                # Soft thresholding: shrinks coefficients smoothly
                coeffs[j] = pywt.threshold(detail_coeffs, threshold, mode="soft")

        elif threshold_method == "sure":
            # SURE (Stein's Unbiased Risk Estimate): data-adaptive threshold
            for j in range(1, len(coeffs)):
                threshold = np.std(coeffs[j]) * np.sqrt(2 * np.log(len(coeffs[j])))
                coeffs[j] = pywt.threshold(coeffs[j], threshold, mode="soft")

        # Reconstruct the signal
        reconstructed = pywt.waverec(coeffs, wavelet, mode=mode)

        # Handle potential length mismatch
        return reconstructed[:n]

    else:
        raise ValueError(
            f"Unknown filtering method: {method}. "
            f"Choose from: 'gaussian', 'savgol', 'wavelet', 'none'"
        )


def filter_signals(data: np.ndarray, method: str = "gaussian", **kwargs) -> np.ndarray:
    """
    Filter multiple signals (2D array compatibility wrapper).

    Parameters
    ----------
    data : ndarray
        Data with shape (n_signals, n_timepoints) or 1D array
    method : str
        Filtering method: 'gaussian', 'savgol', 'wavelet', or 'none'
    **kwargs : dict
        Method-specific parameters (see filter_1d_timeseries)

    Returns
    -------
    ndarray
        Filtered data with same shape as input

    Raises
    ------
    ValueError
        If unknown method or invalid parameters
    
    Examples
    --------
    >>> # Filter multiple signals
    >>> signals = np.random.randn(10, 1000)  # 10 signals, 1000 points each
    >>> filtered = filter_signals(signals, method='gaussian', sigma=2.0)
    
    >>> # Also works with 1D arrays
    >>> signal = np.random.randn(1000)
    >>> filtered = filter_signals(signal, method='savgol')    """
    if data.ndim == 1:
        return filter_1d_timeseries(data, method=method, **kwargs)

    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        filtered_data[i] = filter_1d_timeseries(data[i], method=method, **kwargs)

    return filtered_data


def adaptive_filter_signals(data: np.ndarray, snr_threshold: float = 2.0) -> np.ndarray:
    """
    Adaptively filter signals based on signal-to-noise ratio.

    Parameters
    ----------
    data : ndarray
        Data with shape (n_signals, n_timepoints)
    snr_threshold : float
        SNR threshold for determining filter strength. Default 2.0.

    Returns
    -------
    ndarray
        Adaptively filtered data with same shape as input

    Raises
    ------
    ValueError
        If data is not 2D or snr_threshold <= 0

    Notes
    -----
    Uses simple binary threshold: strong filtering (sigma=2.0) for 
    low SNR, light filtering (sigma=0.5) for high SNR.
    
    Examples
    --------
    >>> # Adaptively filter based on estimated SNR
    >>> signals = np.random.randn(5, 1000)
    >>> filtered = adaptive_filter_signals(signals, snr_threshold=3.0)
    
    >>> # Lower threshold applies stronger filtering to more signals
    >>> filtered = adaptive_filter_signals(signals, snr_threshold=1.0)    """
    # Validate inputs
    if data.ndim != 2:
        raise ValueError(f"Data must be 2D, got shape {data.shape}")
    check_positive(snr_threshold=snr_threshold)
    
    filtered_data = np.zeros_like(data)

    for i in range(data.shape[0]):
        signal = data[i]

        # Estimate SNR (simple approach)
        signal_power = np.var(signal)
        noise_power = np.var(np.diff(signal))  # High-freq noise estimate
        snr = signal_power / (noise_power + 1e-10)

        # Adaptive filtering based on SNR
        if snr < snr_threshold:
            # High noise - stronger filtering
            sigma = 2.0
        else:
            # Low noise - lighter filtering
            sigma = 0.5

        filtered_data[i] = gaussian_filter1d(signal, sigma=sigma)

    return filtered_data
