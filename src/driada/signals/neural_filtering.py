"""
Neural signal filtering utilities for manifold analysis.

This module provides filtering functions to preprocess neural signals
before applying dimensionality reduction and manifold analysis.
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import pywt


def filter_1d_timeseries(data, method='gaussian', **kwargs):
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
        
    Examples
    --------
    >>> # Gaussian smoothing for general noise reduction
    >>> filtered = filter_1d_timeseries(data, method='gaussian', sigma=1.5)
    
    >>> # Savitzky-Golay for preserving peaks while smoothing
    >>> filtered = filter_1d_timeseries(data, method='savgol', window_length=5)
    
    >>> # Wavelet denoising for multi-scale noise removal
    >>> filtered = filter_1d_timeseries(data, method='wavelet', wavelet='db4')
    """
    if method == 'none':
        return data.copy()
    
    # Ensure we have a 1D array
    data = np.asarray(data).ravel()
    
    if method == 'gaussian':
        # Gaussian filter: good for general smoothing
        sigma = kwargs.get('sigma', 1.0)
        return gaussian_filter1d(data, sigma=sigma)
        
    elif method == 'savgol':
        # Savitzky-Golay: preserves features like peaks better than Gaussian
        window_length = kwargs.get('window_length', 5)
        polyorder = kwargs.get('polyorder', 2)
        
        # Ensure window_length is odd
        if window_length % 2 == 0:
            window_length += 1
            
        # Check if signal is long enough
        if len(data) <= window_length:
            return data.copy()
            
        return savgol_filter(data, window_length, polyorder)
        
    elif method == 'wavelet':
        # Wavelet denoising: excellent for multi-scale noise removal
        wavelet = kwargs.get('wavelet', 'db4')  # Daubechies 4 is a good default
        level = kwargs.get('level', None)
        mode = kwargs.get('mode', 'smooth')  # boundary handling
        threshold_method = kwargs.get('threshold_method', 'mad')
        
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
        if threshold_method == 'mad':
            # MAD-based threshold: robust to outliers
            for j in range(1, len(coeffs)):
                detail_coeffs = coeffs[j]
                # Estimate noise level using Median Absolute Deviation
                sigma = np.median(np.abs(detail_coeffs)) / 0.6745
                # Universal threshold (Donoho & Johnstone)
                threshold = sigma * np.sqrt(2 * np.log(n))
                # Soft thresholding: shrinks coefficients smoothly
                coeffs[j] = pywt.threshold(detail_coeffs, threshold, mode='soft')
                
        elif threshold_method == 'sure':
            # SURE (Stein's Unbiased Risk Estimate): data-adaptive threshold
            for j in range(1, len(coeffs)):
                threshold = np.std(coeffs[j]) * np.sqrt(2 * np.log(len(coeffs[j])))
                coeffs[j] = pywt.threshold(coeffs[j], threshold, mode='soft')
        
        # Reconstruct the signal
        reconstructed = pywt.waverec(coeffs, wavelet, mode=mode)
        
        # Handle potential length mismatch
        return reconstructed[:n]
        
    else:
        raise ValueError(f"Unknown filtering method: {method}. "
                        f"Choose from: 'gaussian', 'savgol', 'wavelet', 'none'")


def filter_neural_signals(neural_data, method='gaussian', **kwargs):
    """
    Filter multiple neural signals (2D array compatibility wrapper).
    
    Parameters
    ----------
    neural_data : ndarray
        Neural data with shape (n_neurons, n_timepoints)
    method : str
        Filtering method: 'gaussian', 'savgol', 'wavelet', or 'none'
    **kwargs : dict
        Method-specific parameters (see filter_1d_timeseries)
        
    Returns
    -------
    ndarray
        Filtered neural data with same shape as input
    """
    if neural_data.ndim == 1:
        return filter_1d_timeseries(neural_data, method=method, **kwargs)
    
    filtered_data = np.zeros_like(neural_data)
    for i in range(neural_data.shape[0]):
        filtered_data[i] = filter_1d_timeseries(neural_data[i], method=method, **kwargs)
    
    return filtered_data


def adaptive_filter_neural_signals(neural_data, snr_threshold=2.0):
    """Adaptively filter neural signals based on signal-to-noise ratio
    
    Parameters:
    -----------
    neural_data : ndarray
        Neural data with shape (n_neurons, n_timepoints)
    snr_threshold : float
        SNR threshold for determining filter strength
        
    Returns:
    --------
    ndarray
        Adaptively filtered neural data
    """
    filtered_data = np.zeros_like(neural_data)
    
    for i in range(neural_data.shape[0]):
        signal = neural_data[i]
        
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


def manifold_preprocessing(neural_data, method='adaptive', **kwargs):
    """Preprocess neural data specifically for manifold analysis
    
    Parameters:
    -----------
    neural_data : ndarray
        Neural data with shape (n_neurons, n_timepoints)
    method : str
        Preprocessing method: 'adaptive', 'gaussian', 'savgol', or 'none'
    **kwargs : dict
        Method-specific parameters
        
    Returns:
    --------
    ndarray
        Preprocessed neural data ready for manifold analysis
    """
    if method == 'adaptive':
        return adaptive_filter_neural_signals(neural_data, **kwargs)
    else:
        return filter_neural_signals(neural_data, method=method, **kwargs)