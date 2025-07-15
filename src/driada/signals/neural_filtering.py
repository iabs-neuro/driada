"""
Neural signal filtering utilities for manifold analysis.

This module provides filtering functions to preprocess neural signals
before applying dimensionality reduction and manifold analysis.
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d


def filter_neural_signals(neural_data, method='gaussian', **kwargs):
    """Filter neural signals to improve manifold analysis
    
    Parameters:
    -----------
    neural_data : ndarray
        Neural data with shape (n_neurons, n_timepoints)
    method : str
        Filtering method: 'gaussian', 'savgol', or 'none'
    **kwargs : dict
        Method-specific parameters
        
    Returns:
    --------
    ndarray
        Filtered neural data with same shape as input
        
    Examples:
    ---------
    >>> # Gaussian filtering
    >>> filtered = filter_neural_signals(data, method='gaussian', sigma=1.5)
    
    >>> # Savitzky-Golay filtering
    >>> filtered = filter_neural_signals(data, method='savgol', 
    ...                                 window_length=5, polyorder=2)
    """
    if method == 'none':
        return neural_data
    
    filtered_data = np.zeros_like(neural_data)
    
    if method == 'gaussian':
        sigma = kwargs.get('sigma', 1.0)
        for i in range(neural_data.shape[0]):
            filtered_data[i] = gaussian_filter1d(neural_data[i], sigma=sigma)
            
    elif method == 'savgol':
        window_length = kwargs.get('window_length', 5)
        polyorder = kwargs.get('polyorder', 2)
        # Ensure window_length is odd and valid
        if window_length % 2 == 0:
            window_length += 1
        
        for i in range(neural_data.shape[0]):
            if len(neural_data[i]) > window_length:
                filtered_data[i] = savgol_filter(neural_data[i], window_length, polyorder)
            else:
                filtered_data[i] = neural_data[i]
    else:
        raise ValueError(f"Unknown filtering method: {method}")
    
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