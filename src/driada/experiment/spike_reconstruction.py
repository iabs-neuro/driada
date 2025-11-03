"""
Spike reconstruction module for DRIADA.

This module provides functions for reconstructing spike trains from calcium
imaging data using various methods.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, Union, List
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from ..information.info_base import TimeSeries, MultiTimeSeries
from .wavelet_event_detection import (
    WVT_EVENT_DETECTION_PARAMS,
    extract_wvt_events,
    events_to_ts_array,
    ridges_to_containers,
)
from ..utils.data import check_positive, check_nonnegative


def reconstruct_spikes(
    calcium: MultiTimeSeries,
    method: str = "wavelet",
    fps: float = 20.0,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[MultiTimeSeries, Dict[str, Any]]:
    """
    Reconstruct spike trains from calcium signals.
    
    This function serves as a router to different spike reconstruction methods.
    All methods operate on the scaled calcium data (values normalized to [0, 1]).

    Parameters
    ----------
    calcium : MultiTimeSeries
        Calcium imaging data with each component being a neuron. Must have
        valid scaled data accessible via calcium.scdata.
    method : str or callable, optional
        Reconstruction method. Options:
        - 'wavelet': Wavelet-based detection (default)
        - 'threshold': Simple threshold-based detection  
        - callable: Custom function with signature (calcium, fps, params) -> (spikes, metadata)
    fps : float, optional
        Sampling rate in frames per second. Must be positive. Default: 20.0.
    params : dict, optional
        Method-specific parameters. Contents depend on chosen method.
        Default: empty dict.

    Returns
    -------
    spikes : MultiTimeSeries
        Reconstructed spike trains as binary (discrete) time series.
        Each component represents one neuron.
    metadata : dict
        Reconstruction metadata including:
        - 'method': str - Method used
        - 'parameters': dict - Parameters used
        - Method-specific fields (see individual method docs)
        
    Raises
    ------
    ValueError
        If method is unknown string.
        If fps is not positive.
    AttributeError
        If calcium lacks required attributes (e.g., scdata).
    TypeError  
        If callable method has wrong signature.
        
    Examples
    --------
    >>> # Create example calcium data
    >>> import numpy as np
    >>> from driada.information import TimeSeries, MultiTimeSeries
    >>> n_neurons, n_frames = 5, 1000
    >>> raw_data = np.random.rand(n_neurons, n_frames)
    >>> calcium_ts_list = [TimeSeries(raw_data[i], discrete=False) for i in range(n_neurons)]
    >>> calcium_data = MultiTimeSeries(calcium_ts_list)
    >>> 
    >>> # Using wavelet method (default)
    >>> spikes, meta = reconstruct_spikes(calcium_data, fps=30.0)
    >>> meta['method']
    'wavelet'
    >>> spikes.n_dim == n_neurons
    True
    
    >>> # Using threshold method with custom parameters
    >>> params = {'threshold_std': 3.0, 'smooth_sigma': 1.5}
    >>> spikes, meta = reconstruct_spikes(calcium_data, 'threshold', 30.0, params)
    >>> meta['method']
    'threshold'
    >>> meta['parameters']['threshold_std']
    3.0
    
    >>> # Using custom reconstruction function
    >>> def custom_method(calcium, fps, params):
    ...     # Simple mock implementation
    ...     n_neurons, n_frames = calcium.scdata.shape
    ...     spike_data = np.zeros((n_neurons, n_frames))
    ...     # Add a few spikes to avoid zero columns error
    ...     for i in range(n_neurons):
    ...         spike_data[i, i*10:(i+1)*10] = 1
    ...     spike_ts = [TimeSeries(spike_data[i], discrete=True) for i in range(n_neurons)]
    ...     return MultiTimeSeries(spike_ts, allow_zero_columns=True), {'custom_info': 'test'}
    >>> spikes, meta = reconstruct_spikes(calcium_data, custom_method, 30.0)
    >>> meta['custom_info']
    'test'
    
    Notes
    -----
    All built-in methods use the scaled calcium data (calcium.scdata) which
    is normalized to [0, 1]. This ensures consistent behavior across different
    calcium indicator types and experimental conditions.    """
    # Input validation
    check_positive(fps=fps)
    
    # Validate calcium has required attributes
    if not hasattr(calcium, 'scdata'):
        raise AttributeError("calcium must have 'scdata' attribute (scaled data)")
        
    params = params or {}

    if callable(method):
        # Custom method
        try:
            return method(calcium, fps, params)
        except TypeError as e:
            raise TypeError(
                f"Custom method must have signature (calcium, fps, params) -> "
                f"(MultiTimeSeries, dict). Error: {e}"
            )

    elif method == "wavelet":
        return wavelet_reconstruction(calcium, fps, params)

    elif method == "threshold":
        return threshold_reconstruction(calcium, fps, params)

    else:
        raise ValueError(
            f"Unknown method '{method}'. Use 'wavelet', 'threshold', "
            f"or provide a callable."
        )


def wavelet_reconstruction(
    calcium: MultiTimeSeries, fps: float, params: Dict[str, Any]
) -> Tuple[MultiTimeSeries, Dict[str, Any]]:
    """
    Wavelet-based spike reconstruction.
    
    Uses continuous wavelet transform to detect calcium transients. The method
    operates on scaled calcium data (normalized to [0, 1]).

    Parameters
    ----------
    calcium : MultiTimeSeries
        Calcium imaging signals. Must have scdata attribute.
    fps : float
        Sampling rate in frames per second. Overrides default fps in parameters.
    params : dict
        Parameters that update WVT_EVENT_DETECTION_PARAMS defaults:
        - 'sigma': int - Smoothing parameter for peak detection (frames)
        - 'eps': int - Minimum spacing between consecutive events (frames)
        - 'scale_length_thr': int - Min scales where ridge is present
        - 'max_scale_thr': int - Index of scale with max ridge intensity
        - 'max_ampl_thr': float - Max ridge intensity threshold
        - 'max_dur_thr': int - Max event duration threshold
        See WVT_EVENT_DETECTION_PARAMS for defaults.

    Returns
    -------
    spikes : MultiTimeSeries
        Binary spike trains (discrete). Allow_zero_columns=True for empty neurons.
    metadata : dict
        Contains:
        - 'method': 'wavelet'
        - 'parameters': dict - All parameters used
        - 'start_events': list - Event start indices per neuron
        - 'end_events': list - Event end indices per neuron
        - 'ridges': list - Ridge information per neuron
        
    Raises
    ------
    AttributeError
        If calcium lacks scdata attribute.
    ValueError
        If calcium data is empty or invalid shape.
        
    Notes
    -----
    Default parameters are defined in WVT_EVENT_DETECTION_PARAMS. The fps
    parameter always overrides the default fps value.    """
    # Input validation
    check_positive(fps=fps)
    
    if not hasattr(calcium, 'scdata'):
        raise AttributeError("calcium must have 'scdata' attribute")
        
    # Get scaled calcium data as numpy array for better spike detection
    calcium_data = np.asarray(calcium.scdata)  # Use scaled data
    
    # Validate data shape
    if calcium_data.ndim != 2:
        raise ValueError(f"calcium data must be 2D (neurons x time), got shape {calcium_data.shape}")
    if calcium_data.size == 0:
        raise ValueError("calcium data cannot be empty")

    # Set up wavelet parameters
    wvt_kwargs = WVT_EVENT_DETECTION_PARAMS.copy()
    wvt_kwargs["fps"] = fps
    wvt_kwargs.update(params)

    # Extract events
    st_ev_inds, end_ev_inds, all_ridges = extract_wvt_events(calcium_data, wvt_kwargs)

    # Convert to spike array
    spikes_data = events_to_ts_array(
        calcium_data.shape[1], st_ev_inds, end_ev_inds, fps
    )

    # Create spike MultiTimeSeries
    spike_ts_list = [
        TimeSeries(spikes_data[i, :], discrete=True)
        for i in range(spikes_data.shape[0])
    ]
    spikes = MultiTimeSeries(spike_ts_list, allow_zero_columns=True)

    # Prepare metadata
    metadata = {
        "method": "wavelet",
        "parameters": wvt_kwargs,
        "start_events": st_ev_inds,
        "end_events": end_ev_inds,
        "ridges": [ridges_to_containers(ridges) for ridges in all_ridges],
    }

    return spikes, metadata


def threshold_reconstruction(
    calcium: Union[MultiTimeSeries, TimeSeries], fps: float, params: Dict[str, Any]
) -> Tuple[MultiTimeSeries, Dict[str, Any]]:
    """
    Simple threshold-based spike reconstruction.

    This method detects spikes when the derivative of the calcium signal
    exceeds a threshold. Operates on scaled calcium data (normalized to [0, 1]).

    Parameters
    ----------
    calcium : MultiTimeSeries or TimeSeries
        Calcium signals. Must have scdata attribute.
        If TimeSeries, will be treated as single neuron.
    fps : float
        Sampling rate in frames per second. Must be positive.
    params : dict
        Parameters including:

        * threshold_std : float, number of STDs above mean for detection. Must be positive. Default: 2.5.
        * smooth_sigma : float, gaussian smoothing sigma in frames. Must be non-negative. Default: 2.
        * min_spike_interval : float, minimum interval between spikes in seconds. Must be non-negative. Default: 0.1.

    Returns
    -------
    spikes : MultiTimeSeries
        Binary spike trains (discrete). Allow_zero_columns=True for empty neurons.
    metadata : dict
        Contains:
        - 'method': 'threshold'
        - 'parameters': dict with all parameters used including fps
        - 'spike_times': list of arrays - Frame indices of detected spikes per neuron

    Raises
    ------
    AttributeError
        If calcium lacks scdata attribute.
    ValueError
        If calcium data is empty or invalid shape.
        If parameters are out of valid range.
        If min_spike_interval * fps < 0.5 (would result in zero minimum distance).

    Notes
    -----
    The derivative is computed with np.diff and zero-padded at the start.
    This affects the first frame which cannot have a spike detected.    """
    # Input validation
    check_positive(fps=fps)

    if not hasattr(calcium, 'scdata'):
        raise AttributeError("calcium must have 'scdata' attribute")

    # Default parameters
    threshold_std = params.get("threshold_std", 2.5)
    smooth_sigma = params.get("smooth_sigma", 2)
    min_spike_interval = params.get("min_spike_interval", 0.1)

    # Validate parameters
    check_positive(threshold_std=threshold_std)
    check_nonnegative(smooth_sigma=smooth_sigma, min_spike_interval=min_spike_interval)

    min_spike_frames = int(min_spike_interval * fps)
    if min_spike_interval > 0 and min_spike_frames < 1:
        raise ValueError(f"min_spike_interval * fps = {min_spike_interval * fps:.2f} < 1, "
                         "would result in zero minimum distance between spikes")

    calcium_data = np.asarray(calcium.scdata)  # Use scaled data

    # Normalize to 2D array (neurons x time)
    if calcium_data.ndim == 1:
        calcium_data = calcium_data.reshape(1, -1)
    elif calcium_data.ndim != 2:
        raise ValueError(f"calcium data must be 1D or 2D, got shape {calcium_data.shape}")

    if calcium_data.size == 0:
        raise ValueError("calcium data cannot be empty")
        
    n_neurons, n_frames = calcium_data.shape
    spikes_data = np.zeros_like(calcium_data)

    all_spike_times = []

    for i in range(n_neurons):
        # Get calcium trace
        trace = calcium_data[i, :]

        # Smooth the signal
        smoothed = gaussian_filter1d(trace, sigma=smooth_sigma)

        # Compute derivative (rate of calcium increase)
        diff = np.diff(smoothed)
        diff = np.concatenate([[0], diff])  # Pad to maintain size

        # Compute threshold
        threshold = np.mean(diff) + threshold_std * np.std(diff)

        # Find peaks in derivative
        peaks, properties = find_peaks(
            diff, height=threshold, distance=min_spike_frames
        )

        # Mark spikes
        spikes_data[i, peaks] = 1
        all_spike_times.append(peaks)

    # Create spike MultiTimeSeries
    spike_ts_list = [
        TimeSeries(spikes_data[i, :], discrete=True) for i in range(n_neurons)
    ]
    spikes = MultiTimeSeries(spike_ts_list, allow_zero_columns=True)

    # Prepare metadata
    metadata = {
        "method": "threshold",
        "parameters": {
            "threshold_std": threshold_std,
            "smooth_sigma": smooth_sigma,
            "min_spike_interval": min_spike_interval,
            "fps": fps,
        },
        "spike_times": all_spike_times,
    }

    return spikes, metadata
