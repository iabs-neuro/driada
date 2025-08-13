"""
Spike reconstruction module for DRIADA.

This module provides functions for reconstructing spike trains from calcium
imaging data using various methods.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from ..information.info_base import TimeSeries, MultiTimeSeries
from .wavelet_event_detection import (
    WVT_EVENT_DETECTION_PARAMS,
    extract_wvt_events,
    events_to_ts_array,
    ridges_to_containers,
)


def reconstruct_spikes(
    calcium: MultiTimeSeries,
    method: str = "wavelet",
    fps: float = 20.0,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[MultiTimeSeries, Dict[str, Any]]:
    """
    Reconstruct spike trains from calcium signals.

    Parameters
    ----------
    calcium : MultiTimeSeries
        Calcium imaging data with each component being a neuron
    method : str or callable
        Reconstruction method: 'wavelet', 'threshold', or callable
    fps : float
        Sampling rate in frames per second
    params : dict, optional
        Method-specific parameters

    Returns
    -------
    spikes : MultiTimeSeries
        Reconstructed spike trains (discrete)
    metadata : dict
        Reconstruction metadata
    """
    params = params or {}

    if callable(method):
        # Custom method
        return method(calcium, fps, params)

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

    Parameters
    ----------
    calcium : MultiTimeSeries
        Calcium signals
    fps : float
        Sampling rate
    params : dict
        Wavelet parameters

    Returns
    -------
    spikes : MultiTimeSeries
        Spike trains
    metadata : dict
        Reconstruction metadata
    """
    # Get scaled calcium data as numpy array for better spike detection
    calcium_data = np.asarray(calcium.scdata)  # Use scaled data

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
    spikes = MultiTimeSeries(spike_ts_list)

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
    calcium: MultiTimeSeries, fps: float, params: Dict[str, Any]
) -> Tuple[MultiTimeSeries, Dict[str, Any]]:
    """
    Simple threshold-based spike reconstruction.

    This method detects spikes when the derivative of the calcium signal
    exceeds a threshold, similar to classical spike detection methods.

    Parameters
    ----------
    calcium : MultiTimeSeries
        Calcium signals
    fps : float
        Sampling rate
    params : dict
        Parameters including:
        - threshold_std : float, number of STDs above mean for detection (default: 2.5)
        - smooth_sigma : float, gaussian smoothing sigma in frames (default: 2)
        - min_spike_interval : float, minimum interval between spikes in seconds (default: 0.1)

    Returns
    -------
    spikes : MultiTimeSeries
        Binary spike trains
    metadata : dict
        Reconstruction metadata
    """
    # Default parameters
    threshold_std = params.get("threshold_std", 2.5)
    smooth_sigma = params.get("smooth_sigma", 2)
    min_spike_interval = params.get("min_spike_interval", 0.1)
    min_spike_frames = int(min_spike_interval * fps)

    calcium_data = np.asarray(calcium.scdata)  # Use scaled data
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
    spikes = MultiTimeSeries(spike_ts_list)

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
