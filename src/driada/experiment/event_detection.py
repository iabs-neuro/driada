"""Calcium event detection and amplitude extraction.

This module provides functions for detecting calcium events, extracting amplitudes,
and converting events to point process representations.
"""

import numpy as np
from scipy.optimize import nnls

from ..utils.data import check_positive


# Default FPS and timing parameters
DEFAULT_FPS = 20
DEFAULT_MIN_BEHAVIOUR_TIME = 0.25

# Time-based constants for FPS-adaptive parameters (at reference 20 Hz)
# Phase 1: Critical detection parameters
MIN_CA_SHIFT_SEC = 0.25  # Minimum calcium shift in seconds (5 frames @ 20 Hz)
BASELINE_WINDOW_SEC = 1.0  # Baseline window in seconds (20 frames @ 20 Hz)
MAX_FRAMES_FORWARD_SEC = 5.0  # Max forward search for t_off (100 frames @ 20 Hz)
MAX_FRAMES_BACK_SEC = 1.5  # Max backward search for t_rise (30 frames @ 20 Hz)

# Phase 2: Robustness thresholds for kinetics measurement
MIN_DECAY_FRAMES_SEC = 0.5  # Minimum decay duration for t_off fitting (10 frames @ 20 Hz)
MIN_RISE_FRAMES_SEC = 0.5  # Minimum rise duration for t_rise fitting (10 frames @ 20 Hz)
MIN_VALID_POINTS_SEC = 0.25  # Minimum valid data points (5 frames @ 20 Hz)
SAVGOL_WINDOW_SEC = 0.25  # Savitzky-Golay smoothing window (5 frames @ 20 Hz)
BASELINE_OFFSET_SEC = 0.25  # Default baseline offset for fast indicators (5 frames @ 20 Hz)

# Legacy frame-based constant (deprecated, use MIN_CA_SHIFT_SEC * fps instead)
MIN_CA_SHIFT = 5


class SimpleEvent:
    """Lightweight container for calcium event boundaries.

    Minimal event representation compatible with optimize_kinetics().
    Unlike Ridge objects which track full wavelet information, SimpleEvent
    only stores event start/end times for fast threshold-based detection.

    Attributes
    ----------
    start : float
        Starting time index of the event (frame number).
    end : float
        Ending time index of the event (frame number).

    Examples
    --------
    >>> event = SimpleEvent(start=100.0, end=120.0)
    >>> event.start
    100.0
    >>> event.end
    120.0
    >>> event.duration
    20.0

    Notes
    -----
    This class provides the minimal interface required by optimize_kinetics():
    - ridge.start: event start frame
    - ridge.end: event end frame

    Performance comparison vs Ridge (wavelet detection):
    - Creation: ~100x faster (no wavelet transform)
    - Memory: ~10x smaller (no scale/amplitude arrays)
    - Detection: O(N) vs O(N²) for wavelet
    """

    def __init__(self, start, end):
        """Initialize event with start and end times.

        Parameters
        ----------
        start : float
            Starting frame index.
        end : float
            Ending frame index (inclusive).
        """
        self.start = float(start)
        self.end = float(end)

    @property
    def duration(self):
        """Event duration in frames."""
        return self.end - self.start

    def __repr__(self):
        return f"SimpleEvent(start={self.start:.1f}, end={self.end:.1f})"


def extract_event_amplitudes(
    ca_signal,
    st_ev_inds,
    end_ev_inds,
    baseline_window=20,
    already_dff=False,
    baseline_offset=0,
    baseline_offset_sec=None,
    use_peak_refinement=False,
    t_rise_frames=None,
    t_off_frames=None,
    fps=None,
    peak_search_window_sec=0.2,
):
    """Extract amplitudes from calcium signal for detected events.

    For raw fluorescence signals, applies dF/F0 normalization following
    Neugornet et al. 2021 (PMC8032960). For pre-normalized dF/F signals,
    directly extracts peak values.

    Parameters
    ----------
    ca_signal : ndarray
        Calcium signal (1D). Can be raw fluorescence or pre-normalized dF/F.
    st_ev_inds : list of int
        Start indices for each event.
    end_ev_inds : list of int
        End indices for each event.
    baseline_window : int, optional
        Number of frames before event to use for F0 calculation.
        Default is 20 frames.
    already_dff : bool, optional
        If True, signal is already dF/F normalized and peak values are
        extracted directly. If False, applies dF/F0 normalization.
        Default is False for backward compatibility.
    baseline_offset : int, optional
        Number of frames to skip before baseline window (to avoid rise phase).
        Baseline is calculated from frames [start-baseline_window-baseline_offset, start-baseline_offset].
        Default is 0 (baseline immediately before event).
        NOTE: Prefer baseline_offset_sec for FPS-independent specification.
    baseline_offset_sec : float, optional
        Alternative to baseline_offset: offset in seconds (converted to frames using fps).
        Useful for GCaMP indicators with non-zero rise time (~0.25s for GCaMP6f).
        If provided, overrides baseline_offset. Requires fps parameter.
    use_peak_refinement : bool, optional
        If True, refine peak location before calculating baseline (similar to 'onset' placement).
        Ensures baseline is calculated relative to TRUE peak, not wavelet detection start.
        Critical for overlapping events where wavelet detection timing is imprecise.
        Default is False for backward compatibility.
    t_rise_frames : float, optional
        Rise time in frames. Required if use_peak_refinement=True.
    t_off_frames : float, optional
        Decay time in frames. Required if use_peak_refinement=True.
    fps : float, optional
        Sampling rate in Hz. Required if use_peak_refinement=True.
    peak_search_window_sec : float, optional
        Time window in seconds to search for refined peak location.
        Only used if use_peak_refinement=True. Default is 0.2s.

    Returns
    -------
    amplitudes : list of float
        Amplitude for each event (dF/F0 if already_dff=False, peak dF/F if True).

    Notes
    -----
    For events at the start where baseline_window extends before 0,
    uses available data. Returns 0 amplitude for events with invalid
    F0 (zero or negative baseline when already_dff=False).

    When use_peak_refinement=True, searches for TRUE peak in expanded window
    (similar to 'onset' placement logic) and calculates baseline relative to
    refined peak instead of wavelet detection start. This is critical for
    overlapping events where baseline must avoid elevated signal from previous events.

    References
    ----------
    Neugornet A, O'Donovan B, Ortinski PI (2021). Comparative Effects of
    Event Detection Methods on the Analysis and Interpretation of Ca2+
    Imaging Data. Front Neurosci 15:620869.
    """
    if use_peak_refinement:
        if (t_rise_frames is None) or (t_off_frames is None) or (fps is None):
            raise ValueError(
                "t_rise_frames, t_off_frames, and fps required when use_peak_refinement=True"
            )

    # Convert baseline_offset_sec to frames if provided
    if baseline_offset_sec is not None:
        if fps is None:
            raise ValueError("fps parameter required when using baseline_offset_sec")
        baseline_offset = int(baseline_offset_sec * fps)

    ca_signal = np.asarray(ca_signal)
    amplitudes = []
    peak_search_frames = int(peak_search_window_sec * fps) if use_peak_refinement else 0
    for start, end in zip(st_ev_inds, end_ev_inds):
        event_segment = ca_signal[start:end]
        if len(event_segment) == 0:
            amplitudes.append(0)
            continue
        if use_peak_refinement:
            search_start = max(0, start - peak_search_frames)
            search_end = min(len(ca_signal), end + peak_search_frames)
            search_segment = ca_signal[search_start:search_end]
            true_peak_offset = np.argmax(search_segment)
            true_peak_idx = search_start + true_peak_offset
            reference_idx = true_peak_idx
            peak_value = ca_signal[true_peak_idx]
        else:
            reference_idx = start
            peak_value = np.max(event_segment)
        if already_dff:
            baseline_end = max(0, reference_idx - baseline_offset)
            baseline_start = max(0, baseline_end - baseline_window)
            baseline_segment = ca_signal[baseline_start:baseline_end]
            if len(baseline_segment) > 0:
                local_baseline = np.median(baseline_segment)
            else:
                local_baseline = 0
            amplitude = peak_value - local_baseline
        else:
            baseline_end = max(0, reference_idx - baseline_offset)
            baseline_start = max(0, baseline_end - baseline_window)
            baseline_segment = ca_signal[baseline_start:baseline_end]
            if len(baseline_segment) == 0:
                amplitudes.append(0)
                continue
            F0 = np.median(baseline_segment)
            if F0 <= 0:
                amplitudes.append(0)
                continue
            amplitude = (peak_value - F0) / F0
        amplitudes.append(max(0, amplitude))
    return amplitudes


def deconvolve_given_event_times(
    ca_signal, event_times, t_rise_frames, t_off_frames, event_mask=None
):
    """Extract amplitudes via non-negative least squares deconvolution.

    Given detected event times and known calcium kernel parameters, finds
    the optimal amplitudes that best reconstruct the observed signal.
    This solves: signal ≈ Σᵢ aᵢ · kernel(t - tᵢ)

    Handles overlapping events naturally by jointly optimizing all amplitudes
    to minimize reconstruction error.

    Parameters
    ----------
    ca_signal : ndarray
        Calcium signal (1D array, already dF/F normalized).
    event_times : array-like
        Event onset times as frame indices.
    t_rise_frames : float
        Rise time of calcium kernel in frames.
    t_off_frames : float
        Decay time of calcium kernel in frames.
    event_mask : ndarray of bool, optional
        Binary mask indicating frames to fit (typically expanded event regions).
        If provided, NNLS only fits these frames, reducing baseline noise
        contribution while covering full calcium transients.
        Default None fits all frames (legacy behavior).

    Returns
    -------
    amplitudes : ndarray
        Optimal amplitudes for each event (non-negative).

    Notes
    -----
    Uses scipy.optimize.nnls for non-negative least squares solution.

    When event_mask is provided, only masked frames contribute to the fit.
    The mask should cover full calcium transients (rise + decay), not just peaks.
    This reduces baseline noise while maintaining proper transient fitting.

    References
    ----------
    Lawson CL, Hanson RJ (1995). Solving Least Squares Problems.
    SIAM, Philadelphia.
    """
    ca_signal = np.asarray(ca_signal)
    event_times = np.asarray(event_times, dtype=int)
    n_frames = len(ca_signal)
    n_events = len(event_times)
    if n_events == 0:
        return np.array([])

    # Build design matrix with vectorized kernel generation
    K = np.zeros((n_frames, n_events))

    # Pre-compute normalized kernel template (maximum length we'll need)
    # This avoids recomputing exponentials for each event
    t_template = np.arange(n_frames)
    kernel_template = (1 - np.exp(-t_template / t_rise_frames)) * np.exp(-t_template / t_off_frames)
    kernel_max = np.max(kernel_template)
    if kernel_max > 0:
        kernel_template = kernel_template / kernel_max

    # Filter valid events and apply kernel template
    for i, event_time_idx in enumerate(event_times):
        if event_time_idx < 0 or event_time_idx >= n_frames:
            continue
        remaining_frames = n_frames - event_time_idx
        K[event_time_idx:, i] = kernel_template[:remaining_frames]

    # Apply event mask if provided
    if event_mask is not None:
        event_mask = np.asarray(event_mask, dtype=bool)
        if len(event_mask) != n_frames:
            raise ValueError(
                f"event_mask length ({len(event_mask)}) must match signal length ({n_frames})"
            )
        K_fit = K[event_mask, :]
        ca_fit = ca_signal[event_mask]
    else:
        K_fit = K
        ca_fit = ca_signal

    (amplitudes, residual_norm) = nnls(K_fit, ca_fit)
    return amplitudes


def compute_kernel_peak_offset(t_rise_frames, t_off_frames):
    """Compute the time offset from onset to peak for double-exponential kernel.

    The kernel (1 - exp(-t/t_rise)) * exp(-t/t_off) starts at 0, rises to
    a peak, then decays. This function returns the time (in frames) from
    onset (t=0) to peak.

    Parameters
    ----------
    t_rise_frames : float
        Rise time constant in frames.
    t_off_frames : float
        Decay time constant in frames.

    Returns
    -------
    float
        Time offset from onset to peak in frames. Returns 0 if kinetics
        are degenerate (t_rise ≈ t_off).
    """
    if t_off_frames <= t_rise_frames or abs(t_off_frames - t_rise_frames) < 0.1:
        return 0.0
    peak_offset = (
        t_rise_frames
        * t_off_frames
        * np.log(t_off_frames / t_rise_frames)
        / (t_off_frames - t_rise_frames)
    )
    if np.isnan(peak_offset) or np.isinf(peak_offset):
        return 0.0
    return peak_offset


def estimate_onset_times(ca_signal, st_inds, end_inds, t_rise_frames, t_off_frames):
    """Estimate spike onset times from detected event boundaries.

    Event detectors (wavelet, threshold) find the rising/peak region of
    calcium transients, not the true spike onset. This function estimates
    onset times by finding the peak within each event and back-calculating
    using the kernel peak offset.

    Parameters
    ----------
    ca_signal : ndarray
        Calcium signal (1D array).
    st_inds : list of int
        Event start frame indices (from detector).
    end_inds : list of int
        Event end frame indices (from detector).
    t_rise_frames : float
        Rise time constant in frames.
    t_off_frames : float
        Decay time constant in frames.

    Returns
    -------
    list of int
        Estimated onset times for each event.
    """
    ca_signal = np.asarray(ca_signal)
    n_frames = len(ca_signal)
    kernel_peak_offset = compute_kernel_peak_offset(t_rise_frames, t_off_frames)

    onset_times = []
    for st, end in zip(st_inds, end_inds):
        st_int, end_int = int(st), int(end)
        if end_int <= st_int or st_int < 0 or end_int > n_frames:
            onset_times.append(max(0, st_int))
            continue
        # Find peak within event boundaries
        event_segment = ca_signal[st_int:end_int]
        peak_offset = np.argmax(event_segment)
        peak_idx = st_int + peak_offset
        # Estimate onset by subtracting kernel peak offset
        onset = int(max(0, peak_idx - kernel_peak_offset))
        onset_times.append(onset)

    return onset_times


def amplitudes_to_point_events(
    length,
    ca_signal,
    st_ev_inds,
    end_ev_inds,
    amplitudes,
    placement="peak",
    t_rise_frames=None,
    t_off_frames=None,
    fps=None,
    peak_search_window_sec=0.2,
):
    """Convert event boundaries and amplitudes to point event array.

    Stores amplitudes at specific positions as delta functions. Temporal
    spreading is handled by convolution with calcium kernel.

    Parameters
    ----------
    length : int
        Length of output array (n_frames).
    ca_signal : ndarray
        Calcium signal (used for 'peak' placement).
    st_ev_inds : list of int
        Start indices for each event.
    end_ev_inds : list of int
        End indices for each event.
    amplitudes : list of float
        Amplitude (dF/F0) for each event.
    placement : {'start', 'peak', 'onset', 'onset_refined'}, optional
        Where to place amplitude:
        - 'start': at event start index (as detected by wavelet)
        - 'peak': at actual calcium peak within event window
        - 'onset': at estimated spike onset (peak - kernel_peak_offset), using peak within event boundaries
        - 'onset_refined': at estimated spike onset with peak refinement search outside event boundaries (legacy, causes lag on noisy data)
    t_rise_frames : float, optional
        Rise time in frames. Required for 'onset' placement.
    t_off_frames : float, optional
        Decay time in frames. Required for 'onset' placement.
    fps : float, optional
        Sampling rate in Hz. Required for 'onset' placement with peak refinement.
    peak_search_window_sec : float, optional
        Time window in seconds to search for refined peak location around
        wavelet-detected peak. Only used for 'onset' placement. Default is 0.2s.

    Returns
    -------
    point_events : ndarray
        Array of zeros with amplitudes at event positions (1D, float).
        Sparse continuous array suitable for TimeSeries(discrete=False).

    Notes
    -----
    If multiple events have centers/peaks at same index, amplitudes are summed.
    This is rare but can occur with closely spaced events.

    The 'onset' placement finds peak within event boundaries, then calculates
    the kernel peak offset to place spikes at estimated onset. This trusts
    wavelet detection and is robust to noise.

    The 'onset_refined' placement (legacy) searches outside event boundaries
    for peak refinement. This can cause temporal lag on noisy data and is
    not recommended for real calcium imaging data.
    """
    check_positive(length=length)
    if placement not in ("start", "peak", "onset", "onset_refined"):
        raise ValueError(
            f"placement must be 'start', 'peak', 'onset', or 'onset_refined', got {placement}"
        )
    if placement in ("onset", "onset_refined"):
        if t_rise_frames is None or t_off_frames is None:
            raise ValueError("Both t_rise_frames and t_off_frames required for 'onset' placement")
        if fps is None:
            raise ValueError("fps required for 'onset' placement")
    ca_signal = np.asarray(ca_signal)
    point_events = np.zeros(length, dtype=float)
    peak_search_frames = int(peak_search_window_sec * fps) if fps is not None else 4
    for start, end, amplitude in zip(st_ev_inds, end_ev_inds, amplitudes):
        if amplitude <= 0:
            continue
        if end <= start or start < 0 or end > length:
            continue
        if placement == "start":
            idx = int(start)
        elif placement == "peak":
            event_segment = ca_signal[start:end]
            peak_offset = np.argmax(event_segment)
            idx = int(start + peak_offset)
        elif placement == "onset":
            # Simple onset placement - uses peak within event boundaries
            kernel_peak_offset = compute_kernel_peak_offset(t_rise_frames, t_off_frames)
            event_segment = ca_signal[start:end]
            peak_offset = np.argmax(event_segment)
            peak_idx = start + peak_offset
            idx = int(peak_idx - kernel_peak_offset)
        elif placement == "onset_refined":
            # Onset with peak refinement - searches outside event boundaries
            kernel_peak_offset = compute_kernel_peak_offset(t_rise_frames, t_off_frames)
            search_start = max(0, start - peak_search_frames)
            search_end = min(length, end + peak_search_frames)
            search_segment = ca_signal[search_start:search_end]
            true_peak_offset = np.argmax(search_segment)
            true_peak_idx = search_start + true_peak_offset
            idx = int(true_peak_idx - kernel_peak_offset)
        if 0 <= idx < length:
            point_events[idx] += amplitude
    return point_events


def _calculate_event_r2(
    calcium_signal, reconstruction, n_mad=4, event_mask=None, wvt_ridges=None, fps=None
):
    """Calculate R² on event regions.

    Parameters
    ----------
    calcium_signal : ndarray
        Original calcium signal
    reconstruction : ndarray
        Reconstructed calcium signal
    n_mad : float, optional
        Number of MADs above median for event threshold. Default: 4.0.
        Ignored if event_mask is provided.
    event_mask : ndarray, optional
        Boolean or binary mask indicating event regions. If provided, uses this mask directly.
    wvt_ridges : list, optional
        List of wavelet ridge objects with start/end attributes. Used to construct
        event mask if event_mask is None.
    fps : float, optional
        Frame rate in Hz. Required if constructing mask from wvt_ridges.

    Returns
    -------
    float
        Event R² value (higher is better). Returns NaN if no events detected.

    Raises
    ------
    ValueError
        If neither event_mask nor wvt_ridges is provided.
    """
    # Construct event mask if not provided
    if event_mask is None:
        if wvt_ridges is not None and len(wvt_ridges) > 0:
            if fps is None:
                raise ValueError("fps required to construct event mask from wvt_ridges")
            # Extract start/end indices from ridges
            st_inds = [int(ridge.start) for ridge in wvt_ridges if ridge.start >= 0]
            end_inds = [int(ridge.end) for ridge in wvt_ridges if ridge.end >= 0]
            if len(st_inds) > 0:
                from .wavelet_event_detection import events_to_ts_array

                events_array = events_to_ts_array(len(calcium_signal), [st_inds], [end_inds], fps)
                event_mask = events_array[0]
            else:
                return np.nan
        else:
            raise ValueError(
                "Either event_mask or wvt_ridges must be provided for event R² calculation. "
                "Run reconstruct_spikes() with create_event_regions=True to populate neuron.events."
            )

    event_mask = event_mask > 0
    if np.sum(event_mask) == 0:
        return np.nan
    ca_events = calcium_signal[event_mask]
    recon_events = reconstruction[event_mask]
    residuals = ca_events - recon_events
    ss_residual = np.sum(residuals**2)
    ss_total = np.sum((ca_events - np.mean(ca_events)) ** 2)
    if ss_total == 0:
        return np.nan
    return 1 - ss_residual / ss_total
