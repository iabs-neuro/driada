import numpy as np
import logging
from scipy.stats import median_abs_deviation
from scipy.optimize import minimize
from scipy.signal import fftconvolve, savgol_filter

from ..information.info_base import TimeSeries
from ..utils.data import check_positive, check_nonnegative
from ..utils.jit import conditional_njit
from .wavelet_event_detection import extract_wvt_events, events_to_ts_array

DEFAULT_T_RISE = 0.25
DEFAULT_T_OFF = 2
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

# Statistical constants
MAD_SCALE_FACTOR = 1.4826  # Scaling factor for MAD → std consistency (normal distribution)
                            # This is 1 / (sqrt(2) * erfcinv(1.5))

# Kernel generation constants
KERNEL_LENGTH_FRAMES = 500  # Minimum kernel length; actual length is max(500, 5 × t_off)


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


class Neuron:
    '''
    Neural calcium and spike data processing.
    
    This class handles calcium imaging time series data and spike trains,
    providing methods for preprocessing, spike-calcium deconvolution,
    and various shuffling techniques for statistical testing.
    
    Parameters
    ----------
    cell_id : int or str
        Unique identifier for the neuron
    ca : array-like
        Calcium imaging time series data. Must be 1D array of finite values.
    sp : array-like, optional
        Spike train data (binary array where 1 indicates spike). If None, 
        spike-related methods will not be available.
    default_t_rise : float, default=0.25
        Rise time constant in seconds for calcium transients. Must be positive.
    default_t_off : float, default=2.0
        Decay time constant in seconds for calcium transients. Must be positive.
    fps : float, default=20.0
        Sampling rate in frames per second. Must be positive.
    fit_individual_t_off : bool, default=False
        Whether to fit decay time for this specific neuron using optimization.
        
    Attributes
    ----------
    cell_id : int or str
        The neuron identifier
    t_rise : float
        Rise time constant in frames (optimized, set by get_kinetics())
    t_off : float
        Decay time constant in frames (optimized, set by get_kinetics())
    fps : float
        Sampling rate in frames per second
    ca_ts : TimeSeries
        Preprocessed calcium time series
    sp_ts : TimeSeries or None
        Spike train time series (if spikes provided)
        
    Notes
    -----
    The class assumes spike data is binary (0 or 1 values). Non-binary 
    spike data may produce incorrect results in spike counting.    '''
    
    def spike_form(t, t_rise, t_off):
        '''Calculate normalized calcium response kernel shape.

        Computes the double-exponential kernel used to model calcium
        indicator dynamics with separate rise and decay time constants.

        Parameters
        ----------
        t : array-like
            Time points (in frames). Must be non-negative.
        t_rise : float
            Rise time constant (in frames). Must be positive.
        t_off : float
            Decay time constant (in frames). Must be positive.

        Returns
        -------
        ndarray
            Normalized kernel values with peak = 1.

        Raises
        ------
        ValueError
            If t_rise or t_off are not positive.

        Notes
        -----
        The kernel has the form: (1 - exp(-t/τ_rise)) * exp(-t/τ_off)
        normalized to have maximum value of 1.        '''
        check_positive(t_rise=t_rise, t_off=t_off)
        return Neuron._spike_form_jit(t, t_rise, t_off)

    spike_form = staticmethod(spike_form)
    
    def _spike_form_jit(t, t_rise, t_off):
        '''JIT-compiled core computation for spike_form.

        Computes normalized double-exponential calcium response kernel.
        This is the performance-critical inner loop separated for JIT compilation.

        Parameters
        ----------
        t : ndarray
            Time points in frames. Assumed to be non-negative.
        t_rise : float
            Rise time constant in frames. Assumed positive.
        t_off : float
            Decay time constant in frames. Assumed positive.

        Returns
        -------
        ndarray
            Normalized kernel with maximum value of 1.

        Raises
        ------
        ValueError
            If computed kernel has zero maximum (numerical issue).

        Notes
        -----
        Input validation is performed in the wrapper function spike_form().
        JIT compilation provides significant speedup for large arrays.        '''
        form = (1 - np.exp(-t / t_rise)) * np.exp(-t / t_off)
        max_val = np.max(form)
        if max_val == 0:
            raise ValueError('Kernel form has zero maximum')
        return form / max_val

    _spike_form_jit = staticmethod(conditional_njit(_spike_form_jit))
    
    def get_restored_calcium(sp, t_rise, t_off):
        '''Reconstruct calcium signal from spike train.

        Convolves spike train with double-exponential kernel to simulate
        calcium indicator dynamics. The output has the same length as the
        input spike train by truncating the convolution tail.

        Parameters
        ----------
        sp : array-like
            Spike train. Can be binary (0/1) or amplitude-weighted (float).
            For best reconstruction fidelity, use neuron.asp.data with
            amplitude information. Must be 1D array.
        t_rise : float
            Rise time constant (in frames). Must be positive.
        t_off : float
            Decay time constant (in frames). Must be positive.

        Returns
        -------
        ndarray
            Reconstructed calcium signal with same length as sp.

        Raises
        ------
        ValueError
            If t_rise or t_off are not positive, or if sp is empty.

        Notes
        -----
        Uses FFT-based convolution for optimal performance. Kernel length is
        adaptive: max(500, 5 × t_off frames) to ensure complete kernel capture
        for all decay time constants.

        The convolution naturally handles amplitude-weighted spikes, where
        each spike value represents event strength in dF/F0 units.        '''
        sp = np.asarray(sp)
        if sp.size == 0:
            raise ValueError('Spike train cannot be empty')
        check_positive(t_rise=t_rise, t_off=t_off)

        # Adaptive kernel length: 5× decay time for complete kernel, minimum 500 frames
        # Safety check: cap at 2000 frames to prevent memory issues from bad t_off
        # (t_off > 400 frames or ~8s @ 20Hz is suspicious for typical indicators)
        kernel_length = max(KERNEL_LENGTH_FRAMES, int(5 * t_off))
        if kernel_length > 2000:
            import warnings
            warnings.warn(
                f'Kernel length {kernel_length} (from t_off={t_off:.1f} frames) capped at 2000. '
                f'This may indicate incorrect t_off measurement. Typical calcium indicators have '
                f't_off < 200 frames (~8-10s @ 20Hz).',
                UserWarning
            )
            kernel_length = 2000

        x = np.arange(kernel_length)
        spform = Neuron.spike_form(x, t_rise, t_off)
        conv = fftconvolve(sp, spform, mode='full')
        return conv[:len(sp)]

    get_restored_calcium = staticmethod(get_restored_calcium)
    
    def ca_mse_error(t_off, ca, spk, t_rise):
        '''Calculate RMSE between observed calcium and reconstructed from spikes.
        
        This function is designed to be used with scipy.optimize.minimize,
        hence the parameter order with t_off first.
        
        Parameters
        ----------
        t_off : float
            Decay time constant (in frames). Must be positive.
        ca : array-like
            Observed calcium signal. Must be 1D.
        spk : array-like
            Spike train. Must be 1D with same length as ca.
        t_rise : float
            Rise time constant (in frames). Must be positive.
            
        Returns
        -------
        float
            Root mean square error between observed and reconstructed calcium.
            
        Raises
        ------
        ValueError
            If arrays have different lengths or time constants are invalid.
            
        Notes
        -----
        Parameter order (t_off first) is optimized for scipy.optimize.minimize
        where t_off is the parameter being optimized.        '''
        ca = np.asarray(ca)
        spk = np.asarray(spk)
        if len(ca) != len(spk):
            raise ValueError(f'''ca and spk must have same length: {len(ca)} vs {len(spk)}''')
        check_positive(t_rise=t_rise, t_off=t_off)
        re_ca = Neuron.get_restored_calcium(spk, t_rise, t_off)
        return np.sqrt(np.sum((ca - re_ca) ** 2) / len(ca))

    ca_mse_error = staticmethod(ca_mse_error)
    
    def calcium_preprocessing(ca, seed=None):
        '''Preprocess calcium signal for spike reconstruction.
        
        Applies preprocessing steps:
        - Converts to float64 for numerical stability
        - Clips negative values to 0 (calcium cannot be negative)
        - Adds tiny noise to prevent numerical singularities
        
        Parameters
        ----------
        ca : array-like
            Raw calcium signal. Must be 1D.
        seed : int, optional
            Random seed for reproducible noise. If None, uses current state.
            
        Returns
        -------
        ndarray
            Preprocessed calcium signal as float64 array.
            
        Raises
        ------
        ValueError
            If ca is empty.
            
        Notes
        -----
        The small noise (1e-8 scale) prevents division by zero and other
        numerical issues in downstream spike reconstruction algorithms.        '''
        ca = np.asarray(ca)
        if ca.size == 0:
            raise ValueError('Calcium signal cannot be empty')
        if seed is not None:
            np.random.seed(seed)
        return Neuron._calcium_preprocessing_jit(ca)

    calcium_preprocessing = staticmethod(calcium_preprocessing)
    
    def _calcium_preprocessing_jit(ca):
        '''JIT-compiled core computation for calcium_preprocessing.
        
        Applies numerical preprocessing to calcium signal for stability.
        This is the performance-critical inner loop separated for JIT compilation.
        
        Parameters
        ----------
        ca : ndarray
            Calcium signal array. Will be converted to float64.
            
        Returns
        -------
        ndarray
            Preprocessed signal with negative values clipped and noise added.
            
        Notes
        -----
        - Negative values are clipped to 0 (physical constraint)
        - Small uniform noise (1e-8 scale) prevents numerical issues
        - Random state should be set externally if reproducibility needed
        - JIT compilation provides significant speedup for large arrays
        
        Side Effects
        ------------
        Uses np.random without explicit state management. Set seed
        externally for reproducibility.        '''
        ca = ca.astype(np.float64)
        ca[ca < 0] = 0
        ca += np.random.random(len(ca)) * 1e-08
        return ca

    _calcium_preprocessing_jit = staticmethod(conditional_njit(_calcium_preprocessing_jit))
    
    def extract_event_amplitudes(ca_signal, st_ev_inds, end_ev_inds, baseline_window=20,
                                 already_dff=False, baseline_offset=0, baseline_offset_sec=None,
                                 use_peak_refinement=False, t_rise_frames=None,
                                 t_off_frames=None, fps=None, peak_search_window_sec=0.2):
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
                raise ValueError('t_rise_frames, t_off_frames, and fps required when use_peak_refinement=True')

        # Convert baseline_offset_sec to frames if provided
        if baseline_offset_sec is not None:
            if fps is None:
                raise ValueError('fps parameter required when using baseline_offset_sec')
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

    extract_event_amplitudes = staticmethod(extract_event_amplitudes)
    
    def deconvolve_given_event_times(ca_signal, event_times, t_rise_frames, t_off_frames, event_mask=None):
        '''Extract amplitudes via non-negative least squares deconvolution.

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
        '''
        from scipy.optimize import nnls
        ca_signal = np.asarray(ca_signal)
        event_times = np.asarray(event_times, dtype=int)
        n_frames = len(ca_signal)
        n_events = len(event_times)
        if n_events == 0:
            return np.array([])

        # Build design matrix
        K = np.zeros((n_frames, n_events))
        for i, event_time_idx in enumerate(event_times):
            if event_time_idx < 0 or event_time_idx >= n_frames:
                continue
            remaining_frames = n_frames - event_time_idx
            t_array = np.arange(remaining_frames)
            kernel = (1 - np.exp(-t_array / t_rise_frames)) * np.exp(-t_array / t_off_frames)
            kernel_max = np.max(kernel)
            if kernel_max > 0:
                kernel = kernel / kernel_max
            K[event_time_idx:, i] = kernel

        # Apply event mask if provided
        if event_mask is not None:
            event_mask = np.asarray(event_mask, dtype=bool)
            if len(event_mask) != n_frames:
                raise ValueError(f"event_mask length ({len(event_mask)}) must match signal length ({n_frames})")
            K_fit = K[event_mask, :]
            ca_fit = ca_signal[event_mask]
        else:
            K_fit = K
            ca_fit = ca_signal

        (amplitudes, residual_norm) = nnls(K_fit, ca_fit)
        return amplitudes

    deconvolve_given_event_times = staticmethod(deconvolve_given_event_times)

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
        peak_offset = t_rise_frames * t_off_frames * np.log(t_off_frames / t_rise_frames) / (t_off_frames - t_rise_frames)
        if np.isnan(peak_offset) or np.isinf(peak_offset):
            return 0.0
        return peak_offset

    compute_kernel_peak_offset = staticmethod(compute_kernel_peak_offset)

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
        kernel_peak_offset = Neuron.compute_kernel_peak_offset(t_rise_frames, t_off_frames)

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

    estimate_onset_times = staticmethod(estimate_onset_times)

    def amplitudes_to_point_events(length, ca_signal, st_ev_inds, end_ev_inds,
                                    amplitudes, placement='peak', t_rise_frames=None,
                                    t_off_frames=None, fps=None,
                                    peak_search_window_sec=0.2):
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
        if placement not in ('start', 'peak', 'onset', 'onset_refined'):
            raise ValueError(f'''placement must be \'start\', \'peak\', \'onset\', or \'onset_refined\', got {placement}''')
        if placement in ('onset', 'onset_refined'):
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
            if placement == 'start':
                idx = int(start)
            elif placement == 'peak':
                event_segment = ca_signal[start:end]
                peak_offset = np.argmax(event_segment)
                idx = int(start + peak_offset)
            elif placement == 'onset':
                # Simple onset placement - uses peak within event boundaries
                kernel_peak_offset = Neuron.compute_kernel_peak_offset(t_rise_frames, t_off_frames)
                event_segment = ca_signal[start:end]
                peak_offset = np.argmax(event_segment)
                peak_idx = start + peak_offset
                idx = int(peak_idx - kernel_peak_offset)
            elif placement == 'onset_refined':
                # Onset with peak refinement - searches outside event boundaries
                kernel_peak_offset = Neuron.compute_kernel_peak_offset(t_rise_frames, t_off_frames)
                search_start = max(0, start - peak_search_frames)
                search_end = min(length, end + peak_search_frames)
                search_segment = ca_signal[search_start:search_end]
                true_peak_offset = np.argmax(search_segment)
                true_peak_idx = search_start + true_peak_offset
                idx = int(true_peak_idx - kernel_peak_offset)
            if 0 <= idx < length:
                point_events[idx] += amplitude
        return point_events

    amplitudes_to_point_events = staticmethod(amplitudes_to_point_events)
    
    def _get_t_rise(self):
        '''Optimized rise time constant (frames).

        Returns None if not yet optimized. Setter validates and clears cached metrics.
        '''
        return self._t_rise

    def _set_t_rise(self, value):
        '''Set rise time constant with validation and cache clearing.

        Parameters
        ----------
        value : float or None
            Rise time in frames (must be positive if not None).
        '''
        if value is not None:
            check_positive(value=value)
        self._t_rise = value
        self._clear_cached_metrics()

    t_rise = property(_get_t_rise, _set_t_rise)
    
    def _get_t_off(self):
        '''Optimized decay time constant (frames).

        Returns None if not yet optimized. Setter validates and clears cached metrics.
        '''
        return self._t_off

    def _set_t_off(self, value):
        '''Set decay time constant with validation and cache clearing.

        Parameters
        ----------
        value : float or None
            Decay time in frames (must be positive and > t_rise if not None).
        '''
        if value is not None:
            check_positive(value=value)
            if self._t_rise is not None and value <= self._t_rise:
                raise ValueError(f'''t_off ({value:.2f}) must be greater than t_rise ({self._t_rise:.2f})''')
        self._t_off = value
        self._clear_cached_metrics()

    t_off = property(_get_t_off, _set_t_off)
    
    def __init__(self, cell_id, ca, sp, default_t_rise=DEFAULT_T_RISE,
                 default_t_off=DEFAULT_T_OFF, fps=DEFAULT_FPS,
                 fit_individual_t_off=False, optimize_kinetics=False,
                 asp=None, wvt_ridges=None, seed=None):
        """Initialize Neuron object with calcium and spike data.

        Parameters
        ----------
        cell_id : str or int
            Unique identifier for this neuron.
        ca : array-like
            Calcium fluorescence signal. Must be 1D.
        sp : array-like or None
            Binary spike train. If provided, must have same length as ca.
        default_t_rise : float, optional
            Default rise time constant in seconds. Default is DEFAULT_T_RISE.
        default_t_off : float, optional
            Default decay time constant in seconds. Default is DEFAULT_T_OFF.
        fps : float, optional
            Sampling rate in Hz. Must be positive. Default is DEFAULT_FPS.
        fit_individual_t_off : bool, optional
            **DEPRECATED**: Use `optimize_kinetics` instead. If True, fit individual
            decay time using old method. Default is False.
        optimize_kinetics : bool or str, optional
            If True or 'direct', optimize kinetics parameters (t_rise, t_off) using
            direct measurement from event shapes. If False, use default parameters.
            Default is False.
            Requires either `asp` parameter or prior call to reconstruct_spikes().
        asp : array-like or None, optional
            Pre-computed amplitude spikes (from prior reconstruction). If provided,
            enables kinetics optimization without re-running reconstruction.
        wvt_ridges : list or None, optional
            Pre-computed wavelet ridges (from prior reconstruction). Used for
            kinetics optimization to provide event boundaries.
        seed : int, optional
            Random seed for preprocessing reproducibility.

        Raises
        ------
        ValueError
            If ca is empty, fps is not positive, or array lengths don't match.
            If optimize_kinetics=True but no spike data available.
        TypeError
            If ca cannot be converted to numeric array.

        Notes
        -----
        The shuffle mask excludes MIN_CA_SHIFT_SEC * fps * t_off frames from each end
        to prevent artifacts in temporal shuffling analyses.

        **New workflow** (recommended):
        >>> neuron = Neuron(cell_id=0, ca=calcium, sp=None, fps=20)
        >>> neuron.reconstruct_spikes(method='wavelet')
        >>> neuron.get_kinetics()  # Optimize after reconstruction

        **Or pass pre-computed data**:
        >>> neuron = Neuron(cell_id=0, ca=calcium, sp=None, fps=20,
        ...                 asp=amp_spikes, wvt_ridges=ridges,
        ...                 optimize_kinetics=True)
        """
        ca = np.asarray(ca)
        if ca.size == 0:
            raise ValueError('Calcium signal cannot be empty')
        check_positive(fps=fps, default_t_rise=default_t_rise, default_t_off=default_t_off)
        self.cell_id = cell_id
        self.ca = TimeSeries(Neuron.calcium_preprocessing(ca, seed=seed), discrete=False)
        from sklearn.preprocessing import MinMaxScaler
        self.ca_scaler = MinMaxScaler()
        self.ca_scaler.fit(self.ca.data.reshape(-1, 1))
        if sp is None:
            self.sp = None
        else:
            sp = np.asarray(sp)
            if len(sp) != len(ca):
                raise ValueError(f'''Spike train length {len(sp)} must match calcium length {len(ca)}''')
            self.sp = TimeSeries(sp.astype(int), discrete=True)
        self.n_frames = len(self.ca.data)
        self.fps = fps
        self.sp_count = np.sum(self.sp.data) if self.sp is not None else 0
        self.events = None
        if asp is not None:
            asp = np.asarray(asp)
            if len(asp) != len(ca):
                raise ValueError(f'''asp length {len(asp)} must match calcium length {len(ca)}''')
            self.asp = TimeSeries(asp, discrete=False)
        else:
            self.asp = None
        self.wvt_ridges = wvt_ridges
        self._t_rise = None
        self._t_off = None
        self.noise_ampl = None
        self.mad = None
        self.snr = None
        self.wavelet_snr = None
        self._kinetics_info = None
        self.reconstruction_r2 = None
        self.snr_reconstruction = None
        self.mae = None
        self.event_count = None
        self._reconstructed = None
        self._reconstructed_scaled = None
        self._has_reconstructed = False  # Track if reconstruction was done before
        if fps is None:
            fps = DEFAULT_FPS
        if default_t_rise is None:
            default_t_rise = DEFAULT_T_RISE
        if default_t_off is None:
            default_t_off = DEFAULT_T_OFF
        self.default_t_off = default_t_off * fps
        self.default_t_rise = default_t_rise * fps
        if fit_individual_t_off and optimize_kinetics:
            raise ValueError('Cannot specify both fit_individual_t_off and optimize_kinetics')
        if fit_individual_t_off:
            import warnings
            warnings.warn(
                'fit_individual_t_off is deprecated and will be removed in v1.0. '
                'Use optimize_kinetics=True instead.',
                DeprecationWarning,
                stacklevel=2
            )
            if self.sp is not None or self.asp is not None:
                (self.t_off, self.noise_ampl) = self._fit_t_off()
                t_off = self.t_off
            else:
                t_off = self.default_t_off
        elif optimize_kinetics:
            method = 'direct' if optimize_kinetics is True else optimize_kinetics
            self.get_kinetics(method, fps)
            t_off = self.t_off if self.t_off is not None else self.default_t_off
        else:
            t_off = self.default_t_off
        self.ca.shuffle_mask = np.ones(self.n_frames, dtype=bool)
        # FPS-adaptive minimum shift: MIN_CA_SHIFT_SEC seconds worth of frames
        min_shift_frames = int(MIN_CA_SHIFT_SEC * self.fps)
        min_shift = int(t_off * min_shift_frames)
        self.ca.shuffle_mask[:min_shift] = False
        self.ca.shuffle_mask[self.n_frames - min_shift:] = False

    
    def reconstruct_spikes(self, method="wavelet", iterative=True, n_iter=3,
                          min_events_threshold=2, adaptive_thresholds=False,
                          amplitude_method="deconvolution", show_progress=False, create_event_regions=False,
                          event_mask_expansion_sec=5.0, **kwargs):
        """Reconstruct spikes from calcium signal.

        Reconstructs discrete spike events from continuous calcium
        fluorescence traces using wavelet or threshold-based methods.

        Parameters
        ----------
        method : str, optional
            Reconstruction method: 'wavelet' or 'threshold'.
            Default is 'wavelet'.
        iterative : bool, optional
            Use iterative wavelet detection with residual analysis (wavelet method only).
            Detects events in residuals across multiple iterations to handle overlapping
            events and improve detection of smaller events. Default is True.
        n_iter : int, optional
            Number of iterations (only if iterative=True). Default is 3.
        min_events_threshold : int, optional
            Stop iterating if fewer events detected (only if iterative=True). Default is 2.
        adaptive_thresholds : bool, optional
            Progressively relax detection thresholds across iterations (only if iterative=True).
            Default is False.
        amplitude_method : str, optional
            Method for extracting event amplitudes: 'peak' or 'deconvolution'.
            - 'peak': Peak-based extraction with baseline subtraction (backward compatible)
            - 'deconvolution': Non-negative least squares deconvolution (default, optimal for overlapping events)
            Default is 'deconvolution'.
        show_progress : bool, optional
            Whether to show progress bar during wavelet detection. Default is False
            (no progress bar for single neuron processing).
        create_event_regions : bool, optional
            If True, creates self.events (binary rectangular regions marking event durations).
            This is legacy behavior - modern code should use self.asp (amplitude spikes) instead.
            Only needed for backward compatibility or specific visualization needs.
            Default is False to avoid unnecessary computation and warnings.
        event_mask_expansion_sec : float, optional
            Time in seconds to expand event mask around detected events for NNLS deconvolution.
            The mask is expanded by ±event_mask_expansion_sec to cover the full calcium transient
            (rise + decay). Larger values include more of the decay but also more baseline noise.
            Default is 5.0 seconds (optimal balance for GCaMP6s with t_off ~2s).
        **kwargs
            Additional parameters depend on method:

            For 'wavelet':
            * fps : float, optional
                Sampling rate in Hz. Default is DEFAULT_FPS (20.0 Hz).
            * min_event_dur : float, optional
                Minimum event duration in seconds. Default is 0.5.
            * max_event_dur : float, optional
                Maximum event duration in seconds. Default is 2.5.

            For 'threshold':
            * threshold_std : float, optional
                Number of standard deviations above mean. Default is 2.5.
            * smooth_sigma : float, optional
                Gaussian smoothing sigma in frames. Default is 2.
            * min_spike_interval : float, optional
                Minimum interval between spikes in seconds. Default is 0.1.

        Returns
        -------
        ndarray or None
            If create_event_regions=True: Binary event regions with shape (n_frames,). Values are 0 or 1.
            If create_event_regions=False: None (no binary regions created).
            Always populates self.asp (amplitude spikes) and self.sp (binary spikes) attributes.
            Optionally populates self.events (binary regions) if create_event_regions=True.

        Raises
        ------
        NotImplementedError
            If method is not 'wavelet' or 'threshold'.
        ValueError
            If parameters are invalid for the chosen method.

        Notes
        -----
        The wavelet method uses continuous wavelet transform to detect
        calcium transient events. The threshold method uses derivative-based
        spike detection with Gaussian smoothing.

        Iterative detection (default) performs multiple detection passes on residuals,
        removing detected events and searching for additional events in the remaining
        signal. This approach significantly improves detection of overlapping and
        smaller events compared to single-pass detection.

        For single-pass detection (backward compatible), set iterative=False.        """
        # Warn if re-running reconstruction without optimized kinetics
        # (skip warning on first reconstruction - user needs events before optimizing)
        if self._has_reconstructed and (self.t_rise is None or self.t_off is None):
            import warnings
            fps_for_warning = kwargs.get('fps', self.fps if self.fps is not None else DEFAULT_FPS)
            warnings.warn(
                f"Neuron {self.cell_id}: Re-running reconstruction with default kinetics. "
                f"Consider optimizing first: neuron.optimize_kinetics(method='direct', fps={fps_for_warning})",
                UserWarning
            )
        self._has_reconstructed = True

        if method == 'wavelet':
            fps = kwargs.get('fps', self.fps)
            min_event_dur = kwargs.get('min_event_dur', 0.5)
            max_event_dur = kwargs.get('max_event_dur', 2.5)
            check_positive(fps=fps, min_event_dur=min_event_dur, max_event_dur=max_event_dur)
            if min_event_dur >= max_event_dur:
                raise ValueError(f'''min_event_dur ({min_event_dur}) must be less than max_event_dur ({max_event_dur})''')
            wvt_kwargs = {
                'fps': fps,
                'min_event_dur': min_event_dur,
                'max_event_dur': max_event_dur,
                'scale_length_thr': kwargs.get('scale_length_thr', 15),
                'max_scale_thr': kwargs.get('max_scale_thr', 6),
                'max_ampl_thr': kwargs.get('max_ampl_thr', 0.04),
                'sigma': kwargs.get('sigma', 8) }
            ca_data = self.ca.scdata.reshape(1, -1)
            if iterative:
                # Wavelet iterative: detect events in residuals across multiple iterations
                check_positive(n_iter=n_iter, min_events_threshold=min_events_threshold)
                all_st_inds_list = []
                all_end_inds_list = []
                all_ridges_list = []
                current_signal = self.ca.scdata.copy()

                # Prepare iteration-specific kwargs (adaptive thresholds if enabled)
                if adaptive_thresholds:
                    base_min_dur = min_event_dur
                    base_max_dur = max_event_dur
                    iter_kwargs = []
                    for i in range(n_iter):
                        relax_factor = 1 - i * 0.2
                        iter_kw = wvt_kwargs.copy()  # Start with all wavelet params
                        iter_kw['min_event_dur'] = max(base_min_dur * relax_factor, 0.1)
                        iter_kw['max_event_dur'] = base_max_dur
                        iter_kwargs.append(iter_kw)
                else:
                    iter_kwargs = [wvt_kwargs.copy() for _ in range(n_iter)]

                t_rise = self.t_rise if self.t_rise is not None else self.default_t_rise
                t_off = self.t_off if self.t_off is not None else self.default_t_off

                # Iterative detection loop
                for iter_idx in range(n_iter):
                    current_signal_2d = current_signal.reshape(1, -1)
                    (st_ev_inds, end_ev_inds, filtered_ridges) = extract_wvt_events(
                        current_signal_2d, iter_kwargs[iter_idx], show_progress=show_progress
                    )
                    st_inds = st_ev_inds[0] if len(st_ev_inds) > 0 else []
                    end_inds = end_ev_inds[0] if len(end_ev_inds) > 0 else []
                    ridges = filtered_ridges[0] if len(filtered_ridges) > 0 else []

                    if len(st_inds) >= min_events_threshold:
                        all_st_inds_list.extend(st_inds)
                        all_end_inds_list.extend(end_inds)
                        all_ridges_list.extend(ridges)
                        # Compute residual for next iteration
                        current_signal = self._compute_residual(
                            st_inds, end_inds, current_signal, t_rise, t_off, fps
                        )

                # After ALL iterations: finalize with ALL collected events
                self.wvt_ridges = all_ridges_list
                return self._finalize_detection(
                    all_st_inds_list, all_end_inds_list, t_rise, t_off, fps,
                    amplitude_method, event_mask_expansion_sec, create_event_regions
                )
            # Wavelet non-iterative: single-pass detection
            (st_ev_inds, end_ev_inds, filtered_ridges) = extract_wvt_events(ca_data, wvt_kwargs, show_progress=show_progress)
            self.wvt_ridges = filtered_ridges[0] if len(filtered_ridges) > 0 else []
            st_inds = st_ev_inds[0] if len(st_ev_inds) > 0 else []
            end_inds = end_ev_inds[0] if len(end_ev_inds) > 0 else []
            t_rise = self.t_rise if self.t_rise is not None else self.default_t_rise
            t_off = self.t_off if self.t_off is not None else self.default_t_off

            return self._finalize_detection(
                st_inds, end_inds, t_rise, t_off, fps,
                amplitude_method, event_mask_expansion_sec, create_event_regions
            )
        if method == 'threshold':
            # Threshold-based event detection with full deconvolution pipeline
            # Prepare parameters
            fps = kwargs.get('fps', self.fps if self.fps is not None else DEFAULT_FPS)
            threshold = kwargs.get('threshold', None)
            n_mad = kwargs.get('n_mad', 4.0)
            min_duration_frames = kwargs.get('min_duration_frames', 3)
            merge_gap_frames = kwargs.get('merge_gap_frames', 2)
            use_scaled = kwargs.get('use_scaled', True)
            event_mask_expansion_sec = kwargs.get('event_mask_expansion_sec', 2.0)

            # Get kinetics (use optimized if available, otherwise defaults)
            t_rise = self.t_rise if self.t_rise is not None else self.default_t_rise
            t_off = self.t_off if self.t_off is not None else self.default_t_off

            if iterative:
                # Threshold iterative: detect events in residuals across multiple iterations
                check_positive(n_iter=n_iter, min_events_threshold=min_events_threshold)
                all_st_inds_list = []
                all_end_inds_list = []
                all_events_list = []
                original_signal = self.ca.scdata.copy() if use_scaled else self.ca.data.copy()
                current_signal = original_signal.copy()

                # Compute statistics ONCE from original signal (not residuals!)
                # This prevents finding spurious events in pure noise, where adaptive
                # threshold would keep finding tail exceedances in each iteration.
                if threshold is None:
                    signal_median = np.median(original_signal)
                    signal_mad = median_abs_deviation(original_signal, scale='normal')

                    # Pre-compute thresholds for each iteration based on ORIGINAL statistics
                    if adaptive_thresholds:
                        # Progressively relax threshold: 4.0 -> 3.2 -> 2.4 -> ... MADs
                        iter_n_mads = [max(n_mad * (1 - i * 0.2), 1.0) for i in range(n_iter)]
                    else:
                        iter_n_mads = [n_mad] * n_iter
                    iter_thresholds = [signal_median + nm * signal_mad for nm in iter_n_mads]
                else:
                    # User provided explicit threshold - use it for all iterations
                    iter_thresholds = [threshold] * n_iter

                # Iterative detection loop
                for iter_idx in range(n_iter):
                    detected_events = self.detect_events_threshold(
                        threshold=iter_thresholds[iter_idx],
                        min_duration_frames=min_duration_frames,
                        merge_gap_frames=merge_gap_frames,
                        use_scaled=False,  # We pass custom signal
                        signal=current_signal
                    )
                    st_inds = [int(e.start) for e in detected_events]
                    end_inds = [int(e.end) for e in detected_events]

                    if len(st_inds) >= min_events_threshold:
                        all_st_inds_list.extend(st_inds)
                        all_end_inds_list.extend(end_inds)
                        all_events_list.extend(detected_events)
                        # Compute residual for next iteration
                        current_signal = self._compute_residual(
                            st_inds, end_inds, current_signal, t_rise, t_off, fps
                        )

                # After ALL iterations: finalize with ALL collected events
                self.threshold_events = all_events_list
                return self._finalize_detection(
                    all_st_inds_list, all_end_inds_list, t_rise, t_off, fps,
                    amplitude_method, event_mask_expansion_sec, create_event_regions
                )

            # Threshold non-iterative: single-pass detection
            detected_events = self.detect_events_threshold(
                threshold=threshold,
                n_mad=n_mad,
                min_duration_frames=min_duration_frames,
                merge_gap_frames=merge_gap_frames,
                use_scaled=use_scaled
            )
            self.threshold_events = detected_events
            st_inds = [int(e.start) for e in detected_events]
            end_inds = [int(e.end) for e in detected_events]

            return self._finalize_detection(
                st_inds, end_inds, t_rise, t_off, fps,
                amplitude_method, event_mask_expansion_sec, create_event_regions
            )
        raise NotImplementedError(f'''Method \'{method}\' not implemented. Available methods: \'wavelet\', \'threshold\'''')

    
    def _clear_cached_metrics(self):
        '''Clear all cached quality metrics and reconstructions.

        Should be called whenever ASP data or kinetics parameters change,
        as these changes invalidate all cached metrics that depend on
        reconstruction quality.

        Cached metrics (cleared):
        - self._reconstructed: Cached reconstructed calcium signal
        - self._reconstructed_scaled: Cached scaled reconstruction
        - self.reconstruction_r2: Cached default R² metric (not event-only)
        - self.snr_reconstruction: Cached reconstruction SNR
        - self.mae: Cached Mean Absolute Error
        - self.event_count: Cached number of detected events
        - self.noise_ampl: Cached RMSE (noise amplitude)
        - self.mad: Cached Median Absolute Deviation
        - self.snr: Cached Signal-to-Noise Ratio

        Not cached (computed on-demand, parameter-dependent):
        - get_nmae(n_mad): Depends on n_mad parameter
        - get_nrmse(n_mad): Depends on n_mad parameter
        - get_event_snr(n_mad): Depends on n_mad parameter
        - get_baseline_noise_std(n_mad): Depends on n_mad parameter
        - get_reconstruction_r2(event_only=True, n_mad): With event_only flag

        Notes
        -----
        Kinetics info (_kinetics_info) is NOT cleared as it remains valid
        until a new optimization is performed.
        '''
        self._reconstructed = None
        self._reconstructed_scaled = None
        self.reconstruction_r2 = None
        self.snr_reconstruction = None
        self.mae = None
        self.event_count = None
        self.noise_ampl = None
        self.mad = None
        self.snr = None

    
    def _compute_scaled_reconstruction(self):
        '''Compute and cache scaled reconstruction after ASP is updated.

        Called automatically after reconstruct_spikes() completes to ensure
        scaled reconstruction is available for information theory analyses.
        Uses current kinetics (optimized or default).
        '''
        if self.asp is None:
            return None
        t_rise = self.t_rise if self.t_rise is not None else self.default_t_rise
        t_off = self.t_off if self.t_off is not None else self.default_t_off
        ca_recon = Neuron.get_restored_calcium(self.asp.data, t_rise, t_off)
        self._reconstructed = TimeSeries(ca_recon, discrete=False)
        self._reconstructed_scaled = self.ca_scaler.transform(ca_recon.reshape(-1, 1)).reshape(-1)

    def _extract_amplitudes(self, st_inds, end_inds, t_rise, t_off, fps,
                            amplitude_method, event_mask_expansion_sec):
        """Extract event amplitudes using NNLS deconvolution or peak method.

        Parameters
        ----------
        st_inds : list of int
            Event start frame indices.
        end_inds : list of int
            Event end frame indices.
        t_rise : float
            Rise time in frames.
        t_off : float
            Decay time in frames.
        fps : float
            Sampling rate in Hz.
        amplitude_method : str
            'deconvolution' for NNLS or 'peak' for peak-based extraction.
        event_mask_expansion_sec : float
            Time in seconds to expand event mask around detected events.

        Returns
        -------
        list
            Extracted amplitudes for each event.
        """
        if len(st_inds) == 0:
            return []

        if amplitude_method == 'deconvolution':
            # Estimate true onset times from detection boundaries
            # Detectors find rising/peak regions, not true spike onsets
            onset_times = Neuron.estimate_onset_times(
                self.ca.data, st_inds, end_inds, t_rise, t_off
            )

            # Create expanded event mask for NNLS
            # Expand from onset through decay tail (5 time constants = 99.3% of decay)
            event_mask = np.zeros(self.n_frames, dtype=bool)
            mask_expansion_frames = int(event_mask_expansion_sec * fps)
            decay_tail_frames = int(5 * t_off)
            for onset, end in zip(onset_times, end_inds):
                expanded_start = max(0, onset - mask_expansion_frames)
                expanded_end = min(self.n_frames, max(int(end), onset + decay_tail_frames) + mask_expansion_frames)
                event_mask[expanded_start:expanded_end] = True

            return Neuron.deconvolve_given_event_times(
                self.ca.data, onset_times, t_rise, t_off, event_mask=event_mask
            )
        elif amplitude_method == 'peak':
            baseline_window = int(BASELINE_WINDOW_SEC * fps)
            return Neuron.extract_event_amplitudes(
                self.ca.data, st_inds, end_inds,
                baseline_window=baseline_window,
                already_dff=True
            )
        else:
            raise ValueError(f"Unknown amplitude_method: {amplitude_method}")

    def _create_asp_sp(self, st_inds, end_inds, amplitudes, t_rise, t_off, fps):
        """Create ASP (amplitude spikes) and SP (binary spikes) TimeSeries.

        Parameters
        ----------
        st_inds : list of int
            Event start frame indices.
        end_inds : list of int
            Event end frame indices.
        amplitudes : list
            Extracted amplitudes for each event.
        t_rise : float
            Rise time in frames.
        t_off : float
            Decay time in frames.
        fps : float
            Sampling rate in Hz.

        Returns
        -------
        ndarray
            ASP array (amplitude spikes).
        """
        if len(st_inds) == 0 or len(amplitudes) == 0:
            self.asp = TimeSeries(np.zeros(self.n_frames), discrete=False)
            self.sp = TimeSeries(np.zeros(self.n_frames, dtype=int), discrete=True)
            self.sp_count = 0
            return np.zeros(self.n_frames)

        asp = Neuron.amplitudes_to_point_events(
            self.n_frames, self.ca.data, st_inds, end_inds,
            amplitudes, placement='onset', t_rise_frames=t_rise, t_off_frames=t_off,
            fps=fps
        )
        sp = (asp > 0).astype(int)
        self.asp = TimeSeries(asp, discrete=False)
        self.sp = TimeSeries(sp, discrete=True)
        self.sp_count = int(np.sum(sp))
        return asp

    def _create_event_regions(self, st_inds, end_inds, fps, create_event_regions):
        """Create binary event regions TimeSeries.

        Parameters
        ----------
        st_inds : list of int
            Event start frame indices.
        end_inds : list of int
            Event end frame indices.
        fps : float
            Sampling rate in Hz.
        create_event_regions : bool
            Whether to create event regions.

        Returns
        -------
        ndarray or None
            Event regions array if create_event_regions=True, else None.
        """
        if not create_event_regions:
            self.events = None
            return None

        if len(st_inds) > 0:
            st_ev_inds_2d = [list(st_inds)]
            end_ev_inds_2d = [list(end_inds)]
            events = events_to_ts_array(self.n_frames, st_ev_inds_2d, end_ev_inds_2d, fps)
            self.events = TimeSeries(events.flatten(), discrete=True)
            return events.flatten()
        else:
            events = np.zeros(self.n_frames, dtype=int)
            self.events = TimeSeries(events, discrete=True)
            return events

    def _finalize_reconstruction(self, fps):
        """Finalize reconstruction: clear cache and compute scaled reconstruction.

        Parameters
        ----------
        fps : float
            Sampling rate in Hz (unused but kept for API consistency).
        """
        self._clear_cached_metrics()
        if self.asp is not None:
            self._compute_scaled_reconstruction()

    def _finalize_detection(self, st_inds, end_inds, t_rise, t_off, fps,
                            amplitude_method, event_mask_expansion_sec, create_event_regions):
        """Common finalization for all detection branches.

        Extracts amplitudes, creates ASP/SP, event regions, and finalizes.

        Parameters
        ----------
        st_inds : list of int
            Event start frame indices.
        end_inds : list of int
            Event end frame indices.
        t_rise : float
            Rise time in frames.
        t_off : float
            Decay time in frames.
        fps : float
            Sampling rate in Hz.
        amplitude_method : str
            'deconvolution' or 'peak'.
        event_mask_expansion_sec : float
            Time in seconds to expand event mask.
        create_event_regions : bool
            Whether to create binary event regions.

        Returns
        -------
        ndarray or None
            Event regions if create_event_regions=True, else None.
        """
        amplitudes = self._extract_amplitudes(
            st_inds, end_inds, t_rise, t_off, fps,
            amplitude_method, event_mask_expansion_sec
        )
        self._create_asp_sp(st_inds, end_inds, amplitudes, t_rise, t_off, fps)
        events = self._create_event_regions(st_inds, end_inds, fps, create_event_regions)
        self._finalize_reconstruction(fps)
        return events

    def _compute_residual(self, st_inds, end_inds, signal, t_rise, t_off, fps):
        """Compute residual signal after subtracting quick reconstruction.

        Used in iterative detection to find events in the residual.

        Parameters
        ----------
        st_inds : list of int
            Event start frame indices.
        end_inds : list of int
            Event end frame indices.
        signal : ndarray
            Current signal to subtract from (should be scaled signal).
        t_rise : float
            Rise time in frames.
        t_off : float
            Decay time in frames.
        fps : float
            Sampling rate in Hz.

        Returns
        -------
        ndarray
            Residual signal after subtracting reconstruction.
        """
        if len(st_inds) == 0:
            return signal.copy()

        # Extract amplitudes from the CURRENT signal (not original!)
        # This is critical for iterative mode where signal is the residual
        # Use already_dff=True since signal is already scaled/normalized
        baseline_window = int(BASELINE_WINDOW_SEC * fps)
        quick_amps = Neuron.extract_event_amplitudes(
            signal, st_inds, end_inds,
            baseline_window=baseline_window,
            already_dff=True
        )

        # Create quick ASP
        quick_asp = np.zeros(self.n_frames)
        for st_idx, amp in zip(st_inds, quick_amps):
            if 0 <= st_idx < self.n_frames:
                quick_asp[st_idx] = amp

        # Reconstruct and subtract (no scaling needed - signal is already scaled)
        reconstruction = Neuron.get_restored_calcium(quick_asp, t_rise, t_off)
        return signal - reconstruction


    def get_mad(self):
        '''Get median absolute deviation of calcium signal.
        
        Computes MAD as a robust measure of noise level in the calcium signal.
        Caches the result for efficiency.
        
        Returns
        -------
        float
            Median absolute deviation of the calcium signal, scaled to be
            consistent with standard deviation for normally distributed data.
            
        Notes
        -----
        MAD is more robust to outliers than standard deviation, making it
        ideal for noise estimation in calcium imaging data which often
        contains spike-related transients.        '''
        if self.mad is None:
            # Try co-computation with SNR if spikes/events available
            if (self.asp is not None and np.sum(np.abs(self.asp.data)) > 0) or \
               (self.sp is not None and np.sum(self.sp.data) > 0):
                try:
                    self._calc_snr_simple()  # Computes both SNR and MAD
                except ValueError:
                    # Spikes exist but SNR calc failed - compute MAD independently
                    self.mad = median_abs_deviation(self.ca.data)
            else:
                # No spikes - compute MAD independently
                self.mad = median_abs_deviation(self.ca.data)

        return self.mad

    def get_snr(self, method='simple'):
        '''Get signal-to-noise ratio of calcium signal.

        Unified interface for SNR calculation supporting multiple methods.

        Parameters
        ----------
        method : {'simple', 'wavelet'}, optional
            - 'simple': Peak-based SNR using spike locations (fast, default)
            - 'wavelet': Event-based SNR using wavelet regions (accurate)
            Default is 'simple'.

        Returns
        -------
        float
            Signal-to-noise ratio (dimensionless). Higher values indicate
            stronger signal relative to noise.

        Raises
        ------
        ValueError
            If no spike/event data available, or if method is invalid.

        Notes
        -----
        **Simple method:**
        - SNR = mean(calcium at spike peaks) / MAD(entire signal)
        - Fast computation, uses asp (amplitude spikes) or sp (binary spikes)
        - Caches result in self.snr

        **Wavelet method:**
        - SNR = median(event amplitudes) / std(baseline)
        - More accurate, empirically validated against ground truth
        - Requires prior wavelet reconstruction
        - Caches result in self.wavelet_snr

        Examples
        --------
        >>> neuron.get_snr()                    # Simple SNR (default)
        >>> neuron.get_snr(method='simple')     # Explicit simple
        >>> neuron.get_snr(method='wavelet')    # Wavelet-based SNR
        '''
        if method == 'wavelet':
            return self.get_wavelet_snr()
        elif method == 'simple':
            if self.snr is None:
                self._calc_snr_simple()
            return self.snr
        else:
            raise ValueError(f"Invalid method '{method}'. Must be 'simple' or 'wavelet'")

    
    def _calc_snr_simple(self):
        '''Calculate simple peak-based SNR and MAD.

        Internal method that computes both SNR and MAD in a single pass
        for efficiency. Preferentially uses amplitude spikes (asp) over
        binary spikes (sp) for more accurate signal estimation.

        Returns
        -------
        tuple
            (snr, mad) where snr is signal-to-noise ratio and mad is
            median absolute deviation.

        Raises
        ------
        ValueError
            If no spikes are present, if MAD is zero, or if SNR
            calculation results in NaN.        '''
        # Prefer asp (amplitude spikes) over sp (binary spikes)
        if self.asp is not None and np.sum(np.abs(self.asp.data)) > 0:
            spk_inds = np.nonzero(self.asp.data)[0]
        elif self.sp is not None and np.sum(self.sp.data) > 0:
            spk_inds = np.nonzero(self.sp.data)[0]
        else:
            raise ValueError('No spike data available')

        if len(spk_inds) == 0:
            raise ValueError('No spikes found!')

        mad = median_abs_deviation(self.ca.data)
        if mad == 0:
            raise ValueError('MAD is zero, cannot compute SNR')

        sn = np.mean(self.ca.data[spk_inds]) / mad
        if np.isnan(sn):
            raise ValueError('Error in SNR calculation')

        # Cache both values
        self.snr = sn
        self.mad = mad

        return (sn, mad)

    
    def get_reconstruction_r2(self, event_only=False, n_mad=3.0, use_detected_events=True):
        '''Get R² for calcium reconstruction quality.

        Computes the coefficient of determination (R²) measuring how well
        the double-exponential model fits the observed calcium signal.
        Higher values indicate better model fit. Standard R² is cached.

        Parameters
        ----------
        event_only : bool, default=False
            If True, compute R² only in detected event regions.
            If False, compute standard R² over entire signal (cached).
        n_mad : float, default=3.0
            Number of MAD (Median Absolute Deviation) units above median for event detection.
            Only used when event_only=True and use_detected_events=False.
        use_detected_events : bool, default=True
            If True, use self.events (wavelet-detected event regions) for event mask.
            If False, use MAD-based threshold on original signal (legacy behavior).
            Only used when event_only=True.

        Returns
        -------
        float
            R² value between -inf and 1. Values closer to 1 indicate better fit.
            R² > 0.9: Excellent fit (high-quality calcium transients)
            R² 0.7-0.9: Good fit (acceptable quality)
            R² 0.5-0.7: Moderate fit (check for artifacts)
            R² < 0.5: Poor fit (likely artifacts or model mismatch)

        Raises
        ------
        ValueError
            If amplitude spike data is not available, or if use_detected_events=True
            but self.events is None.

        Notes
        -----
        Standard R² = 1 - (SS_residual / SS_total) computed over entire signal.
        Event R² = 1 - (SS_residual_events / SS_total_events) in event regions only.

        When use_detected_events=True (recommended), event regions are defined by
        self.events (wavelet ridge detection), ensuring alignment with reconstruction.

        When use_detected_events=False (legacy), event regions are defined by
        MAD threshold on the original signal.

        Uses cached RMSE from get_noise_ampl() for efficiency in standard mode.
        Event-only mode is NOT cached as it depends on parameters.
        '''
        if self.asp is None:
            raise ValueError('Amplitude spikes required for reconstruction R². Call reconstruct_spikes() first.')
        if event_only:
            t_rise = self.t_rise if self.t_rise is not None else self.default_t_rise
            t_off = self.t_off if self.t_off is not None else self.default_t_off
            ca_reconstructed = Neuron.get_restored_calcium(self.asp.data, t_rise, t_off)

            # Determine event mask source (supports both wavelet and threshold)
            event_mask = None
            event_ridges = None
            if use_detected_events:
                if self.events is not None:
                    event_mask = self.events.data
                elif hasattr(self, 'threshold_events') and self.threshold_events:
                    event_ridges = self.threshold_events
                elif self.wvt_ridges:
                    event_ridges = self.wvt_ridges

            event_r2 = Neuron._calculate_event_r2(
                self.ca.data, ca_reconstructed, n_mad,
                event_mask=event_mask,
                wvt_ridges=event_ridges,
                fps=self.fps
            )
            if np.isnan(event_r2):
                raise ValueError('Event R² calculation failed. No events detected or insufficient event data. Consider lowering n_mad parameter.')
            return event_r2
        if self.reconstruction_r2 is None:
            rmse = self.get_noise_ampl()
            ss_residual = rmse ** 2 * self.n_frames
            ss_total = np.sum((self.ca.data - np.mean(self.ca.data)) ** 2)
            if ss_total == 0:
                raise ValueError('Total variance is zero, cannot compute R²')
            self.reconstruction_r2 = 1 - ss_residual / ss_total
        return self.reconstruction_r2

    
    def get_snr_reconstruction(self):
        '''Get reconstruction-based SNR.

        Computes SNR as the ratio of calcium signal standard deviation
        to reconstruction error (RMSE). Provides a model-based quality
        metric complementary to the peak-based SNR. Caches the result.

        Returns
        -------
        float
            Reconstruction SNR value (positive). Higher is better.
            SNR_recon > 20: Excellent reconstruction fidelity
            SNR_recon 10-20: Good reconstruction fidelity
            SNR_recon 5-10: Fair reconstruction fidelity
            SNR_recon < 5: Poor reconstruction (check signal quality)

        Raises
        ------
        ValueError
            If spike data is not available for reconstruction.

        Notes
        -----
        SNR_reconstruction = std(Ca) / RMSE

        This metric differs from get_snr() which uses peak amplitudes
        at spike times. Reconstruction SNR measures overall model fit quality.

        Uses cached RMSE from get_noise_ampl() for efficiency.
        '''
        if self.snr_reconstruction is None:
            rmse = self.get_noise_ampl()
            signal_std = np.std(self.ca.data)
            if rmse == 0:
                raise ValueError('RMSE is zero, cannot compute reconstruction SNR')
            self.snr_reconstruction = signal_std / rmse
        return self.snr_reconstruction

    
    def get_reconstruction_scaled(self, t_rise=None, t_off=None):
        """Get reconstruction transformed using ca.scdata's scaler.

        Applies the exact same MinMaxScaler transformation to the reconstruction
        that was used to create ca.scdata. This ensures consistency for information
        theory and dimensionality reduction analyses that use ca.scdata.

        Caches result in self._reconstructed_scaled for efficiency. Cache is cleared
        when ASP or kinetics change (via _clear_cached_metrics).

        Parameters
        ----------
        t_rise : float, optional
            Rise time in seconds. If None, uses optimized or default value.
        t_off : float, optional
            Decay time in seconds. If None, uses optimized or default value.

        Returns
        -------
        ndarray
            Reconstruction scaled using ca_scaler.transform(). Shape matches ca.data.
            **Note:** Values may fall outside [0,1] if reconstruction differs from
            training data (ca.data) in amplitude or baseline.

        Raises
        ------
        ValueError
            If spike data (asp) is not available.

        Notes
        -----
        **Scaling Behavior:**

        Uses `scaler.transform()` NOT `scaler.fit_transform()`:
        - Fitted on ca.data during __init__
        - Applied to reconstruction (which may exceed ca.data range)
        - Result can fall outside [0,1] - this is EXPECTED and CORRECT

        **When values exceed [0,1]:**
        - recon > data_max → scaled > 1.0 (e.g., missed event in original)
        - recon < data_min → scaled < 0.0 (e.g., baseline noise, wrong kinetics)

        This is the mathematically correct behavior for scaler.transform() and
        preserves the relative scaling needed for information theory.

        **Use Cases:**
        - Information theory analyses (mutual information with ca.scdata)
        - Dimensionality reduction (PCA/ICA on ca.scdata-like signals)
        - Comparing reconstruction vs ca.scdata in scaled space

        **DO NOT use for metrics:** R², MAE, RMSE should use ca.data (original scale).

        Examples
        --------
        >>> neuron.reconstruct_spikes()
        >>> recon_scaled = neuron.get_reconstruction_scaled()
        >>>
        >>> # Check range violations
        >>> pct_below = 100 * np.sum(recon_scaled < 0) / len(recon_scaled)
        >>> pct_above = 100 * np.sum(recon_scaled > 1.0) / len(recon_scaled)
        >>> print(f'Below 0: {pct_below:.1f}%, Above 1: {pct_above:.1f}%')
        >>>
        >>> # Use with dimensionality reduction
        >>> from sklearn.decomposition import PCA
        >>> pca = PCA(n_components=5)
        >>> pca.fit(ca.scdata_matrix)  # Fit on scaled data
        >>> recon_proj = pca.transform(recon_scaled.reshape(1, -1))

        See Also
        --------
        get_reconstruction_r2 : R² using ca.data (for metrics)
        get_mae : MAE using ca.data (for metrics)
        """
        if self.asp is None:
            raise ValueError('No spike data available. Call reconstruct_spikes() first.')
        if self._reconstructed_scaled is not None and t_rise is None and t_off is None:
            return self._reconstructed_scaled
        if t_rise is None:
            t_rise = self.t_rise if self.t_rise is not None else self.default_t_rise
        if t_off is None:
            t_off = self.t_off if self.t_off is not None else self.default_t_off
        ca_recon = Neuron.get_restored_calcium(self.asp.data, t_rise, t_off)
        ca_recon_scaled = self.ca_scaler.transform(ca_recon.reshape(-1, 1)).reshape(-1)
        if t_rise == (self.t_rise if self.t_rise is not None else self.default_t_rise) and t_off == (self.t_off if self.t_off is not None else self.default_t_off):
            self._reconstructed_scaled = ca_recon_scaled
        return ca_recon_scaled

    
    def get_mae(self):
        '''Get Mean Absolute Error between observed and reconstructed calcium.

        Computes MAE as the mean absolute deviation between observed calcium
        signal and reconstruction from detected spikes. Provides intuitive
        measure of reconstruction quality in original signal units (ΔF/F).
        Caches the result.

        Returns
        -------
        float
            MAE value (non-negative). Lower is better.
            MAE < 0.1: Excellent reconstruction (typical for high SNR)
            MAE 0.1-0.2: Good reconstruction (acceptable quality)
            MAE 0.2-0.3: Moderate reconstruction (check for issues)
            MAE > 0.3: Poor reconstruction (likely artifacts or failures)

        Raises
        ------
        ValueError
            If amplitude spike data is not available.

        Notes
        -----
        MAE = mean(|Ca_observed - Ca_reconstructed|)

        Unlike RMSE, MAE treats all errors equally (no squaring). Useful
        for understanding typical deviation and detecting outlier sensitivity:
        - If RMSE >> MAE: Few large errors dominate (e.g., missed events)
        - If RMSE ≈ MAE: Errors uniformly distributed (e.g., white noise)

        The MAE/RMSE ratio can diagnose error distribution patterns.
        '''
        if self.mae is None:
            if self.asp is None:
                raise ValueError('Amplitude spikes required for MAE calculation. Call reconstruct_spikes() first.')
            t_rise = self.t_rise if self.t_rise is not None else self.default_t_rise
            t_off = self.t_off if self.t_off is not None else self.default_t_off
            ca_fitted = Neuron.get_restored_calcium(self.asp.data, t_rise, t_off)
            self.mae = np.mean(np.abs(self.ca.data - ca_fitted))
        return self.mae

    
    def get_event_rmse(self, n_mad=4.0, use_detected_events=True):
        '''Get RMSE during event periods only.

        Measures reconstruction error only during calcium transient events,
        ignoring baseline noise. Provides a more accurate measure of
        deconvolution quality than full-signal RMSE, which is dominated by
        baseline regions that reconstruction should NOT fit.

        Parameters
        ----------
        n_mad : float, default=4.0
            Number of MAD (Median Absolute Deviation) units above median
            for event detection threshold. Only used if use_detected_events=False.
        use_detected_events : bool, default=True
            If True, use self.events (wavelet-detected) for event mask.
            If False, use MAD-based threshold (legacy behavior).

        Returns
        -------
        float
            Event RMSE (lower is better). Same units as calcium signal (ΔF/F).
            Event RMSE < 0.05: Excellent event reconstruction
            Event RMSE 0.05-0.10: Good event reconstruction
            Event RMSE 0.10-0.15: Moderate event reconstruction
            Event RMSE > 0.15: Poor event reconstruction

        Raises
        ------
        ValueError
            If amplitude spike data is not available or no events detected.

        Notes
        -----
        Event_RMSE = sqrt(mean((Ca_events - Recon_events)^2))

        Where events are defined as: Ca > median + n_mad * MAD

        Complementary to Event R² - while Event R² shows proportion of variance
        explained, Event RMSE shows absolute reconstruction error magnitude.
        '''
        if self.asp is None:
            raise ValueError('Amplitude spikes required for Event RMSE calculation. Call reconstruct_spikes() first.')
        t_rise = self.t_rise if self.t_rise is not None else self.default_t_rise
        t_off = self.t_off if self.t_off is not None else self.default_t_off
        ca_reconstructed = Neuron.get_restored_calcium(self.asp.data, t_rise, t_off)

        # Determine event mask
        if use_detected_events and self.events is not None:
            event_mask = self.events.data > 0
        else:
            # MAD-based threshold (legacy)
            median = np.median(self.ca.scdata)
            mad = np.median(np.abs(self.ca.scdata - median)) * MAD_SCALE_FACTOR
            threshold = median + n_mad * mad
            event_mask = self.ca.scdata > threshold

        if np.sum(event_mask) == 0:
            raise ValueError(f'''No events detected. Consider lowering n_mad parameter or check wavelet detection.''')
        ca_events = self.ca.data[event_mask]
        recon_events = ca_reconstructed[event_mask]
        residuals = ca_events - recon_events
        event_rmse = float(np.sqrt(np.mean(residuals ** 2)))
        return event_rmse

    
    def get_event_mae(self, n_mad=4.0, use_detected_events=True):
        '''Get MAE during event periods only.

        Measures reconstruction error only during calcium transient events,
        ignoring baseline noise. Provides a more accurate measure of
        deconvolution quality than full-signal MAE, which is dominated by
        baseline regions that reconstruction should NOT fit.

        Parameters
        ----------
        n_mad : float, default=4.0
            Number of MAD (Median Absolute Deviation) units above median
            for event detection threshold. Only used if use_detected_events=False.
        use_detected_events : bool, default=True
            If True, use self.events (wavelet-detected) for event mask.
            If False, use MAD-based threshold (legacy behavior).

        Returns
        -------
        float
            Event MAE (lower is better). Same units as calcium signal (ΔF/F).
            Event MAE < 0.05: Excellent event reconstruction
            Event MAE 0.05-0.10: Good event reconstruction
            Event MAE 0.10-0.15: Moderate event reconstruction
            Event MAE > 0.15: Poor event reconstruction

        Raises
        ------
        ValueError
            If amplitude spike data is not available or no events detected.

        Notes
        -----
        Event_MAE = mean(|Ca_events - Recon_events|)

        Where events are defined as: Ca > median + n_mad * MAD

        Unlike Event RMSE, Event MAE does not square errors, making it less
        sensitive to outliers. Useful for understanding typical deviation:
        - If Event_RMSE >> Event_MAE: Few large errors (missed/false events)
        - If Event_RMSE ≈ Event_MAE: Errors uniformly distributed
        '''
        if self.asp is None:
            raise ValueError('Amplitude spikes required for Event MAE calculation. Call reconstruct_spikes() first.')
        t_rise = self.t_rise if self.t_rise is not None else self.default_t_rise
        t_off = self.t_off if self.t_off is not None else self.default_t_off
        ca_reconstructed = Neuron.get_restored_calcium(self.asp.data, t_rise, t_off)

        # Determine event mask
        if use_detected_events and self.events is not None:
            event_mask = self.events.data > 0
        else:
            # MAD-based threshold (legacy)
            median = np.median(self.ca.scdata)
            mad = np.median(np.abs(self.ca.scdata - median)) * MAD_SCALE_FACTOR
            threshold = median + n_mad * mad
            event_mask = self.ca.scdata > threshold

        if np.sum(event_mask) == 0:
            raise ValueError(f'''No events detected. Consider lowering n_mad parameter or check wavelet detection.''')
        ca_events = self.ca.data[event_mask]
        recon_events = ca_reconstructed[event_mask]
        event_mae = float(np.mean(np.abs(ca_events - recon_events)))
        return event_mae

    
    def get_event_count(self = None, n_mad = None):
        '''Get count of detected spike events (ridges passing thresholds).

        Counts the number of spike events detected by wavelet reconstruction
        that pass all thresholds. Returns count of non-zero entries in
        amplitude spike data. Result is cached for efficiency.

        Parameters
        ----------
        n_mad : float, default=3.0
            Not currently used - reserved for future threshold filtering.
            Kept for API consistency with other event-based methods.

        Returns
        -------
        int
            Number of detected spike events (ridges).

        Raises
        ------
        ValueError
            If spike reconstruction has not been performed.

        Notes
        -----
        Counts non-zero entries in self.asp.data (amplitude spike array).
        Each non-zero entry represents a detected calcium event that passed
        wavelet ridge filtering and amplitude thresholds during reconstruction.

        Useful for:
        - Assessing detection sensitivity
        - Computing event-based statistics
        - Validating reconstruction parameters
        '''
        if self.event_count is None:
            if self.asp is None:
                raise ValueError('Spike reconstruction required for event counting. Call reconstruct_spikes() first.')
            self.event_count = int(np.count_nonzero(self.asp.data))
        return self.event_count

    
    def get_baseline_noise_std(self, n_mad=3.0):
        '''Get baseline noise standard deviation from residual signal.

        Estimates noise level as the standard deviation of reconstruction
        residuals (original - reconstructed). This measures the actual
        reconstruction error, including measurement noise, unmodeled dynamics,
        and reconstruction inaccuracies.

        Parameters
        ----------
        n_mad : float, default=3.0
            Not used in this implementation. Kept for API compatibility.

        Returns
        -------
        float
            Standard deviation of residual signal.

        Raises
        ------
        ValueError
            If reconstruction has not been performed.

        Notes
        -----
        Residual = Ca_original - Ca_reconstructed

        The residual contains:
        - Measurement noise
        - Unmodeled small events
        - Drift and baseline fluctuations
        - Reconstruction errors

        This is the proper noise estimate for normalized metrics (NMAE, NRMSE).
        '''
        if self.asp is None:
            raise ValueError('Spike reconstruction required for noise estimation. Call reconstruct_spikes() first.')
        recon = self.reconstructed
        if recon is None:
            raise ValueError('Reconstruction failed or returned None')
        residuals = self.ca.data - recon.data
        return float(np.std(residuals))

    
    def get_event_snr(self, n_mad=4.0):
        '''Get event SNR in dB (signal quality metric).

        Computes SNR as the ratio of mean event amplitude to baseline noise std,
        expressed in decibels. Measures how clearly calcium events stand out
        from baseline noise. Higher values indicate cleaner signals.

        Parameters
        ----------
        n_mad : float, default=3.0
            Number of MAD units above median for event detection threshold.

        Returns
        -------
        float
            Event SNR in decibels (dB). Higher is better.
            SNR > 15 dB: Excellent signal quality
            SNR 10-15 dB: Good signal quality
            SNR 5-10 dB: Moderate signal quality
            SNR < 5 dB: Poor signal quality (noisy)

        Raises
        ------
        ValueError
            If no events detected or baseline noise is zero.

        Notes
        -----
        SNR_dB = 20 * log10(mean(events) / std(baseline))

        This metric assesses signal quality independent of reconstruction.
        Low SNR suggests noisy data or detection threshold issues.
        '''
        if self.ca is None or len(self.ca.data) == 0:
            raise ValueError('Calcium signal data required for event SNR')
        median = np.median(self.ca.data)
        mad = np.median(np.abs(self.ca.data - median)) * MAD_SCALE_FACTOR
        threshold = median + n_mad * mad
        event_mask = self.ca.data > threshold
        baseline_mask = ~event_mask
        event_data = self.ca.data[event_mask]
        baseline_data = self.ca.data[baseline_mask]
        if len(event_data) == 0:
            raise ValueError('No events detected. Consider lowering n_mad parameter.')
        if len(baseline_data) < 10:
            raise ValueError('Insufficient baseline data for noise estimation.')
        event_mean = np.mean(event_data)
        baseline_std = np.std(baseline_data)
        if baseline_std == 0:
            raise ValueError('Baseline noise std is zero, cannot compute SNR')
        snr_linear = event_mean / baseline_std
        return float(20 * np.log10(snr_linear))


    def get_wavelet_snr(self):
        '''Get event-based signal-to-noise ratio.

        Uses detected event regions to separate signal from baseline,
        providing accurate SNR measurement that accounts for event timing and
        shape. Works with both wavelet and threshold detection methods.

        Returns
        -------
        float
            Event SNR (signal_strength / baseline_noise).
            Higher values indicate better signal quality.

        Raises
        ------
        ValueError
            If reconstruction not performed with create_event_regions=True,
            no events detected, insufficient data (< 3 events), or baseline
            noise is zero.

        Notes
        -----
        Requires prior call to:
            neuron.reconstruct_spikes(method='wavelet', create_event_regions=True)
            OR
            neuron.reconstruct_spikes(method='threshold', create_event_regions=True)

        SNR calculation:
        1. Baseline: median and MAD from non-event frames
        2. Signal: median of peak amplitudes across all events
        3. SNR = (signal - baseline_median) / baseline_noise

        Uses peak amplitudes (not event medians) to correctly handle sparse
        high-amplitude events.

        See Also
        --------
        get_snr : Simple SNR based on spike times
        get_event_snr : Alias for this method
        reconstruct_spikes : Spike reconstruction (wavelet or threshold)

        Examples
        --------
        >>> neuron = Neuron(cell_id=0, ca=calcium_data, sp=None)
        >>> neuron.reconstruct_spikes(method='threshold', create_event_regions=True)
        >>> snr = neuron.get_event_snr()  # or get_wavelet_snr()
        >>> print(f"Signal quality (SNR): {snr:.2f}")
        '''
        if self.wavelet_snr is None:
            self.wavelet_snr = self._calc_wavelet_snr()
        return self.wavelet_snr

    # Alias for method-agnostic naming
    get_event_snr = get_wavelet_snr


    def _calc_wavelet_snr(self):
        '''Calculate event-based SNR using detected event regions.

        Internal method that computes SNR from detected event regions.
        Works with both wavelet and threshold detection methods.
        Uses peak amplitudes to handle sparse high-amplitude events correctly.

        Returns
        -------
        float
            Event SNR value.

        Raises
        ------
        ValueError
            If events not detected, insufficient data, or baseline noise is zero.
        '''
        # Try self.events first, then construct from threshold_events or wvt_ridges
        events_mask = None
        if self.events is not None and self.events.data is not None:
            events_mask = self.events.data.astype(bool)
        elif hasattr(self, 'threshold_events') and self.threshold_events:
            # Construct mask from threshold events
            events_mask = np.zeros(len(self.ca.data), dtype=bool)
            for event in self.threshold_events:
                st, end = int(event.start), min(int(event.end), len(self.ca.data))
                events_mask[st:end] = True
        elif self.wvt_ridges:
            # Construct mask from wavelet ridges
            events_mask = np.zeros(len(self.ca.data), dtype=bool)
            for ridge in self.wvt_ridges:
                st, end = int(ridge.start), min(int(ridge.end), len(self.ca.data))
                events_mask[st:end] = True
        else:
            raise ValueError(
                'No event regions detected. '
                'Call reconstruct_spikes(create_event_regions=True) first, '
                'or use detect_events_threshold() to detect events.'
            )

        ca = self.ca.data

        # Check if any events detected
        if not np.any(events_mask):
            raise ValueError('No events in event mask')

        # Calculate baseline from non-event regions
        baseline_mask = ~events_mask
        baseline_values = ca[baseline_mask]

        if len(baseline_values) < 10:
            raise ValueError(
                f'Insufficient baseline frames ({len(baseline_values)}). '
                'Need at least 10 baseline frames.'
            )

        baseline_median = np.median(baseline_values)
        baseline_noise = median_abs_deviation(baseline_values, scale='normal')

        if baseline_noise == 0:
            raise ValueError('Baseline noise is zero, cannot compute SNR')

        # Identify individual events (transitions in mask)
        event_starts = np.where(np.diff(events_mask.astype(int)) == 1)[0] + 1
        event_ends = np.where(np.diff(events_mask.astype(int)) == -1)[0] + 1

        # Handle edge cases
        if events_mask[0]:
            event_starts = np.concatenate([[0], event_starts])
        if events_mask[-1]:
            event_ends = np.concatenate([event_ends, [len(events_mask)]])

        n_events = min(len(event_starts), len(event_ends))

        if n_events < 3:
            raise ValueError(
                f'Too few events detected ({n_events}). '
                'Need at least 3 events for reliable SNR calculation.'
            )

        # Extract peak amplitude for each event
        event_amplitudes = []
        for i in range(n_events):
            event_region = ca[event_starts[i]:event_ends[i]]
            if len(event_region) > 0:
                event_amplitudes.append(np.max(event_region))

        event_amplitudes = np.array(event_amplitudes)

        if len(event_amplitudes) == 0:
            raise ValueError('No valid event amplitudes extracted')

        # Signal strength using peak amplitudes
        signal_strength = np.median(event_amplitudes) - baseline_median

        # Robust SNR
        snr_wavelet = signal_strength / baseline_noise

        return float(snr_wavelet)


    def get_nmae(self, n_mad=3.0):
        '''Get Normalized Mean Absolute Error (MAE / baseline_noise_std).

        Computes MAE divided by baseline noise standard deviation. This
        normalization accounts for varying noise levels across neurons,
        enabling fair quality comparison. Values indicate error magnitude
        relative to baseline noise level.

        Parameters
        ----------
        n_mad : float, default=3.0
            Number of MAD units above median for baseline detection.

        Returns
        -------
        float
            Normalized MAE (dimensionless ratio). Lower is better.
            NMAE < 1.0: Error less than baseline noise (excellent)
            NMAE < 2.0: Error ~2× baseline noise (good)
            NMAE < 3.0: Error ~3× baseline noise (moderate)
            NMAE > 3.0: Error >> baseline noise (poor)

        Raises
        ------
        ValueError
            If baseline noise cannot be estimated or is zero.

        Notes
        -----
        NMAE = MAE / std(baseline)

        Unlike raw MAE, NMAE is comparable across neurons with different
        noise levels. Interpretation: error magnitude in "noise units".
        '''
        mae = self.get_mae()
        baseline_std = self.get_baseline_noise_std(n_mad=n_mad)
        if baseline_std == 0:
            raise ValueError('Baseline noise std is zero, cannot normalize MAE')
        return float(mae / baseline_std)

    
    def get_nrmse(self, n_mad=3.0):
        '''Get Normalized RMSE (RMSE / baseline_noise_std).

        Computes RMSE divided by baseline noise standard deviation. This
        normalization accounts for varying noise levels across neurons.
        More sensitive to large errors than NMAE due to squaring.

        Parameters
        ----------
        n_mad : float, default=3.0
            Number of MAD units above median for baseline detection.

        Returns
        -------
        float
            Normalized RMSE (dimensionless ratio). Lower is better.
            NRMSE < 1.0: Error less than baseline noise (excellent)
            NRMSE < 2.0: Error ~2× baseline noise (good)
            NRMSE < 3.0: Error ~3× baseline noise (moderate)
            NRMSE > 3.0: Error >> baseline noise (poor)

        Raises
        ------
        ValueError
            If baseline noise cannot be estimated or is zero.

        Notes
        -----
        NRMSE = RMSE / std(baseline)

        NRMSE > NMAE indicates few large errors dominate (e.g., missed events).
        NRMSE ≈ NMAE indicates uniformly distributed errors.
        '''
        rmse = self.get_noise_ampl()
        baseline_std = self.get_baseline_noise_std(n_mad)
        if baseline_std == 0:
            raise ValueError('Baseline noise std is zero, cannot normalize RMSE')
        return float(rmse / baseline_std)

    
    def get_kinetics(self, method='direct', fps=20, use_cached=True, update_reconstruction=True, **kwargs):
        '''Get optimized calcium kinetics parameters (t_rise, t_off).

        Simple wrapper around optimize_kinetics() for easy access to optimized parameters.
        Caches results for efficiency - only runs optimization once per neuron.

        Updates neuron attributes:
        - self.t_rise: optimized rise time (frames)
        - self.t_off: optimized decay time (frames)
        - self._kinetics_info: full optimization results dict

        Parameters
        ----------
        method : str, optional
            Optimization method. Currently only 'direct' is supported.
            Default: 'direct'.
        fps : float, optional
            Sampling rate in frames per second. Default: 20.0.
        use_cached : bool, optional
            If True and optimization already run, return cached results.
            If False, always re-run optimization. Default: True.
        update_reconstruction : bool, optional
            If True, recompute reconstruction with optimized kinetics.
            Default: True.
        **kwargs : dict, optional
            Additional arguments passed to optimize_kinetics()
            (e.g., t_rise_range, t_off_range, ftol, gtol, maxiter, etc.)

        Returns
        -------
        dict
            Dictionary with keys:
            - \'t_rise\': float, rise time in seconds
            - \'t_off\': float, decay time in seconds
            - \'t_rise_frames\': float, rise time in frames
            - \'t_off_frames\': float, decay time in frames
            - \'event_r2\': float, event R² with optimized parameters
            - \'default_r2\': float, event R² with default parameters
            - \'improvement\': float, R² improvement
            - \'method\': str, optimization method used
            - (additional fields from optimize_kinetics())

        Raises
        ------
        ValueError
            If no events detected (call reconstruct_spikes() first).

        Examples
        --------
        >>> # Optimize kinetics and access via attributes
        >>> neuron = Neuron(cell_id=0, ca=calcium_trace, sp=None, fps=20)
        >>> neuron.reconstruct_spikes(method=\'wavelet\')
        >>> result = neuron.get_kinetics(fps=20)
        >>>
        >>> # Access optimized parameters via attributes (in frames)
        >>> t_rise_frames = neuron.t_rise
        >>> t_off_frames = neuron.t_off
        >>>
        >>> # Or via returned dict (in seconds)
        >>> t_rise_sec = result[\'t_rise\']
        >>> t_off_sec = result[\'t_off\']
        >>> print(f"R² improvement: {result[\'improvement\']:+.3f}")
        >>>
        >>> # Access full optimization info
        >>> print(f"Converged: {neuron._kinetics_info[\'converged\']}")
        >>> print(f"Time: {neuron._kinetics_info[\'time\']:.2f}s")

        Notes
        -----
        - Results are cached in self._kinetics_info after first call
        - Optimized parameters stored in self.t_rise and self.t_off (frames)
        - Uses fast direct derivative measurement method
        - For backward compatibility: neuron.t_off now contains optimized value
        '''
        if use_cached and hasattr(self, '_kinetics_info') and self._kinetics_info is not None:
            return self._kinetics_info

        # Run optimization and optionally update reconstruction with new kinetics
        result = self.optimize_kinetics(
            method=method,
            fps=fps,
            update_reconstruction=update_reconstruction,
            **kwargs
        )

        # Cache the result
        self._kinetics_info = result
        return result


    def get_t_off(self):
        """Get calcium decay time constant.

        .. deprecated:: 0.5.0
           Use :meth:`get_kinetics` instead for better optimization that jointly
           optimizes both t_rise and t_off using correct event R² metric.
           This method only fits t_off using simple MSE and is significantly slower.

        Fits the decay time constant by optimizing the match between
        observed calcium and reconstructed calcium from spikes. Caches
        the result for efficiency.

        Returns
        -------
        float
            Decay time constant in frames.

        Raises
        ------
        ValueError
            If spike data is not available for fitting.

        Notes
        -----
        Uses scipy.optimize.minimize to find the optimal t_off value
        that minimizes the RMSE between observed and reconstructed calcium.

        **DEPRECATED**: Use get_kinetics() instead:

        >>> # Old way (deprecated)
        >>> t_off = neuron.get_t_off()

        >>> # New way (recommended)
        >>> kinetics = neuron.get_kinetics()
        >>> t_off = kinetics['t_off_frames']  # in frames
        >>> t_rise = kinetics['t_rise_frames']  # bonus: also get t_rise
        """
        import warnings
        warnings.warn("get_t_off() is deprecated and will be removed in v0.6.0. Use get_kinetics() instead for better joint optimization of t_rise and t_off. Example: kinetics = neuron.get_kinetics(); t_off = kinetics['t_off_frames']", DeprecationWarning, stacklevel=2)
        if self.t_off is None:
            (self.t_off, self.noise_ampl) = self._fit_t_off()
        return self.t_off

    
    def get_noise_ampl(self):
        '''Get noise amplitude estimate from calcium-spike reconstruction.

        Returns the root mean square error (RMSE) between observed calcium
        and reconstructed calcium from spikes using current kinetics parameters.
        This provides an estimate of the noise level in the calcium signal
        after accounting for spike-related transients.

        Returns
        -------
        float
            RMSE between observed and reconstructed calcium signal.

        Raises
        ------
        ValueError
            If spike data is not available for reconstruction.

        Notes
        -----
        Uses cached reconstruction from get_reconstructed() which respects
        optimized kinetics (self.t_rise, self.t_off). This is different from
        MAD, as it specifically measures the residual error after spike-to-calcium
        reconstruction. The value is cached after first computation.
        '''
        if self.noise_ampl is None:
            # Prefer asp, fall back to sp for backward compatibility
            if self.asp is None and self.sp is None:
                raise ValueError('Spike reconstruction required for noise amplitude. Call reconstruct_spikes() first.')
            recon = self.get_reconstructed()
            if recon is None:
                raise ValueError('Reconstruction failed or returned None')
            residuals = self.ca.data - recon.data
            self.noise_ampl = float(np.sqrt(np.mean(residuals ** 2)))
        return self.noise_ampl

    
    def _fit_t_off(self):
        '''Fit optimal calcium decay time constant from spike-calcium pairs.
        
        Uses scipy.optimize.minimize to find the t_off value that minimizes
        the RMSE between observed and reconstructed calcium signals.
        
        Returns
        -------
        tuple
            (t_off, rmse) where t_off is the optimal decay time constant
            in frames (capped at 5x default) and rmse is the reconstruction
            error.
            
        Raises
        ------
        ValueError
            If spike data is not available.
            
        Notes
        -----
        If the fitted t_off exceeds 5x the default value, it is capped
        and a warning is logged, as this typically indicates signal
        quality issues.        '''
        if self.asp is not None and np.sum(np.abs(self.asp.data)) > 0:
            spike_data = self.asp.data
        elif self.sp is not None:
            spike_data = self.sp.data
        else:
            raise ValueError('Spike data required for t_off fitting')
        res = minimize(
            Neuron.ca_mse_error,
            np.array([self.default_t_off]),
            args=(self.ca.data, spike_data, self.default_t_rise),
            bounds=[(self.default_t_rise * 1.1, None)]
        )
        opt_t_off = res.x[0]
        noise_amplitude = res.fun
        logger = logging.getLogger(__name__)
        if opt_t_off <= self.default_t_rise:
            logger.warning(f'''Optimization failed for neuron {self.cell_id}: fitted t_off ({opt_t_off:.2f}) <= t_rise ({self.default_t_rise:.2f}). Using
 default t_off={self.default_t_off}''')
            opt_t_off = self.default_t_off
        elif opt_t_off > self.default_t_off * 5:
            logger.warning(f'''Calculated t_off={int(opt_t_off)} for neuron {self.cell_id} is suspiciously high, check signal quality. t_off has been automatically lowered to {self.default_t_off * 5}''')
            opt_t_off = self.default_t_off * 5
        if not res.success:
            logger.warning(f'''Optimization did not converge for neuron {self.cell_id}: {res.message}. Using fitted value {opt_t_off:.2f} with caution.''')
        return (opt_t_off, noise_amplitude)

    
    def get_shuffled_calcium(self, method="roll_based", return_array=True, seed=None, **kwargs):
        """Get shuffled calcium signal using various randomization methods.
        
        Creates surrogate data that preserves certain statistical properties
        of the original calcium signal while destroying temporal relationships.
        
        Parameters
        ----------
        method : {'roll_based', 'waveform_based', 'chunks_based'}, optional
            Shuffling method to use:
            - 'roll_based': Circular shift by random offset
            - 'waveform_based': Shuffle spikes then reconstruct calcium
            - 'chunks_based': Divide signal into chunks and reorder
            Default is 'roll_based'.
        return_array : bool, optional
            If True, return numpy array. If False, return TimeSeries object.
            Default is True.
        seed : int, optional
            Random seed for reproducible shuffling.
        **kwargs
            Additional arguments passed to shuffling method:
            - For chunks_based: n (int) - number of chunks
            
        Returns
        -------
        ndarray or TimeSeries
            Shuffled calcium signal with same length as original.
            
        Raises
        ------
        AttributeError
            If the specified method does not exist.
            
        Notes
        -----
        Different methods preserve different signal properties:
        - roll_based: Preserves all autocorrelations
        - waveform_based: Preserves spike waveform shapes  
        - chunks_based: Preserves local signal structure within chunks        """
        valid_methods = [
            'roll_based',
            'waveform_based',
            'chunks_based']
        if method not in valid_methods:
            raise ValueError(f'''Invalid method \'{method}\'. Must be one of {valid_methods}''')
        fn = getattr(self, f'''_shuffle_calcium_data_{method}''')

        # Call the shuffling method
        shuffled_data = fn(seed=seed, **kwargs)

        # Return as array or TimeSeries based on return_array parameter
        if return_array:
            return shuffled_data
        else:
            from ..information.info_base import TimeSeries
            return TimeSeries(data=shuffled_data, discrete=False)


    def _shuffle_calcium_data_waveform_based(self, seed=None, **kwargs):
        '''Shuffle calcium by reconstructing from ISI-shuffled spikes.
        
        Preserves spike waveform shapes while randomizing spike timing
        based on inter-spike interval statistics.
        
        Parameters
        ----------
        seed : int, optional
            Random seed for reproducible shuffling.
        **kwargs
            Additional arguments (for compatibility with shuffle interface).
            
        Returns
        -------
        ndarray
            Shuffled calcium signal.
            
        Raises
        ------
        ValueError
            If spike data is not available.        '''
        if self.asp is not None and np.sum(np.abs(self.asp.data)) > 0:
            spike_data = self.asp.data
        elif self.sp is not None:
            spike_data = self.sp.data
        else:
            raise ValueError('Spike data required for waveform-based shuffling')

        # Waveform-based shuffling REQUIRES accurate kinetics for proper background extraction
        # Optimize if not already done
        if self.t_rise is None or self.t_off is None:
            try:
                self.get_kinetics(fps=self.fps)
            except (ValueError, AttributeError):
                # Optimization failed - use defaults
                import warnings
                warnings.warn(
                    f"Kinetics optimization failed for neuron {self.cell_id}. "
                    "Using default kinetics for waveform-based shuffling. "
                    "Results may be less accurate.",
                    UserWarning
                )

        # Use optimized kinetics if available, otherwise fall back to defaults
        opt_t_rise = self.t_rise if self.t_rise is not None else self.default_t_rise
        opt_t_off = self.t_off if self.t_off is not None else self.default_t_off
        conv = Neuron.get_restored_calcium(spike_data, opt_t_rise, opt_t_off)
        background = self.ca.data - conv[:len(self.ca.data)]
        pspk = self._shuffle_spikes_data_isi_based(seed=seed)
        if self.asp is not None and np.sum(np.abs(self.asp.data)) > 0:
            amp_indices = np.nonzero(self.asp.data)[0]
            amplitudes = self.asp.data[amp_indices]
            pspk_scaled = pspk.astype(float)
            pspk_indices = np.nonzero(pspk)[0]
            if len(pspk_indices) == len(amplitudes):
                pspk_scaled[pspk_indices] = amplitudes
            elif len(pspk_indices) > 0:
                pspk_scaled[pspk_indices] = np.mean(amplitudes)
            psconv = Neuron.get_restored_calcium(pspk_scaled, opt_t_rise, opt_t_off)
        else:
            psconv = Neuron.get_restored_calcium(pspk, opt_t_rise, opt_t_off)
        shuf_ca = psconv[:len(self.ca.data)] + background
        return shuf_ca

    
    def _shuffle_calcium_data_chunks_based(self, n=100, seed=None, **kwargs):
        '''Shuffle calcium by dividing into chunks and reordering.
        
        Preserves local calcium dynamics within chunks while destroying
        long-range temporal relationships.
        
        Parameters
        ----------
        n : int, optional
            Number of chunks to divide signal into. Must be positive.
            Default is 100.
        seed : int, optional
            Random seed for reproducible shuffling.
        **kwargs
            Additional keyword arguments (unused, for API compatibility).
            
        Returns
        -------
        ndarray
            Shuffled calcium signal with same length as original.
            
        Notes
        -----
        - Chunks may have unequal sizes if signal length not divisible by n
        - Preserves local dynamics (within-chunk patterns)
        - Destroys global temporal structure
        - Useful for testing significance of long-range correlations        '''
        check_positive(n=n)
        if seed is not None:
            np.random.seed(seed)
        ca = self.ca.data
        chunks = np.array_split(ca, n)
        inds = np.arange(len(chunks))
        np.random.shuffle(inds)
        shuf_ca = np.concatenate([chunks[i] for i in inds])
        return shuf_ca

    
    def _shuffle_calcium_data_roll_based(self, shift=None, seed=None, **kwargs):
        '''Shuffle calcium by circular shift (rolling).
        
        Preserves all autocorrelations and power spectrum while destroying
        temporal relationships with external signals.
        
        Parameters
        ----------
        shift : int, optional
            Shift amount in frames. If None, randomly chosen between
            3*t_off and n_frames-3*t_off.
        seed : int, optional
            Random seed for reproducible shuffling.
        **kwargs
            Additional arguments (for compatibility with shuffle interface).
            
        Returns
        -------
        ndarray
            Circularly shifted calcium signal.
            
        Raises
        ------
        ValueError
            If signal is too short for valid shuffling range.        '''
        # Roll-based shuffling only needs t_off for shift range calculation
        # Constant offset is sufficient (doesn't need precise optimization)
        opt_t_off = self.t_off if self.t_off is not None else self.default_t_off
        if shift is None:
            if seed is not None:
                np.random.seed(seed)
            min_shift = int(3 * opt_t_off)
            max_shift = self.n_frames - int(3 * opt_t_off)
            if min_shift >= max_shift:
                raise ValueError(f'''Signal too short for roll-based shuffling. Need at least {2 * int(3 * opt_t_off)} frames, but have {self.n_frames}''')
            shift = np.random.randint(min_shift, max_shift)
        elif not isinstance(shift, (int, np.integer)):
            raise ValueError(f'''shift must be integer, got {type(shift).__name__}''')
        if shift < 0 or shift >= self.n_frames:
            raise ValueError(f'''shift must be in range [0, {self.n_frames - 1}], got {shift}''')
        shuf_ca = np.roll(self.ca.data, shift)
        return shuf_ca

    
    def get_shuffled_spikes(self, method="isi_based", return_array=True, seed=None, **kwargs):
        """Get shuffled spike train.
        
        Creates surrogate spike data that preserves certain statistical
        properties while destroying temporal relationships.
        
        Parameters
        ----------
        method : {'isi_based'}, optional
            Shuffling method to use. Currently only 'isi_based' is supported,
            which preserves inter-spike interval statistics.
            Default is 'isi_based'.
        return_array : bool, optional
            If True, return numpy array. If False, return TimeSeries object.
            Default is True.
        seed : int, optional
            Random seed for reproducible shuffling.
        **kwargs
            Additional arguments passed to shuffling method.
            
        Returns
        -------
        ndarray or TimeSeries
            Shuffled spike train with same number of spikes as original.
            
        Raises
        ------
        AttributeError
            If no spike data is available.
        ValueError
            If method is not recognized.
            
        Notes
        -----
        The ISI-based method preserves the distribution of inter-spike
        intervals while randomizing spike positions.        """
        if self.sp is None:
            raise AttributeError('Unable to shuffle spikes without spikes data')
        valid_methods = [
            'isi_based']
        if method not in valid_methods:
            raise ValueError(f'''Invalid method \'{method}\'. Must be one of {valid_methods}''')
        fn = getattr(self, f'''_shuffle_spikes_data_{method}''')

        # Call the shuffling method
        shuffled_data = fn(seed=seed, **kwargs)

        # Return as array or TimeSeries based on return_array parameter
        if return_array:
            return shuffled_data
        else:
            from ..information.info_base import TimeSeries
            return TimeSeries(data=shuffled_data, discrete=True)  # discrete=True for spikes


    def reconstructed(self):
        '''Get reconstructed calcium signal from amplitude spikes (cached).

        Convenience property that returns cached reconstruction. For more
        control, use get_reconstructed() method instead.

        Returns
        -------
        TimeSeries or None
            Reconstructed calcium signal as TimeSeries object. Returns None
            if amplitude spike data is not available.

        See Also
        --------
        get_reconstructed : Method with force_reconstruction and custom parameters
        '''
        return self.get_reconstructed()

    reconstructed = property(reconstructed)
    
    def get_reconstructed(self, force_reconstruction=False, **kwargs):
        """Get reconstructed calcium signal from amplitude spikes.

        Lazily computes and caches the reconstructed calcium signal by
        convolving detected amplitude spikes with the calcium kernel.

        Parameters
        ----------
        force_reconstruction : bool, optional
            If True, bypass cache and force recomputation. Default is False.
        **kwargs : dict, optional
            Custom reconstruction parameters:

            - t_rise_frames : float, optional
                Custom rise time in frames. If not provided, uses optimized t_rise
                from get_kinetics().
            - t_off_frames : float, optional
                Custom decay time in frames. If not provided, uses optimized t_off
                from get_kinetics().
            - spike_data : array-like, optional
                Custom spike data to reconstruct from. If not provided, uses self.asp.data.

        Returns
        -------
        TimeSeries or None
            Reconstructed calcium signal as TimeSeries object. Returns None
            if amplitude spike data is not available (call reconstruct_spikes()
            first) and no custom spike_data provided.

        Notes
        -----
        The default reconstruction uses:
        - self.asp.data: Amplitude spikes (dF/F units)
        - self.t_rise: Optimized rise time from get_kinetics() (lazy, cached)
        - self.t_off: Optimized decay time from get_kinetics() (lazy, cached)

        The reconstructed signal can be compared with self.ca to assess
        reconstruction quality. Use get_reconstruction_r2(), get_mae(),
        or get_snr_reconstruction() for quantitative metrics.

        Custom parameters are NOT cached. Only default reconstruction is cached.

        Examples
        --------
        >>> neuron = Neuron(cell_id=1, ca=calcium_data, sp=None, fps=20)
        >>> neuron.reconstruct_spikes(method='wavelet')
        >>>
        >>> # Get cached 
default reconstruction
        >>> recon = neuron.reconstructed
        >>>
        >>> # Force recomputation
        >>> recon = neuron.get_reconstructed(force_reconstruction=True)
        >>>
        >>> # Use custom decay time
        >>> recon_fast = neuron.get_reconstructed(t_off_frames=20)
        >>> recon_slow = neuron.get_reconstructed(t_off_frames=60)
        """
        has_custom_params = bool(kwargs)
        if has_custom_params or force_reconstruction:
            spike_data = kwargs.get('spike_data', None)
            if spike_data is None:
                # Prefer asp, fall back to sp for backward compatibility
                if self.asp is not None:
                    spike_data = self.asp.data
                elif self.sp is not None:
                    import warnings
                    warnings.warn(
                        "Using binary spikes (sp) instead of amplitude spikes (asp). "
                        "Reconstruction will use uniform amplitudes. "
                        "Call reconstruct_spikes() to get amplitude-based reconstruction.",
                        UserWarning
                    )
                    spike_data = self.sp.data
                else:
                    return None
            t_rise_frames = kwargs.get('t_rise_frames', None)
            t_off_frames = kwargs.get('t_off_frames', None)
            if t_rise_frames is None:
                t_rise_frames = self.t_rise if self.t_rise is not None else self.default_t_rise
            if t_off_frames is None:
                t_off_frames = self.t_off if self.t_off is not None else self.default_t_off
            reconstructed_data = Neuron.get_restored_calcium(spike_data, t_rise_frames, t_off_frames)
            return TimeSeries(reconstructed_data, discrete=False)
        if self._reconstructed is None:
            # Prefer asp, fall back to sp for backward compatibility
            if self.asp is not None:
                spike_data = self.asp.data
            elif self.sp is not None:
                import warnings
                warnings.warn(
                    "Using binary spikes (sp) instead of amplitude spikes (asp). "
                    "Reconstruction will use uniform amplitudes. "
                    "Call reconstruct_spikes() to get amplitude-based reconstruction.",
                    UserWarning
                )
                spike_data = self.sp.data
            else:
                return None

            t_rise = self.t_rise if self.t_rise is not None else self.default_t_rise
            t_off = self.t_off if self.t_off is not None else self.default_t_off
            reconstructed_data = Neuron.get_restored_calcium(spike_data, t_rise, t_off)
            self._reconstructed = TimeSeries(reconstructed_data, discrete=False)
        return self._reconstructed

    
    def _shuffle_spikes_data_isi_based(self, seed=None):
        '''Shuffle spikes preserving inter-spike interval statistics.
        
        Randomizes spike positions while maintaining the distribution
        of time intervals between spikes.
        
        Parameters
        ----------
        seed : int, optional
            Random seed for reproducible shuffling.
            
        Returns
        -------
        ndarray
            Shuffled spike train with same length and number of spikes.
            Binary array where 1 indicates spike.
            
        Notes
        -----
        - Preserves ISI distribution but not ISI sequence order
        - First spike position is randomized within valid range
        - Handles edge cases: empty spike trains, boundary conditions
        - May produce different temporal patterns despite same ISI distribution        '''
        if seed is not None:
            np.random.seed(seed)
        nfr = self.n_frames
        pseudo_spikes = np.zeros(nfr)
        event_inds = np.where(self.sp.data != 0)[0]
        if len(event_inds) == 0:
            return self.sp.data
        event_vals = self.sp.data[event_inds].copy()
        event_range = max(event_inds) - min(event_inds)
        max_start = max(1, nfr - event_range - 1)
        first_random_pos = np.random.choice(max_start)
        interspike_intervals = np.diff(event_inds)
        rng = np.arange(len(interspike_intervals))
        np.random.shuffle(rng)
        disordered_interspike_intervals = interspike_intervals[rng]
        pseudo_event_inds = np.cumsum(np.insert(disordered_interspike_intervals, 0, first_random_pos))
        valid_mask = pseudo_event_inds < nfr
        pseudo_event_inds = pseudo_event_inds[valid_mask]
        event_vals = event_vals[:len(pseudo_event_inds)]
        np.random.shuffle(event_vals)
        pseudo_spikes[pseudo_event_inds] = event_vals
        return pseudo_spikes

    
    def optimize_kinetics(self, method='direct', fps=20, update_reconstruction=True,
                         max_event_dur_multiplier=4, detection_method='auto', **kwargs):
        """
        Universal kinetics optimization with automatic reconstruction update.

        Single entry point for all optimization methods. Automatically updates
        instance kinetics (self.t_rise, self.t_off) and optionally re-runs spike
        detection with new parameters.


        Parameters
        ----------
        method : str, optional
            Optimization method. Currently only 'direct' is supported.
            Default: 'direct'
        fps : float, optional
            Sampling rate in frames per second. Default: 20.0
        update_reconstruction : bool, optional
            If True, automatically re-run spike detection with optimized kinetics.
            This ensures events match the new parameters. Default: True.
        max_event_dur_multiplier : float, optional
            Multiplier for calculating max_event_dur when update_reconstruction=True.
            Formula: max_event_dur = t_rise + multiplier * t_off.
            Higher values detect longer events but may merge overlapping events.
            Lower values improve precision but may miss event tails.
            Recommended range: 3.0-5.0. Default: 4.0 (optimal balance).
        detection_method : {'auto', 'wavelet', 'threshold'}, optional
            Method to use for event re-detection when update_reconstruction=True:
            - 'auto': Use threshold if threshold_events exist, else wavelet (default)
            - 'wavelet': Always use wavelet detection (slower, more sensitive)
            - 'threshold': Always use threshold detection (faster, requires high SNR)
            Default: 'auto'
        **kwargs : dict
            Method-specific parameters including:
            - min_r2 : float, minimum R² for t_off fit quality (default: 0.8).
              Events with poor exponential fit are rejected.
            See _optimize_kinetics_direct() for full list.

        Returns
        -------
        dict
            Optimization results with keys:
            - 'optimized': bool, whether optimization succeeded
            - 't_rise': float, optimized rise time (seconds)
            - 't_off': float, optimized decay time (seconds)
            - 'method': str, method used
            - Additional method-specific metrics

        Examples
        --------
        >>> # Fast threshold-based workflow (100-500x faster)
        >>> neuron.detect_events_threshold(n_mad=4.0)
        >>> result = neuron.optimize_kinetics(method='direct', fps=30)
        >>> # Auto-detects threshold mode, re-runs threshold detection with optimized kinetics

        >>> # Explicit threshold mode for iterative refinement
        >>> neuron.detect_events_threshold(n_mad=4.0)
        >>> result = neuron.optimize_kinetics(
        ...     method='direct', fps=30, detection_method='threshold'
        ... )

        >>> # Traditional wavelet workflow (slower but more sensitive)
        >>> neuron.reconstruct_spikes(method='wavelet')
        >>> result = neuron.optimize_kinetics(method='direct', fps=30)
        >>> # Uses wavelet detection for re-detection

        >>> # Skip auto-reconstruction for speed (e.g., in batch processing)
        >>> result = neuron.optimize_kinetics(
        ...     method='direct', fps=30, update_reconstruction=False
        ... )
        >>> # Later: neuron.reconstruct_spikes(method='wavelet')

        Notes
        -----
        - Consistently updates self.t_rise/t_off and reconstructs events
        - Setting update_reconstruction=False allows manual control of reconstruction timing

        See Also
        --------
        _optimize_kinetics_direct : Fast derivative-based measurement
        """
        if method != 'direct':
            raise ValueError(f"Only 'direct' method is supported, got '{method}'")

        # Call direct optimization method with FPS-adaptive defaults
        default_max_forward = int(MAX_FRAMES_FORWARD_SEC * fps)
        default_max_back = int(MAX_FRAMES_BACK_SEC * fps)
        result = self._optimize_kinetics_direct(
            fps=fps,
            asp=kwargs.get('asp', None),
            wvt_ridges=kwargs.get('wvt_ridges', None),
            max_frames_forward=kwargs.get('max_frames_forward', default_max_forward),
            max_frames_back=kwargs.get('max_frames_back', default_max_back),
            min_events=kwargs.get('min_events', 5),
            aggregation=kwargs.get('aggregation', 'median'),
            min_r2=kwargs.get('min_r2', 0.8)
        )
        
        # Update instance kinetics if optimization succeeded
        if result.get('optimized', False):
            self.t_rise = result['t_rise'] * fps  # Convert seconds to frames
            self.t_off = result['t_off'] * fps

            # Optionally update reconstruction with new kinetics
            if update_reconstruction:
                # Determine which detection method to use
                if detection_method == 'auto':
                    # Auto: prefer threshold if previously used, else wavelet
                    use_threshold = hasattr(self, 'threshold_events') and self.threshold_events is not None and len(self.threshold_events) > 0
                elif detection_method == 'threshold':
                    use_threshold = True
                elif detection_method == 'wavelet':
                    use_threshold = False
                else:
                    raise ValueError(f"detection_method must be 'auto', 'wavelet', or 'threshold', got '{detection_method}'")

                if use_threshold:
                    # Full threshold reconstruction to update ASP with new kinetics
                    self.reconstruct_spikes(
                        method='threshold',
                        n_mad=kwargs.get('n_mad', 4.0),
                        min_duration_frames=kwargs.get('min_duration_frames', 3),
                        merge_gap_frames=kwargs.get('merge_gap_frames', 2),
                        iterative=kwargs.get('iterative', False),
                        n_iter=kwargs.get('n_iter', 3),
                        adaptive_thresholds=kwargs.get('adaptive_thresholds', True),
                        create_event_regions=True
                    )
                else:
                    # Wavelet detection (slower but more sensitive)
                    # Calculate optimal event duration based on kinetics
                    min_event_dur = 0.5  # seconds, reasonable minimum
                    max_event_dur = result['t_rise'] + max_event_dur_multiplier * result['t_off']

                    # Re-run spike detection with optimized kinetics
                    # Preserve event regions for wavelet SNR calculation
                    self.reconstruct_spikes(
                        method='wavelet',
                        iterative=kwargs.get('iterative', False),
                        n_iter=kwargs.get('n_iter', 3),
                        adaptive_thresholds=kwargs.get('adaptive_thresholds', True),
                        fps=fps,
                        min_event_dur=min_event_dur,
                        max_event_dur=max_event_dur,
                        event_mask_expansion_sec=kwargs.get('event_mask_expansion_sec', 5.0),
                        create_event_regions=True
                    )
        
        return result

    def _measure_t_off_from_peak(self, signal, peak_idx, fps, max_frames=100,
                                   min_r2=0.8):
        '''Measure t_off by forward exponential decay fitting from peak.

        Parameters
        ----------
        signal : ndarray
            Calcium signal
        peak_idx : int
            Index of peak in signal
        fps : float
            Frames per second
        max_frames : int, optional
            Maximum frames to look forward. Default: 100.
        min_r2 : float, optional
            Minimum R² for fit quality. Events with poor fit (e.g., contaminated
            by close events) are rejected. Default: 0.8.

        Returns
        -------
        float or None
            Estimated t_off in seconds, or None if measurement fails or fit
            quality is below min_r2.
        '''
        decay_start = peak_idx
        decay_end = min(len(signal), peak_idx + max_frames)

        # FPS-adaptive thresholds
        min_decay_frames = int(MIN_DECAY_FRAMES_SEC * fps)
        min_valid_points = int(MIN_VALID_POINTS_SEC * fps)

        if decay_end - decay_start < min_decay_frames:
            return None
        decay_signal = signal[decay_start:decay_end]
        if np.max(decay_signal) <= 0:
            return None
        decay_signal = decay_signal / np.max(decay_signal)
        valid = decay_signal > 0.01
        if np.sum(valid) < min_valid_points:
            return None
        log_y = np.log(decay_signal[valid])
        t = np.arange(len(decay_signal))[valid] / fps
        if len(t) < min_valid_points:
            return None

        # Fit and compute R²
        coeffs = np.polyfit(t, log_y, 1)
        slope, intercept = coeffs

        # Check fit quality - reject contaminated events
        y_pred = slope * t + intercept
        ss_res = np.sum((log_y - y_pred) ** 2)
        ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        if r2 < min_r2:
            return None

        tau = -1 / slope if slope < 0 else None
        if tau is not None:
            if 0.1 < tau < 30:  # Upper bound increased from 10 to handle long decay signals
                return tau
            return None

    
    def _measure_t_rise_derivative(self, signal, peak_idx, fps, max_frames_back=30):
        '''Measure t_rise using maximum derivative method (RECOMMENDED).

        This method uses the relationship: dy/dt_max = A/tau for exponential rise.
        Demonstrated to have 4-33% error vs 36-89% for log-space method.

        Parameters
        ----------
        signal : ndarray
            Calcium signal
        peak_idx : int
            Index of peak in signal
        fps : float
            Frames per second
        max_frames_back : int, optional
            Maximum frames to look backward. Default: 30 (1s at 30fps).

        Returns
        -------
        float or None
            Estimated t_rise in seconds, or None if measurement fails.
        '''
        rise_end = peak_idx + 1
        rise_start = max(0, peak_idx - max_frames_back)

        # FPS-adaptive thresholds
        min_rise_frames = int(MIN_RISE_FRAMES_SEC * fps)
        min_valid_points = int(MIN_VALID_POINTS_SEC * fps)
        savgol_window = max(5, int(SAVGOL_WINDOW_SEC * fps))
        # Savitzky-Golay requires odd window length
        if savgol_window % 2 == 0:
            savgol_window += 1

        if rise_end - rise_start < min_rise_frames:
            return None
        rise_signal = signal[rise_start:rise_end]
        if np.max(rise_signal) <= 0:
            return None
        if len(rise_signal) >= min_valid_points:
            try:
                # Use FPS-adaptive window length (must be odd and <= len(rise_signal))
                actual_window = min(savgol_window, len(rise_signal))
                if actual_window % 2 == 0:
                    actual_window -= 1
                if actual_window >= 5:  # Minimum for polyorder=2
                    smoothed = savgol_filter(rise_signal, actual_window, 2)
                else:
                    smoothed = rise_signal
            except:
                smoothed = rise_signal
            derivative = np.gradient(smoothed) * fps
            max_deriv_idx = np.argmax(derivative)
            max_deriv = derivative[max_deriv_idx]
            peak_val = smoothed[-1]
            baseline = np.percentile(smoothed[:min(min_valid_points, len(smoothed))], 50)
            amplitude = peak_val - baseline
            # Use minimum derivative threshold to avoid division by tiny values
            if max_deriv > 1e-6 and amplitude > 0:
                tau = amplitude / max_deriv
                if 0.01 < tau < 1:
                    return tau
                return None
            return None


    
    def _optimize_kinetics_direct(self, fps=20, asp=None, wvt_ridges=None,
                                  max_frames_forward=100, max_frames_back=30,
                                  min_events=5, aggregation='median', min_r2=0.8):
        '''Internal: Optimize kinetics by direct measurement from detected events.

        This method directly measures t_rise and t_off from calcium signal peaks:
        - t_off: Forward exponential decay fitting (log-space)
        - t_rise: Derivative-based method (tau = amplitude / max_derivative)

        Unlike optimize_kinetics(), this method does NOT use iterative optimization.
        Instead, it measures kinetics directly from each event and aggregates results.

        Advantages:
        -----------
        - **Fast** - Direct measurement from event shapes without iterative optimization
        - **Stable** t_rise estimates with low variance across measurements
        - **No optimization failures** - deterministic measurement
        - **Better for fast indicators** - derivative method shows superior accuracy

        Limitations:
        ------------
        - Requires well-separated peaks (works best with single-pass wavelet detection)
        - Less effective for overlapping events
        - May need larger sample of events for stable estimates

        Parameters
        ----------
        fps : float, optional
            Sampling rate in frames per second. Default: 20.0.
        asp : array-like or TimeSeries, optional
            External amplitude spike data. If provided, uses this instead of self.asp.
            Should be sparse array where asp[i] = amplitude at event positions.
            Default: None (uses self.asp).
        wvt_ridges : list of Ridge or SimpleEvent objects, optional
            External event data (event boundaries). If provided, uses this instead of
            searching for self.threshold_events or self.wvt_ridges. Objects must have
            .start and .end attributes. Used for peak finding.
            Default: None (uses self.threshold_events if available, else self.wvt_ridges).
        max_frames_forward : int, optional
            Maximum frames to look forward for t_off measurement. Default: 100.
        max_frames_back : int, optional
            Maximum frames to look backward for t_rise measurement. Default: 30.
        min_events : int, optional
            Minimum number of successful measurements required. Default: 5.
        aggregation : {\'median\', \'mean\'}, optional
            How to aggregate measurements from multiple events. Default: \'median\' (more robust).
        min_r2 : float, optional
            Minimum R² for t_off fit quality. Events with poor exponential fit
            (e.g., contaminated by close events) are rejected. Default: 0.8.

        Returns
        -------
        dict
            Results dictionary with keys:
            - \'optimized\': bool, True only if BOTH t_rise and t_off were successfully measured
            - \'partially_optimized\': bool, True if exactly one parameter was measured
            - \'used_defaults\': dict, {\'t_rise\': bool, \'t_off\': bool} indicating which params defaulted
            - \'t_rise\': float, measured rise time (seconds), or default if measurement failed
            - \'t_off\': float, measured decay time (seconds), or default if measurement failed
            - \'t_rise_std\': float, standard deviation of t_rise measurements
            - \'t_off_std\': float, standard deviation of t_off measurements
            - \'n_events_used_rise\': int, number of events with successful t_rise measurement
            - \'n_events_used_off\': int, number of events with successful t_off measurement
            - \'n_events_detected\': int, total events detected
            - \'method\': str, always \'direct\'

        Raises
        ------
        ValueError
            If no events detected (call reconstruct_spikes() first).
            If aggregation method not in {\'median\', \'mean\'}.

        Examples
        --------
        >>> # Fast direct measurement (recommended for initial estimates)
        >>> neuron = Neuron(cell_id=0, ca=calcium_trace, sp=None, fps=30)
        >>> neuron.reconstruct_spikes(method=\'wavelet\')
        >>> result = neuron.optimize_kinetics_direct(fps=30)
        >>> print(f"Direct: t_rise={result[\'t_rise\']:.3f}s ± {result[\'t_rise_std\']:.3f}s")
        >>> print(f"Direct: t_off={result[\'t_off\']:.3f}s ± {result[\'t_off_std\']:.3f}s")

        >>> # Use with external events
        >>> result = neuron.optimize_kinetics_direct(fps=30, asp=custom_events)

        Notes
        -----
        - Derivative method for t_rise validated on synthetic data
        - For iterative refinement, combine with reconstruct_spikes(iterative=True)

        See Also
        --------
        optimize_kinetics : Main kinetics optimization interface
        get_kinetics : Cached access to optimization results
        '''
        check_positive(fps=fps, max_frames_forward=max_frames_forward, max_frames_back=max_frames_back, min_events=min_events)
        if aggregation not in ('median', 'mean'):
            raise ValueError(f'''aggregation must be \'median\' or \'mean\', got \'{aggregation}\'''')
        default_t_rise = self.default_t_rise / fps
        default_t_off = self.default_t_off / fps
        calcium_signal = np.asarray(self.ca.data)

        # Check for available event data (wavelet ridges or threshold events)
        if wvt_ridges is not None:
            ridges = wvt_ridges
        elif hasattr(self, 'threshold_events') and self.threshold_events is not None and len(self.threshold_events) > 0:
            ridges = self.threshold_events
        elif self.wvt_ridges is not None and len(self.wvt_ridges) > 0:
            ridges = self.wvt_ridges
        else:
            return {
                'optimized': False,
                'partially_optimized': False,
                'used_defaults': {'t_rise': True, 't_off': True},
                't_rise': default_t_rise,
                't_off': default_t_off,
                't_rise_std': 0,
                't_off_std': 0,
                'n_events_used_rise': 0,
                'n_events_used_off': 0,
                'n_events_detected': 0,
                'method': 'direct',
                'error': 'No events available. Call reconstruct_spikes() or detect_events_threshold() first.' }
        n_events_detected = len(ridges)
        if n_events_detected == 0:
            return {
                'optimized': False,
                'partially_optimized': False,
                'used_defaults': {'t_rise': True, 't_off': True},
                't_rise': default_t_rise,
                't_off': default_t_off,
                't_rise_std': 0,
                't_off_std': 0,
                'n_events_used_rise': 0,
                'n_events_used_off': 0,
                'n_events_detected': 0,
                'method': 'direct',
                'error': 'No events detected' }
        event_positions = []
        for ridge in ridges:
            start_idx = int(ridge.start)
            end_idx = int(ridge.end)
            if end_idx <= start_idx or start_idx < 0 or end_idx >= len(calcium_signal):
                continue
            event_segment = calcium_signal[start_idx:end_idx + 1]
            peak_offset = np.argmax(event_segment)
            peak_idx = start_idx + peak_offset
            event_positions.append(peak_idx)
        event_positions = np.array(event_positions)
        if len(event_positions) == 0:
            return {
                'optimized': False,
                'partially_optimized': False,
                'used_defaults': {'t_rise': True, 't_off': True},
                't_rise': default_t_rise,
                't_off': default_t_off,
                't_rise_std': 0,
                't_off_std': 0,
                'n_events_used_rise': 0,
                'n_events_used_off': 0,
                'n_events_detected': n_events_detected,
                'method': 'direct',
                'error': 'No valid event boundaries found' }
        t_rise_measurements = []
        t_off_measurements = []
        for peak_idx in event_positions:
            t_off = self._measure_t_off_from_peak(calcium_signal, peak_idx, fps, max_frames=max_frames_forward, min_r2=min_r2)
            if t_off is not None:
                t_off_measurements.append(t_off)
            t_rise = self._measure_t_rise_derivative(calcium_signal, peak_idx, fps, max_frames_back=max_frames_back)
            if t_rise is not None:
                t_rise_measurements.append(t_rise)
        if len(t_rise_measurements) < min_events and len(t_off_measurements) < min_events:
            return {
                'optimized': False,
                'partially_optimized': False,
                'used_defaults': {'t_rise': True, 't_off': True},
                't_rise': default_t_rise,
                't_off': default_t_off,
                't_rise_std': 0,
                't_off_std': 0,
                'n_events_used_rise': len(t_rise_measurements),
                'n_events_used_off': len(t_off_measurements),
                'n_events_detected': n_events_detected,
                'method': 'direct',
                'error': 'Insufficient successful measurements' }
        if aggregation == 'median':
            t_rise_final = np.median(t_rise_measurements) if len(t_rise_measurements) >= min_events else default_t_rise
            t_off_final = np.median(t_off_measurements) if len(t_off_measurements) >= min_events else default_t_off
        else:  # aggregation == 'mean'
            t_rise_final = np.mean(t_rise_measurements) if len(t_rise_measurements) >= min_events else default_t_rise
            t_off_final = np.mean(t_off_measurements) if len(t_off_measurements) >= min_events else default_t_off
        t_rise_std = np.std(t_rise_measurements) if len(t_rise_measurements) > 1 else 0
        t_off_std = np.std(t_off_measurements) if len(t_off_measurements) > 1 else 0
        # Track which parameters used defaults due to insufficient measurements
        used_defaults = {
            't_rise': len(t_rise_measurements) < min_events,
            't_off': len(t_off_measurements) < min_events
        }
        # optimized=True only if BOTH parameters were successfully measured
        fully_optimized = not any(used_defaults.values())
        # partially_optimized=True if exactly one parameter was measured
        partially_optimized = any(used_defaults.values()) and not all(used_defaults.values())
        return {
            'optimized': fully_optimized,
            'partially_optimized': partially_optimized,
            'used_defaults': used_defaults,
            't_rise': t_rise_final,
            't_off': t_off_final,
            't_rise_std': t_rise_std,
            't_off_std': t_off_std,
            'n_events_used_rise': len(t_rise_measurements),
            'n_events_used_off': len(t_off_measurements),
            'n_events_detected': n_events_detected,
            'method': 'direct' }

    def detect_events_threshold(self, threshold=None, n_mad=4.0,
                                 min_duration_frames=3, merge_gap_frames=2,
                                 use_scaled=True, signal=None):
        """Detect calcium events using threshold crossings (fast alternative to wavelet).

        This method finds event boundaries by detecting when the calcium signal
        crosses above/below a threshold. Returns SimpleEvent objects compatible
        with optimize_kinetics(), providing ~100-500x speedup vs wavelet detection.

        Parameters
        ----------
        threshold : float, optional
            Absolute threshold value. If None, computed as:
            median(signal) + n_mad * MAD(signal)
            where MAD = median absolute deviation (robust noise estimate).
        n_mad : float, optional
            Number of MAD units above median for auto-threshold.
            Only used if threshold=None. Default: 4.0 (robust detection).
        min_duration_frames : int, optional
            Minimum event duration in frames. Events shorter than this are discarded.
            Default: 3 frames.
        merge_gap_frames : int, optional
            Merge events separated by fewer than this many frames.
            Prevents event fragmentation. Default: 2 frames.
        use_scaled : bool, optional
            If True, use scaled calcium data (self.ca.scdata).
            If False, use raw calcium data (self.ca.data).
            Default: True (recommended for consistent thresholds).
        signal : ndarray, optional
            Custom signal to use for detection. If provided, use_scaled is ignored.
            Used for iterative detection on residual signals.

        Returns
        -------
        list of SimpleEvent
            Detected events with .start and .end attributes (frame indices).
            Empty list if no events detected.

        Raises
        ------
        AttributeError
            If calcium data not available or lacks scdata attribute.
        ValueError
            If parameters are invalid (negative values, percentile out of range).

        Examples
        --------
        >>> # Automatic threshold (recommended)
        >>> neuron = Neuron(cell_id=0, ca=calcium_data, sp=None)
        >>> events = neuron.detect_events_threshold(n_mad=4.0)
        >>> len(events)
        42

        >>> # Then use with optimize_kinetics for fast kinetics estimation
        >>> neuron.threshold_events = events  # Store for optimize_kinetics
        >>> result = neuron.optimize_kinetics(method='direct', fps=20)

        >>> # Manual threshold
        >>> events = neuron.detect_events_threshold(threshold=0.3, use_scaled=True)

        >>> # More sensitive detection (lower threshold)
        >>> events = neuron.detect_events_threshold(n_mad=3.0)

        >>> # Access event properties
        >>> for event in events[:3]:
        ...     print(f"Event: frames {event.start:.0f}-{event.end:.0f}, duration={event.duration:.0f}")

        Notes
        -----
        Performance: O(N) complexity vs O(N²) for wavelet detection
        - Typical speedup: 100-500x faster
        - Example: 1000 frames: ~0.01s (threshold) vs 1-5s (wavelet)

        Algorithm:
        1. Compute threshold (auto or manual)
        2. Find upward crossings (signal goes above threshold) → event starts
        3. Find downward crossings (signal goes below threshold) → event ends
        4. Filter by minimum duration
        5. Merge events with small gaps

        Comparison with wavelet detection:
        - Threshold: Fast, simple, good for high SNR data
        - Wavelet: Slower, more sensitive, better for low SNR or overlapping events

        The detected events are stored in self.threshold_events for later use.

        See Also
        --------
        optimize_kinetics : Use detected events for kinetics estimation
        reconstruct_spikes : Wavelet-based spike detection (slower but more sensitive)
        """
        check_positive(min_duration_frames=min_duration_frames, merge_gap_frames=merge_gap_frames, n_mad=n_mad)

        # Get calcium signal (use custom signal if provided, otherwise from self.ca)
        if signal is not None:
            signal = np.asarray(signal)
        elif use_scaled:
            if not hasattr(self.ca, 'scdata'):
                raise AttributeError(
                    'Scaled calcium data not available. '
                    'Set use_scaled=False to use raw data, or ensure calcium TimeSeries has scdata.'
                )
            signal = np.asarray(self.ca.scdata)
        else:
            signal = np.asarray(self.ca.data)

        # Compute threshold if not provided (robust: median + n_mad * MAD)
        if threshold is None:
            signal_median = np.median(signal)
            signal_mad = median_abs_deviation(signal, scale='normal')
            threshold = signal_median + n_mad * signal_mad

        # Find threshold crossings
        above_threshold = signal > threshold

        # Find transitions: 0→1 = event start, 1→0 = event end
        diff = np.diff(above_threshold.astype(int))
        starts = np.where(diff == 1)[0] + 1  # +1 because diff shifts indices
        ends = np.where(diff == -1)[0] + 1

        # Handle edge cases
        if above_threshold[0]:
            starts = np.concatenate([[0], starts])
        if above_threshold[-1]:
            ends = np.concatenate([ends, [len(signal)]])

        # Ensure equal number of starts and ends
        n_events = min(len(starts), len(ends))
        starts = starts[:n_events]
        ends = ends[:n_events]

        # Filter by minimum duration
        durations = ends - starts
        valid = durations >= min_duration_frames
        starts = starts[valid]
        ends = ends[valid]

        # Merge events with small gaps
        if len(starts) > 1:
            merged_starts = [starts[0]]
            merged_ends = []

            for i in range(1, len(starts)):
                gap = starts[i] - ends[i-1]
                if gap <= merge_gap_frames:
                    # Merge with previous event (extend end)
                    continue
                else:
                    # Close previous event and start new one
                    merged_ends.append(ends[i-1])
                    merged_starts.append(starts[i])

            # Close last event
            merged_ends.append(ends[-1])

            starts = np.array(merged_starts)
            ends = np.array(merged_ends)

        # Create SimpleEvent objects
        events = [SimpleEvent(start=s, end=e) for s, e in zip(starts, ends)]

        # Store for later use
        self.threshold_events = events

        return events


    def _calculate_event_r2(calcium_signal, reconstruction, n_mad=4, event_mask=None, wvt_ridges=None, fps=None):
        '''Calculate R² on event regions.

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
        '''
        # Construct event mask if not provided
        if event_mask is None:
            if wvt_ridges is not None and len(wvt_ridges) > 0:
                if fps is None:
                    raise ValueError('fps required to construct event mask from wvt_ridges')
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
                    'Either event_mask or wvt_ridges must be provided for event R² calculation. '
                    'Run reconstruct_spikes() with create_event_regions=True to populate neuron.events.'
                )

        event_mask = event_mask > 0
        if np.sum(event_mask) == 0:
            return np.nan
        ca_events = calcium_signal[event_mask]
        recon_events = reconstruction[event_mask]
        residuals = ca_events - recon_events
        ss_residual = np.sum(residuals ** 2)
        ss_total = np.sum((ca_events - np.mean(ca_events)) ** 2)
        if ss_total == 0:
            return np.nan
        return 1 - ss_residual / ss_total

    _calculate_event_r2 = staticmethod(_calculate_event_r2)
