import numpy as np
import logging
from scipy.stats import median_abs_deviation
from scipy.optimize import minimize
from ..information.info_base import TimeSeries
from ..utils.data import check_positive, check_nonnegative
from ..utils.jit import conditional_njit
# Import specific functions instead of using *
from .wavelet_event_detection import (
    extract_wvt_events, 
    events_to_ts_array
)

DEFAULT_T_RISE = 0.25  # sec
DEFAULT_T_OFF = 2.0  # sec

DEFAULT_FPS = 20.0  # frames per sec
DEFAULT_MIN_BEHAVIOUR_TIME = 0.25  # sec

MIN_CA_SHIFT = 5  # MIN_SHIFT*t_off is the minimal random signal shift for a given cell


class Neuron:
    """
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
        Rise time constant in seconds
    t_off : float
        Decay time constant in seconds (may be fitted if requested)
    fps : float
        Sampling rate in frames per second
    ca_ts : TimeSeries
        Preprocessed calcium time series
    sp_ts : TimeSeries or None
        Spike train time series (if spikes provided)
        
    Notes
    -----
    The class assumes spike data is binary (0 or 1 values). Non-binary 
    spike data may produce incorrect results in spike counting.    """

    @staticmethod
    def spike_form(t, t_rise, t_off):
        """Calculate normalized calcium response kernel shape.
        
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
        normalized to have maximum value of 1.        """
        # Input validation
        check_positive(t_rise=t_rise, t_off=t_off)
            
        # Use inner JIT-compiled function for computation
        return Neuron._spike_form_jit(t, t_rise, t_off)
    
    @staticmethod
    @conditional_njit
    def _spike_form_jit(t, t_rise, t_off):
        """JIT-compiled core computation for spike_form.
        
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
        JIT compilation provides ~10-100x speedup for large arrays.        """
        form = (1 - np.exp(-t / t_rise)) * np.exp(-t / t_off)
        max_val = np.max(form)
        if max_val == 0:
            raise ValueError("Kernel form has zero maximum")
        return form / max_val

    @staticmethod
    def get_restored_calcium(sp, t_rise, t_off):
        """Reconstruct calcium signal from spike train.
        
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
        Uses a kernel of length 1000 frames, which should be sufficient
        for most calcium indicators (5x decay time for t_off ≤ 200 frames).

        The convolution naturally handles amplitude-weighted spikes, where
        each spike value represents event strength in dF/F0 units.        """
        # Input validation
        sp = np.asarray(sp)
        if sp.size == 0:
            raise ValueError("Spike train cannot be empty")
        check_positive(t_rise=t_rise, t_off=t_off)
            
        x = np.linspace(0, 1000, num=1000)
        spform = Neuron.spike_form(x, t_rise=t_rise, t_off=t_off)
        conv = np.convolve(sp, spform)
        # Return same length as input
        return conv[:len(sp)]

    @staticmethod
    def ca_mse_error(t_off, ca, spk, t_rise):
        """Calculate RMSE between observed calcium and reconstructed from spikes.
        
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
        where t_off is the parameter being optimized.        """
        # Input validation
        ca = np.asarray(ca)
        spk = np.asarray(spk)
        if len(ca) != len(spk):
            raise ValueError(f"ca and spk must have same length: {len(ca)} vs {len(spk)}")
        check_positive(t_rise=t_rise, t_off=t_off)
            
        re_ca = Neuron.get_restored_calcium(spk, t_rise, t_off)
        # No need for np.abs() since we're squaring
        return np.sqrt(np.sum((ca - re_ca) ** 2) / len(ca))

    @staticmethod
    def calcium_preprocessing(ca, seed=None):
        """Preprocess calcium signal for spike reconstruction.
        
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
        numerical issues in downstream spike reconstruction algorithms.        """
        # Input validation
        ca = np.asarray(ca)
        if ca.size == 0:
            raise ValueError("Calcium signal cannot be empty")
            
        # Set seed for reproducibility if provided
        if seed is not None:
            np.random.seed(seed)
            
        # Use inner JIT-compiled function for computation
        return Neuron._calcium_preprocessing_jit(ca)
    
    @staticmethod
    @conditional_njit
    def _calcium_preprocessing_jit(ca):
        """JIT-compiled core computation for calcium_preprocessing.
        
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
        externally for reproducibility.        """
        # Ensure we're working with float array to avoid dtype casting issues
        ca = ca.astype(np.float64)
        ca[ca < 0] = 0  # More efficient than np.where
        ca += np.random.random(len(ca)) * 1e-8
        return ca

    @staticmethod
    def extract_event_amplitudes(ca_signal, st_ev_inds, end_ev_inds, baseline_window=20,
                                 already_dff=False, baseline_offset=0,
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
            Useful for GCaMP indicators with non-zero rise time (~5 frames for GCaMP6f @ 20Hz).
            Default is 0 (baseline immediately before event).
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
            if t_rise_frames is None or t_off_frames is None or fps is None:
                raise ValueError("t_rise_frames, t_off_frames, and fps required when use_peak_refinement=True")

        ca_signal = np.asarray(ca_signal)
        amplitudes = []
        peak_search_frames = int(peak_search_window_sec * fps) if use_peak_refinement else 0

        for start, end in zip(st_ev_inds, end_ev_inds):
            # Extract event segment
            event_segment = ca_signal[start:end]

            # Skip if empty event segment
            if len(event_segment) == 0:
                amplitudes.append(0.0)
                continue

            # Determine peak location and value
            if use_peak_refinement:
                # Find TRUE peak in expanded window (same logic as amplitudes_to_point_events)
                search_start = max(0, start - peak_search_frames)
                search_end = min(len(ca_signal), end + peak_search_frames)
                search_segment = ca_signal[search_start:search_end]
                true_peak_offset = np.argmax(search_segment)
                true_peak_idx = search_start + true_peak_offset

                # Use refined peak for baseline calculation
                reference_idx = true_peak_idx
                peak_value = ca_signal[true_peak_idx]
            else:
                # Use wavelet detection window
                reference_idx = start
                peak_value = np.max(event_segment)

            if already_dff:
                # Signal is already dF/F - extract peak relative to local baseline
                # Use offset window to avoid sampling during rise phase
                # Calculate baseline relative to reference_idx (refined peak or wavelet start)
                baseline_end = max(0, reference_idx - baseline_offset)
                baseline_start = max(0, baseline_end - baseline_window)
                baseline_segment = ca_signal[baseline_start:baseline_end]

                if len(baseline_segment) > 0:
                    # Local baseline = median of preceding frames (before rise phase)
                    local_baseline = np.median(baseline_segment)
                else:
                    # No baseline available - use 0 (e.g., first event in signal)
                    local_baseline = 0.0

                # Amplitude = peak value minus local baseline
                amplitude = peak_value - local_baseline
            else:
                # Apply dF/F0 normalization for raw fluorescence
                # Extract baseline from preceding window relative to reference_idx
                baseline_end = max(0, reference_idx - baseline_offset)
                baseline_start = max(0, baseline_end - baseline_window)
                baseline_segment = ca_signal[baseline_start:baseline_end]

                # Skip if no baseline available
                if len(baseline_segment) == 0:
                    amplitudes.append(0.0)
                    continue

                # F0 = median of baseline (robust to outliers)
                F0 = np.median(baseline_segment)

                # Skip if invalid F0
                if F0 <= 0:
                    amplitudes.append(0.0)
                    continue

                # Calculate dF/F0 for peak value
                amplitude = (peak_value - F0) / F0

            # Store only positive amplitudes (events should increase fluorescence)
            amplitudes.append(max(0.0, amplitude))

        return amplitudes

    @staticmethod
    def deconvolve_given_event_times(ca_signal, event_times, t_rise_frames, t_off_frames):
        """Extract amplitudes via non-negative least squares deconvolution.

        Given detected event times and known calcium kernel parameters, finds
        the optimal amplitudes that best reconstruct the observed signal.
        This solves: signal ≈ Σᵢ aᵢ · kernel(t - tᵢ)

        Handles overlapping events naturally by jointly optimizing all amplitudes
        to minimize reconstruction error across the entire signal.

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

        Returns
        -------
        amplitudes : ndarray
            Optimal amplitudes for each event (non-negative).

        Notes
        -----
        Uses scipy.optimize.nnls for non-negative least squares solution.
        Complexity: O(n_events² · n_frames) - suitable for offline analysis.

        References
        ----------
        Lawson CL, Hanson RJ (1995). Solving Least Squares Problems.
        SIAM, Philadelphia.
        """
        from scipy.optimize import nnls

        ca_signal = np.asarray(ca_signal)
        event_times = np.asarray(event_times, dtype=int)

        n_frames = len(ca_signal)
        n_events = len(event_times)

        if n_events == 0:
            return np.array([])

        # Build design matrix K where K[t, i] = kernel value at time t for event i
        K = np.zeros((n_frames, n_events))

        for i, event_time_idx in enumerate(event_times):
            if event_time_idx < 0 or event_time_idx >= n_frames:
                continue

            # Generate calcium kernel starting at this event time
            remaining_frames = n_frames - event_time_idx
            t_array = np.arange(remaining_frames)

            # Double exponential kernel: (1 - exp(-t/τ_rise)) * exp(-t/τ_decay)
            kernel = (1 - np.exp(-t_array / t_rise_frames)) * np.exp(-t_array / t_off_frames)

            # Normalize kernel to peak = 1
            kernel_max = np.max(kernel)
            if kernel_max > 0:
                kernel = kernel / kernel_max

            # Place kernel in design matrix
            K[event_time_idx:, i] = kernel

        # Solve: signal ≈ K @ amplitudes (with non-negativity constraint)
        amplitudes, residual_norm = nnls(K, ca_signal)

        return amplitudes

    @staticmethod
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
        placement : {'start', 'peak', 'onset'}, optional
            Where to place amplitude:
            - 'start': at event start index (as detected by wavelet)
            - 'peak': at actual calcium peak within event window
            - 'onset': at estimated spike onset (peak - kernel_peak_offset)
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

        The 'onset' placement refines the peak location within a search window
        to compensate for wavelet detection inconsistencies, then calculates
        the kernel peak offset to place spikes accurately.
        """
        check_positive(length=length)

        if not isinstance(placement, str) or placement not in ['start', 'peak', 'onset']:
            raise ValueError(f"placement must be 'start', 'peak', or 'onset', got {placement}")

        if placement == 'onset':
            if t_rise_frames is None or t_off_frames is None:
                raise ValueError("Both t_rise_frames and t_off_frames required for 'onset' placement")
            if fps is None:
                raise ValueError("fps required for 'onset' placement")

        ca_signal = np.asarray(ca_signal)
        point_events = np.zeros(length, dtype=float)

        # Calculate search window in frames for peak refinement
        peak_search_frames = int(peak_search_window_sec * fps) if fps is not None else 4

        for start, end, amplitude in zip(st_ev_inds, end_ev_inds, amplitudes):
            if amplitude <= 0:
                continue

            if placement == 'start':
                idx = int(start)
            elif placement == 'peak':
                # Find actual peak within event window
                event_segment = ca_signal[start:end]
                peak_offset = np.argmax(event_segment)
                idx = int(start + peak_offset)
            elif placement == 'onset':
                # Calculate theoretical kernel peak offset
                # For double exponential: (1-exp(-t/τ_r))*exp(-t/τ_d)
                # Peak at: t_peak = τ_rise * τ_off * ln(τ_off/τ_rise) / (τ_off - τ_rise)
                kernel_peak_offset = (t_rise_frames * t_off_frames *
                                     np.log(t_off_frames / t_rise_frames) /
                                     (t_off_frames - t_rise_frames))

                # Wavelet detection is inconsistent - don't trust the peak it found
                # Instead, search for TRUE peak in expanded window around wavelet event
                # Expand the wavelet window [start:end] by ±peak_search_frames
                search_start = max(0, start - peak_search_frames)
                search_end = min(length, end + peak_search_frames)
                search_segment = ca_signal[search_start:search_end]

                # Find the absolute maximum in this expanded window
                # This is the TRUE peak, not the wavelet's approximate peak
                true_peak_offset = np.argmax(search_segment)
                true_peak_idx = search_start + true_peak_offset

                # Place spike before the true peak by kernel offset
                idx = int(true_peak_idx - kernel_peak_offset)

            # Ensure index is valid
            if 0 <= idx < length:
                point_events[idx] += amplitude  # Sum if collision

        return point_events

    def __init__(
        self,
        cell_id,
        ca,
        sp,
        default_t_rise=DEFAULT_T_RISE,
        default_t_off=DEFAULT_T_OFF,
        fps=DEFAULT_FPS,
        fit_individual_t_off=False,
        seed=None,
    ):
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
            If True, fit individual decay time. Default is False.
        seed : int, optional
            Random seed for preprocessing reproducibility.
            
        Raises
        ------
        ValueError
            If ca is empty, fps is not positive, or array lengths don't match.
        TypeError
            If ca cannot be converted to numeric array.
            
        Notes
        -----
        The shuffle mask excludes MIN_CA_SHIFT * t_off frames from each end
        to prevent artifacts in temporal shuffling analyses.        """
        # Input validation
        ca = np.asarray(ca)
        if ca.size == 0:
            raise ValueError("Calcium signal cannot be empty")
        check_positive(fps=fps, default_t_rise=default_t_rise, default_t_off=default_t_off)
        
        self.cell_id = cell_id
        self.ca = TimeSeries(Neuron.calcium_preprocessing(ca, seed=seed), discrete=False)
        
        if sp is None:
            self.sp = None
        else:
            sp = np.asarray(sp)
            if len(sp) != len(ca):
                raise ValueError(f"Spike train length {len(sp)} must match calcium length {len(ca)}")
            self.sp = TimeSeries(sp.astype(int), discrete=True)
            
        self.n_frames = len(self.ca.data)

        # Count spikes efficiently
        self.sp_count = np.sum(self.sp.data) if self.sp is not None else 0

        # Initialize spike representation attributes
        self.events = None  # Binary event regions (duration)
        self.asp = None     # Amplitude point spikes (continuous)

        # Initialize attributes
        self.t_off = None
        self.noise_ampl = None
        self.mad = None
        self.snr = None

        # Quality metrics (cached)
        self.reconstruction_r2 = None
        self.snr_reconstruction = None
        self.mae = None

        # Use defaults if not provided
        if fps is None:
            fps = DEFAULT_FPS
        if default_t_rise is None:
            default_t_rise = DEFAULT_T_RISE
        if default_t_off is None:
            default_t_off = DEFAULT_T_OFF
            
        # Convert time constants to frames
        self.default_t_off = default_t_off * fps
        self.default_t_rise = default_t_rise * fps

        if fit_individual_t_off:
            t_off = self.get_t_off()
        else:
            t_off = self.default_t_off

        # Add shuffle mask according to computed characteristic calcium decay time
        self.ca.shuffle_mask = np.ones(self.n_frames, dtype=bool)
        min_shift = int(t_off * MIN_CA_SHIFT)
        self.ca.shuffle_mask[:min_shift] = False
        self.ca.shuffle_mask[self.n_frames - min_shift :] = False

    def reconstruct_spikes(self, method="wavelet", iterative=True, n_iter=3,
                          min_events_threshold=2, adaptive_thresholds=False,
                          amplitude_method="deconvolution", show_progress=False, **kwargs):
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
        ndarray
            Binary event regions with shape (n_frames,). Values are 0 or 1.
            Also populates self.events and self.asp attributes.

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
        if method == "wavelet":
            # Get parameters with defaults
            fps = kwargs.get("fps", DEFAULT_FPS)
            min_event_dur = kwargs.get("min_event_dur", 0.5)
            max_event_dur = kwargs.get("max_event_dur", 2.5)
            
            # Validate parameters
            check_positive(fps=fps, min_event_dur=min_event_dur, max_event_dur=max_event_dur)
            
            if min_event_dur >= max_event_dur:
                raise ValueError(f"min_event_dur ({min_event_dur}) must be less than max_event_dur ({max_event_dur})")
            
            # Use wavelet event detection with improved defaults
            wvt_kwargs = {
                "fps": fps,
                "min_event_dur": min_event_dur,
                "max_event_dur": max_event_dur,
                "scale_length_thr": kwargs.get("scale_length_thr", 15),  # Improved from default 40
                "max_scale_thr": kwargs.get("max_scale_thr", 6),  # Improved from default 7
                "max_ampl_thr": kwargs.get("max_ampl_thr", 0.04),  # Improved from default 0.05
                "sigma": kwargs.get("sigma", 8),  # Gaussian smoothing
            }

            # Extract calcium data as 2D array (features x time)
            ca_data = self.ca.data.reshape(1, -1)

            if iterative:
                # Iterative detection with residual analysis
                check_positive(n_iter=n_iter, min_events_threshold=min_events_threshold)

                all_asp_arrays = []
                all_st_inds_list = []
                all_end_inds_list = []
                current_signal = self.ca.data.copy()

                # Setup thresholds for each iteration
                if adaptive_thresholds:
                    # Progressive relaxation
                    base_min_dur = min_event_dur
                    base_max_dur = max_event_dur
                    iter_kwargs = []
                    for i in range(n_iter):
                        relax_factor = 1.0 - (i * 0.2)  # 100%, 80%, 60%
                        iter_kwargs.append({
                            "fps": fps,
                            "min_event_dur": max(base_min_dur * relax_factor, 0.1),
                            "max_event_dur": base_max_dur,
                        })
                else:
                    iter_kwargs = [wvt_kwargs.copy() for _ in range(n_iter)]

                # Iterative detection loop
                for iter_idx in range(n_iter):
                    # Detect events in current signal
                    current_signal_2d = current_signal.reshape(1, -1)
                    st_ev_inds, end_ev_inds, _ = extract_wvt_events(
                        current_signal_2d, iter_kwargs[iter_idx], show_progress=show_progress
                    )

                    # Extract indices for single neuron
                    st_inds = st_ev_inds[0] if len(st_ev_inds) > 0 else []
                    end_inds = end_ev_inds[0] if len(end_ev_inds) > 0 else []

                    # Stop if too few events
                    if len(st_inds) < min_events_threshold:
                        break

                    all_st_inds_list.extend(st_inds)
                    all_end_inds_list.extend(end_inds)

                    # Extract amplitudes
                    if amplitude_method == "deconvolution":
                        # For deconvolution: use event centers, refine amplitudes later
                        event_centers = [(s + e) // 2 for s, e in zip(st_inds, end_inds)]
                        # Temporary amplitudes for residual calculation (will be refined)
                        amplitudes = Neuron.extract_event_amplitudes(
                            current_signal, st_inds, end_inds,
                            baseline_window=20, already_dff=True, baseline_offset=10,
                            use_peak_refinement=True,
                            t_rise_frames=self.default_t_rise,
                            t_off_frames=self.default_t_off,
                            fps=fps, peak_search_window_sec=0.2
                        )
                    else:
                        # Peak-based amplitude extraction (default)
                        amplitudes = Neuron.extract_event_amplitudes(
                            current_signal, st_inds, end_inds,
                            baseline_window=20, already_dff=True, baseline_offset=10,
                            use_peak_refinement=True,
                            t_rise_frames=self.default_t_rise,
                            t_off_frames=self.default_t_off,
                            fps=fps, peak_search_window_sec=0.2
                        )

                    # Create ASP array for this iteration
                    asp_array = Neuron.amplitudes_to_point_events(
                        self.n_frames, current_signal, st_inds, end_inds, amplitudes,
                        placement='onset',
                        t_rise_frames=self.default_t_rise,
                        t_off_frames=self.default_t_off,
                        fps=fps, peak_search_window_sec=0.2
                    )
                    all_asp_arrays.append(asp_array)

                    # Reconstruct and compute residual
                    reconstruction = Neuron.get_restored_calcium(asp_array, self.default_t_rise, self.default_t_off)
                    current_signal = current_signal - reconstruction

                # Refine amplitudes with deconvolution if requested
                if amplitude_method == "deconvolution" and len(all_asp_arrays) > 0:
                    # Collect all event onset times from all iterations
                    all_event_times = []
                    original_signal = self.ca.data.copy()

                    # Find onset times from ASP arrays (where amplitudes are placed)
                    for asp_array in all_asp_arrays:
                        event_indices = np.where(asp_array > 0)[0]
                        all_event_times.extend(event_indices)

                    # Deconvolve on original signal with all detected event times
                    if len(all_event_times) > 0:
                        optimal_amplitudes = Neuron.deconvolve_given_event_times(
                            original_signal,
                            all_event_times,
                            self.default_t_rise,
                            self.default_t_off
                        )

                        # Rebuild ASP with optimal amplitudes
                        combined_asp = np.zeros(self.n_frames)
                        for event_time, amplitude in zip(all_event_times, optimal_amplitudes):
                            if 0 <= event_time < self.n_frames:
                                combined_asp[event_time] = amplitude

                        # Update ASP arrays
                        all_asp_arrays = [combined_asp]

                # Combine all ASP arrays
                if len(all_asp_arrays) > 0:
                    combined_asp = np.sum(all_asp_arrays, axis=0)
                    self.asp = TimeSeries(combined_asp, discrete=False)

                    # Create binary spikes
                    sp_array = (combined_asp > 0).astype(int)
                    self.sp = TimeSeries(sp_array, discrete=True)
                    self.sp_count = np.sum(sp_array)

                    # Create binary event regions from all detected events
                    st_ev_inds = [all_st_inds_list]
                    end_ev_inds = [all_end_inds_list]
                else:
                    # No events detected
                    self.asp = TimeSeries(np.zeros(self.n_frames), discrete=False)
                    self.sp = TimeSeries(np.zeros(self.n_frames, dtype=int), discrete=True)
                    self.sp_count = 0
                    st_ev_inds = [[]]
                    end_ev_inds = [[]]

                # Create binary event regions
                events = events_to_ts_array(self.n_frames, st_ev_inds, end_ev_inds, fps)
                self.events = TimeSeries(events.flatten(), discrete=True)
                return events.flatten()

            # Single-pass detection (original logic)
            # Extract events
            st_ev_inds, end_ev_inds, all_ridges = extract_wvt_events(
                ca_data, wvt_kwargs, show_progress=show_progress
            )

            # Create binary event regions (rectangular pulses with duration)
            events = events_to_ts_array(
                self.n_frames, st_ev_inds, end_ev_inds, fps
            )
            self.events = TimeSeries(events.flatten(), discrete=True)

            # Extract event indices for single neuron
            st_inds = st_ev_inds[0] if len(st_ev_inds) > 0 else []
            end_inds = end_ev_inds[0] if len(end_ev_inds) > 0 else []

            # Create spike representations (point-based, no duration)
            if len(st_inds) > 0:
                if amplitude_method == "deconvolution":
                    # Use 'onset' placement to get event times first
                    dummy_amplitudes = np.ones(len(st_inds))
                    asp_positions = Neuron.amplitudes_to_point_events(
                        self.n_frames,
                        self.ca.data,
                        st_inds,
                        end_inds,
                        dummy_amplitudes,
                        placement='onset',
                        t_rise_frames=self.default_t_rise,
                        t_off_frames=self.default_t_off,
                        fps=fps,
                        peak_search_window_sec=0.2
                    )

                    # Get event times from positions
                    event_times = np.where(asp_positions > 0)[0]

                    # Deconvolve to get optimal amplitudes
                    amplitudes = Neuron.deconvolve_given_event_times(
                        self.ca.data,
                        event_times,
                        self.default_t_rise,
                        self.default_t_off
                    )

                    # Create ASP with deconvolved amplitudes
                    asp_array = np.zeros(self.n_frames)
                    for event_time, amplitude in zip(event_times, amplitudes):
                        if 0 <= event_time < self.n_frames:
                            asp_array[event_time] = amplitude
                else:
                    # Peak-based amplitude extraction (default, backward compatible)
                    amplitudes = Neuron.extract_event_amplitudes(
                        self.ca.data, st_inds, end_inds,
                        baseline_window=20, already_dff=True, baseline_offset=10,
                        use_peak_refinement=True,
                        t_rise_frames=self.default_t_rise,
                        t_off_frames=self.default_t_off,
                        fps=fps, peak_search_window_sec=0.2
                    )

                    # Create amplitude point spikes (continuous, with dF/F amplitudes)
                    # Use 'onset' placement - refines peak location then compensates for kernel delay
                    asp_array = Neuron.amplitudes_to_point_events(
                        self.n_frames,
                        self.ca.data,
                        st_inds,
                        end_inds,
                        amplitudes,
                        placement='onset',  # Refine peak, then place at onset
                        t_rise_frames=self.default_t_rise,
                        t_off_frames=self.default_t_off,
                        fps=fps,
                        peak_search_window_sec=0.2  # ±0.2s search window for peak refinement
                    )
                self.asp = TimeSeries(asp_array, discrete=False)  # Continuous!

                # Create binary point spikes (simply binarize asp)
                sp_array = (asp_array > 0).astype(int)
                self.sp = TimeSeries(sp_array, discrete=True)
                self.sp_count = np.sum(sp_array)
            else:
                # No events detected
                self.asp = TimeSeries(np.zeros(self.n_frames), discrete=False)
                self.sp = TimeSeries(np.zeros(self.n_frames, dtype=int), discrete=True)
                self.sp_count = 0

            return events.flatten()
            
        elif method == "threshold":
            # Import threshold reconstruction
            from .spike_reconstruction import threshold_reconstruction
            
            # Get parameters
            threshold_std = kwargs.get("threshold_std", 2.5)
            smooth_sigma = kwargs.get("smooth_sigma", 2.0)
            min_spike_interval = kwargs.get("min_spike_interval", 0.1)
            fps = kwargs.get("fps", DEFAULT_FPS)
            
            # Validate parameters
            check_positive(threshold_std=threshold_std, smooth_sigma=smooth_sigma, 
                         min_spike_interval=min_spike_interval, fps=fps)
            
            # Apply threshold reconstruction
            spikes = threshold_reconstruction(
                self.ca.data,
                threshold_std=threshold_std,
                smooth_sigma=smooth_sigma,
                min_spike_interval=min_spike_interval,
                fps=fps
            )
            
            return spikes
            
        else:
            raise NotImplementedError(f"Method '{method}' not implemented. Available methods: 'wavelet', 'threshold'")

    def get_mad(self):
        """Get median absolute deviation of calcium signal.
        
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
        contains spike-related transients.        """
        if self.mad is None:
            try:
                self.snr, self.mad = self._calc_snr()
            except ValueError:
                self.mad = median_abs_deviation(self.ca.data)
        return self.mad

    def get_snr(self):
        """Get signal-to-noise ratio of calcium signal.
        
        Computes SNR as the ratio of mean calcium amplitude during spikes
        to the median absolute deviation (noise level). Caches the result
        for efficiency.
        
        Returns
        -------
        float
            Signal-to-noise ratio.
            
        Raises
        ------
        ValueError
            If no spikes are present or SNR calculation fails.
            
        Notes
        -----
        Requires spike data to be available. The SNR provides a measure
        of calcium signal quality relative to baseline noise.        """
        if self.snr is None:
            self.snr, self.mad = self._calc_snr()
        return self.snr

    def _calc_snr(self):
        """Calculate signal-to-noise ratio and MAD.
        
        Internal method that computes both SNR and MAD in a single pass
        for efficiency. SNR is calculated as the mean calcium amplitude
        at spike times divided by the median absolute deviation.
        
        Returns
        -------
        tuple
            (snr, mad) where snr is signal-to-noise ratio and mad is
            median absolute deviation.
            
        Raises
        ------
        ValueError
            If no spikes are present, if MAD is zero, or if SNR 
            calculation results in NaN.        """
        if self.sp is None:
            raise ValueError("No spike data available")
            
        spk_inds = np.nonzero(self.sp.data)[0]
        mad = median_abs_deviation(self.ca.data)
        
        if len(spk_inds) == 0:
            raise ValueError("No spikes found!")
            
        if mad == 0:
            raise ValueError("MAD is zero, cannot compute SNR")
            
        sn = np.mean(self.ca.data[spk_inds]) / mad
        
        if np.isnan(sn):
            raise ValueError("Error in SNR calculation")

        return sn, mad

    def get_reconstruction_r2(self):
        """Get R² for calcium reconstruction quality.

        Computes the coefficient of determination (R²) measuring how well
        the double-exponential model fits the observed calcium signal.
        Higher values indicate better model fit. Caches the result.

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
            If amplitude spike data is not available.

        Notes
        -----
        R² = 1 - (SS_residual / SS_total) where:
        - SS_residual = sum of squared residuals from reconstruction
        - SS_total = total variance in calcium signal

        Uses cached RMSE from get_noise_ampl() for efficiency.
        """
        if self.reconstruction_r2 is None:
            if self.asp is None:
                raise ValueError(
                    "Amplitude spikes required for reconstruction R². "
                    "Call reconstruct_spikes() first."
                )

            # Get RMSE (cached after first call)
            rmse = self.get_noise_ampl()

            # Calculate R²: 1 - (SS_residual / SS_total)
            ss_residual = (rmse ** 2) * self.n_frames
            ss_total = np.sum((self.ca.data - np.mean(self.ca.data)) ** 2)

            if ss_total == 0:
                raise ValueError("Total variance is zero, cannot compute R²")

            self.reconstruction_r2 = 1 - (ss_residual / ss_total)

        return self.reconstruction_r2

    def get_snr_reconstruction(self):
        """Get reconstruction-based SNR.

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
        """
        if self.snr_reconstruction is None:
            rmse = self.get_noise_ampl()
            signal_std = np.std(self.ca.data)

            if rmse == 0:
                raise ValueError("RMSE is zero, cannot compute reconstruction SNR")

            self.snr_reconstruction = signal_std / rmse

        return self.snr_reconstruction

    def get_mae(self):
        """Get Mean Absolute Error between observed and reconstructed calcium.

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
        """
        if self.mae is None:
            if self.asp is None:
                raise ValueError(
                    "Amplitude spikes required for MAE calculation. "
                    "Call reconstruct_spikes() first."
                )

            # Reconstruct calcium from spikes
            t_off = self.get_t_off()
            ca_fitted = Neuron.get_restored_calcium(
                self.asp.data, self.default_t_rise, t_off
            )

            # Compute Mean Absolute Error
            self.mae = np.mean(np.abs(self.ca.data - ca_fitted))

        return self.mae

    def get_t_off(self):
        """Get calcium decay time constant.
        
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
        that minimizes the RMSE between observed and reconstructed calcium.        """
        if self.t_off is None:
            self.t_off, self.noise_ampl = self._fit_t_off()

        return self.t_off

    def get_noise_ampl(self):
        """Get noise amplitude estimate from calcium-spike reconstruction.
        
        Returns the root mean square error (RMSE) between observed calcium
        and optimally reconstructed calcium from spikes. This provides an
        estimate of the noise level in the calcium signal after accounting
        for spike-related transients.
        
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
        This is different from MAD, as it specifically measures the
        residual error after optimal spike-to-calcium reconstruction.
        The value is cached after first computation.        """
        if self.noise_ampl is None:
            self.t_off, self.noise_ampl = self._fit_t_off()

        return self.noise_ampl

    def _fit_t_off(self):
        """Fit optimal calcium decay time constant from spike-calcium pairs.
        
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
        quality issues.        """
        # FUTURE: fit for an arbitrary kernel form.
        # FUTURE: add nonlinear summation fit if needed
        
        # Use amplitude spikes if available, otherwise fall back to binary
        if self.asp is not None and np.sum(np.abs(self.asp.data)) > 0:
            spike_data = self.asp.data
        elif self.sp is not None:
            spike_data = self.sp.data
        else:
            raise ValueError("Spike data required for t_off fitting")

        res = minimize(
            Neuron.ca_mse_error,
            (np.array([self.default_t_off])),
            args=(self.ca.data, spike_data, self.default_t_rise),
            bounds=[(self.default_t_rise * 1.1, None)],  # t_off must be > t_rise
        )
        opt_t_off = res.x[0]
        noise_amplitude = res.fun

        logger = logging.getLogger(__name__)
        
        # Check for invalid (negative or near-zero) values
        if opt_t_off <= self.default_t_rise:
            logger.warning(
                f"Optimization failed for neuron {self.cell_id}: fitted t_off ({opt_t_off:.2f}) "
                f"<= t_rise ({self.default_t_rise:.2f}). Using default t_off={self.default_t_off}"
            )
            opt_t_off = self.default_t_off
        elif opt_t_off > self.default_t_off * 5:
            logger.warning(
                f"Calculated t_off={int(opt_t_off)} for neuron {self.cell_id} is suspiciously high, "
                f"check signal quality. t_off has been automatically lowered to {self.default_t_off*5}"
            )
            opt_t_off = self.default_t_off * 5
            
        # Additional check for optimization failure
        if not res.success:
            logger.warning(
                f"Optimization did not converge for neuron {self.cell_id}: {res.message}. "
                f"Using fitted value {opt_t_off:.2f} with caution."
            )

        return opt_t_off, noise_amplitude

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
        valid_methods = ['roll_based', 'waveform_based', 'chunks_based']
        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Must be one of {valid_methods}")
            
        fn = getattr(self, f"_shuffle_calcium_data_{method}")

        sh_ca = fn(seed=seed, **kwargs)
        sh_ca = Neuron.calcium_preprocessing(sh_ca, seed=seed)
        if not return_array:
            sh_ca = TimeSeries(sh_ca, discrete=False)

        return sh_ca

    def _shuffle_calcium_data_waveform_based(self, seed=None, **kwargs):
        """Shuffle calcium by reconstructing from ISI-shuffled spikes.
        
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
            If spike data is not available.        """
        # Use amplitude spikes if available, otherwise fall back to binary
        if self.asp is not None and np.sum(np.abs(self.asp.data)) > 0:
            spike_data = self.asp.data
        elif self.sp is not None:
            spike_data = self.sp.data
        else:
            raise ValueError("Spike data required for waveform-based shuffling")

        opt_t_off = self.get_t_off()

        # Reconstruct calcium from original spikes
        conv = Neuron.get_restored_calcium(spike_data, self.default_t_rise, opt_t_off)
        background = self.ca.data - conv[: len(self.ca.data)]

        # Shuffle spikes preserving ISI statistics
        pspk = self._shuffle_spikes_data_isi_based(seed=seed)

        # If using amplitude spikes, apply amplitudes to shuffled positions
        if self.asp is not None and np.sum(np.abs(self.asp.data)) > 0:
            # Extract amplitudes from original positions
            amp_indices = np.nonzero(self.asp.data)[0]
            amplitudes = self.asp.data[amp_indices]

            # Apply to shuffled positions
            pspk_scaled = pspk.astype(float)
            pspk_indices = np.nonzero(pspk)[0]

            # Match amplitudes to shuffled positions
            if len(pspk_indices) == len(amplitudes):
                pspk_scaled[pspk_indices] = amplitudes
            elif len(pspk_indices) > 0:
                # Fallback: use mean amplitude
                pspk_scaled[pspk_indices] = np.mean(amplitudes)

            psconv = Neuron.get_restored_calcium(pspk_scaled, self.default_t_rise, opt_t_off)
        else:
            psconv = Neuron.get_restored_calcium(pspk, self.default_t_rise, opt_t_off)

        # Combine shuffled spikes with original background
        shuf_ca = psconv[: len(self.ca.data)] + background
        return shuf_ca

    def _shuffle_calcium_data_chunks_based(self, n=100, seed=None, **kwargs):
        """Shuffle calcium by dividing into chunks and reordering.
        
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
        - Useful for testing significance of long-range correlations        """
        # Validate input
        check_positive(n=n)
        
        # Set seed for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Divide calcium into chunks and shuffle their order
        ca = self.ca.data
        chunks = np.array_split(ca, n)
        inds = np.arange(len(chunks))
        np.random.shuffle(inds)

        # Concatenate shuffled chunks
        shuf_ca = np.concatenate([chunks[i] for i in inds])

        return shuf_ca

    def _shuffle_calcium_data_roll_based(self, shift=None, seed=None, **kwargs):
        """Shuffle calcium by circular shift (rolling).
        
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
            If signal is too short for valid shuffling range.        """
        opt_t_off = self.get_t_off()
        
        if shift is None:
            # Set seed for reproducibility
            if seed is not None:
                np.random.seed(seed)
                
            # Ensure valid range for random shift
            min_shift = int(3 * opt_t_off)
            max_shift = self.n_frames - int(3 * opt_t_off)
            if min_shift >= max_shift:
                raise ValueError(
                    f"Signal too short for roll-based shuffling. "
                    f"Need at least {2 * int(3 * opt_t_off)} frames, but have {self.n_frames}"
                )
            shift = np.random.randint(min_shift, max_shift)
        else:
            # Validate provided shift
            if not isinstance(shift, (int, np.integer)):
                raise ValueError(f"shift must be integer, got {type(shift).__name__}")
            if shift < 0 or shift >= self.n_frames:
                raise ValueError(f"shift must be in range [0, {self.n_frames-1}], got {shift}")

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
            raise AttributeError("Unable to shuffle spikes without spikes data")

        valid_methods = ['isi_based']
        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Must be one of {valid_methods}")

        fn = getattr(self, f"_shuffle_spikes_data_{method}")

        sh_data = fn(seed=seed, **kwargs)
        if not return_array:
            return TimeSeries(sh_data, discrete=True)
        else:
            return sh_data

    def _shuffle_spikes_data_isi_based(self, seed=None):
        """Shuffle spikes preserving inter-spike interval statistics.
        
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
        - May produce different temporal patterns despite same ISI distribution        """
        # Set seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            
        nfr = self.n_frames

        pseudo_spikes = np.zeros(nfr)
        event_inds = np.where(self.sp.data != 0)[0]

        if (
            len(event_inds) == 0
        ):  # if no events were detected, there is nothing to shuffle
            return self.sp.data

        event_vals = self.sp.data[event_inds].copy()  # Copy to avoid modifying original
        
        # Calculate safe range for first position
        event_range = max(event_inds) - min(event_inds)
        max_start = max(1, nfr - event_range - 1)
        first_random_pos = np.random.choice(max_start)

        interspike_intervals = np.diff(event_inds)
        rng = np.arange(len(interspike_intervals))
        np.random.shuffle(rng)
        disordered_interspike_intervals = interspike_intervals[rng]

        pseudo_event_inds = np.cumsum(
            np.insert(disordered_interspike_intervals, 0, first_random_pos)
        )
        
        # Ensure indices are within bounds
        valid_mask = pseudo_event_inds < nfr
        pseudo_event_inds = pseudo_event_inds[valid_mask]
        event_vals = event_vals[:len(pseudo_event_inds)]

        np.random.shuffle(event_vals)
        pseudo_spikes[pseudo_event_inds] = event_vals

        return pseudo_spikes
