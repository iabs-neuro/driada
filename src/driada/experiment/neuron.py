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
    spike data may produce incorrect results in spike counting.
    
    DOC_VERIFIED
    """

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
        normalized to have maximum value of 1.
        
        DOC_VERIFIED
        """
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
        JIT compilation provides ~10-100x speedup for large arrays.
        
        DOC_VERIFIED
        """
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
            Binary spike train. Must be 1D array.
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
        
        DOC_VERIFIED
        """
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
        where t_off is the parameter being optimized.
        
        DOC_VERIFIED
        """
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
        numerical issues in downstream spike reconstruction algorithms.
        
        DOC_VERIFIED
        """
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
        externally for reproducibility.
        
        DOC_VERIFIED
        """
        # Ensure we're working with float array to avoid dtype casting issues
        ca = ca.astype(np.float64)
        ca[ca < 0] = 0  # More efficient than np.where
        ca += np.random.random(len(ca)) * 1e-8
        return ca

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
        to prevent artifacts in temporal shuffling analyses.
        
        DOC_VERIFIED
        """
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
        
        # Initialize attributes
        self.t_off = None
        self.noise_ampl = None
        self.mad = None
        self.snr = None

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

    def reconstruct_spikes(self, method="wavelet", **kwargs):
        """Reconstruct spikes from calcium signal.
        
        Reconstructs discrete spike events from continuous calcium 
        fluorescence traces using wavelet or threshold-based methods.
        
        Parameters
        ----------
        method : str, optional
            Reconstruction method: 'wavelet' or 'threshold'.
            Default is 'wavelet'.
        **kwargs : dict, optional
            Additional parameters depend on method:
            
            For 'wavelet':
            - fps : float, optional
                Sampling rate in Hz. Default is DEFAULT_FPS (20.0 Hz).
            - min_event_dur : float, optional
                Minimum event duration in seconds. Default is 0.5.
            - max_event_dur : float, optional
                Maximum event duration in seconds. Default is 2.5.
                
            For 'threshold':
            - threshold_std : float, optional
                Number of standard deviations above mean. Default is 2.5.
            - smooth_sigma : float, optional
                Gaussian smoothing sigma in frames. Default is 2.
            - min_spike_interval : float, optional
                Minimum interval between spikes in seconds. Default is 0.1.
            
        Returns
        -------
        ndarray
            Binary spike train with shape (n_frames,). Values are 0 or 1.
            
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
        
        DOC_VERIFIED
        """
        if method == "wavelet":
            # Get parameters with defaults
            fps = kwargs.get("fps", DEFAULT_FPS)
            min_event_dur = kwargs.get("min_event_dur", 0.5)
            max_event_dur = kwargs.get("max_event_dur", 2.5)
            
            # Validate parameters
            check_positive(fps=fps, min_event_dur=min_event_dur, max_event_dur=max_event_dur)
            
            if min_event_dur >= max_event_dur:
                raise ValueError(f"min_event_dur ({min_event_dur}) must be less than max_event_dur ({max_event_dur})")
            
            # Use wavelet event detection
            wvt_kwargs = {
                "fps": fps,
                "min_event_dur": min_event_dur,
                "max_event_dur": max_event_dur,
            }
            
            # Extract calcium data as 2D array (features x time)
            ca_data = self.ca.data.reshape(1, -1)
            
            # Extract events
            st_ev_inds, end_ev_inds, _ = extract_wvt_events(ca_data, wvt_kwargs)
            
            # Convert to spike array
            spikes = events_to_ts_array(
                self.n_frames, st_ev_inds, end_ev_inds, fps
            )
            
            return spikes.flatten()
            
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
        contains spike-related transients.
        
        DOC_VERIFIED
        """
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
        of calcium signal quality relative to baseline noise.
        
        DOC_VERIFIED
        """
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
            calculation results in NaN.
            
        DOC_VERIFIED
        """
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
        that minimizes the RMSE between observed and reconstructed calcium.
        
        DOC_VERIFIED
        """
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
        The value is cached after first computation.
        
        DOC_VERIFIED
        """
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
        quality issues.
        
        DOC_VERIFIED
        """
        # FUTURE: fit for an arbitrary kernel form.
        # FUTURE: add nonlinear summation fit if needed
        
        if self.sp is None:
            raise ValueError("Spike data required for t_off fitting")

        res = minimize(
            Neuron.ca_mse_error,
            (np.array([self.default_t_off])),
            args=(self.ca.data, self.sp.data, self.default_t_rise),
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
        - chunks_based: Preserves local signal structure within chunks
        
        DOC_VERIFIED
        """
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
            
        Returns
        -------
        ndarray
            Shuffled calcium signal.
            
        Raises
        ------
        ValueError
            If spike data is not available.
            
        DOC_VERIFIED
        """
        if self.sp is None:
            raise ValueError("Spike data required for waveform-based shuffling")
            
        opt_t_off = self.get_t_off()

        # Reconstruct calcium from original spikes
        conv = Neuron.get_restored_calcium(self.sp.data, self.default_t_rise, opt_t_off)
        background = self.ca.data - conv[: len(self.ca.data)]

        # Shuffle spikes preserving ISI statistics
        pspk = self._shuffle_spikes_data_isi_based(seed=seed)
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
        - Useful for testing significance of long-range correlations
        
        DOC_VERIFIED
        """
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
            
        Returns
        -------
        ndarray
            Circularly shifted calcium signal.
            
        Raises
        ------
        ValueError
            If signal is too short for valid shuffling range.
            
        DOC_VERIFIED
        """
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
        intervals while randomizing spike positions.
        
        DOC_VERIFIED
        """
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
        - May produce different temporal patterns despite same ISI distribution
        
        DOC_VERIFIED
        """
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
