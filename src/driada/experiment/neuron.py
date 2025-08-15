import numpy as np
import logging
from numba import njit
from scipy.stats import median_abs_deviation
from scipy.optimize import minimize
from ..information.info_base import TimeSeries
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
        Calcium imaging time series data
    sp : array-like, optional
        Spike train data (discrete events). If None, spike-related
        methods will not be available.
    default_t_rise : float, default=0.25
        Rise time constant in seconds for calcium transients
    default_t_off : float, default=2.0
        Decay time constant in seconds for calcium transients
    fps : float, default=20.0
        Sampling rate in frames per second
    fit_individual_t_off : bool, default=False
        Whether to fit decay time for this specific neuron
        
    Attributes
    ----------
    cell_id : int or str
        Neuron identifier
    ca : TimeSeries
        Preprocessed calcium time series
    sp : TimeSeries or None
        Spike train time series
    n_frames : int
        Number of time points
    sp_count : int
        Total number of spikes
    t_off : float or None
        Fitted decay time constant (in frames)
    noise_ampl : float or None
        Estimated noise amplitude
    mad : float or None
        Median absolute deviation of calcium signal
    snr : float or None
        Signal-to-noise ratio
    """

    @staticmethod
    @njit()
    def spike_form(t, t_rise, t_off):
        form = (1 - np.exp(-t / t_rise)) * np.exp(-t / t_off)
        return form / max(form)

    @staticmethod
    def get_restored_calcium(sp, t_rise, t_off):
        x = np.linspace(0, 1000, num=1000)
        spform = Neuron.spike_form(x, t_rise=t_rise, t_off=t_off)
        conv = np.convolve(sp, spform)
        return conv

    @staticmethod
    def ca_mse_error(t_off, ca, spk, t_rise):
        """Calculate MSE between calcium and restored calcium from spikes.
        
        Parameters
        ----------
        t_off : float
            Decay time constant (in frames)
        ca : array-like
            Observed calcium signal
        spk : array-like
            Spike train
        t_rise : float
            Rise time constant (in frames)
            
        Returns
        -------
        float
            Root mean square error
        """
        # Note: parameter order matches minimize() expectations
        re_ca = Neuron.get_restored_calcium(spk, t_rise, t_off)
        return np.sqrt(np.sum(np.abs(ca - re_ca[: len(ca)]) ** 2) / len(ca))

    @staticmethod
    @njit
    def calcium_preprocessing(ca):
        """Preprocess calcium signal.
        
        - Converts to float64
        - Clips negative values to 0
        - Adds small noise to prevent numerical issues
        
        Parameters
        ----------
        ca : array-like
            Raw calcium signal
            
        Returns
        -------
        ndarray
            Preprocessed calcium signal
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
    ):

        if default_t_rise is None:
            default_t_rise = DEFAULT_T_RISE
        if default_t_off is None:
            default_t_off = DEFAULT_T_OFF
        if fps is None:
            fps = DEFAULT_FPS

        self.cell_id = cell_id
        self.ca = TimeSeries(Neuron.calcium_preprocessing(ca), discrete=False)
        if sp is None:
            self.sp = None
        else:
            self.sp = TimeSeries(sp.astype(int), discrete=True)
        self.n_frames = len(self.ca.data)

        self.sp_count = (
            np.sum(self.sp.data.astype(bool).astype(int)) if self.sp is not None else 0
        )
        self.t_off = None
        self.noise_ampl = None
        self.mad = None
        self.snr = None

        self.default_t_off = default_t_off * fps
        self.default_t_rise = default_t_rise * fps

        if fit_individual_t_off:
            t_off = self.get_t_off()
        else:
            t_off = self.default_t_off

        # add shuffle mask according to computed characteristic calcium decay time
        self.ca.shuffle_mask = np.ones(self.n_frames).astype(bool)
        min_shift = int(t_off * MIN_CA_SHIFT)
        self.ca.shuffle_mask[:min_shift] = False
        self.ca.shuffle_mask[self.n_frames - min_shift :] = False

    def reconstruct_spikes(self, method="wavelet", **kwargs):
        """Reconstruct spikes from calcium signal.
        
        Parameters
        ----------
        method : str, default='wavelet'
            Reconstruction method. Currently only 'wavelet' is supported.
        **kwargs
            Additional parameters passed to the reconstruction method
            
        Returns
        -------
        ndarray
            Reconstructed spike train as discrete events
        """
        if method == "wavelet":
            # Use wavelet event detection
            wvt_kwargs = {
                "fps": kwargs.get("fps", self.default_t_off / DEFAULT_T_OFF),
                "min_event_dur": kwargs.get("min_event_dur", 0.5),
                "max_event_dur": kwargs.get("max_event_dur", 2.5),
            }
            
            # Extract calcium data as 2D array (features x time)
            ca_data = self.ca.data.reshape(1, -1)
            
            # Extract events
            st_ev_inds, end_ev_inds, _ = extract_wvt_events(ca_data, wvt_kwargs)
            
            # Convert to spike array
            spikes = events_to_ts_array(
                self.n_frames, st_ev_inds, end_ev_inds, wvt_kwargs["fps"]
            )
            
            return spikes.flatten()
        else:
            raise NotImplementedError(f"Method '{method}' not implemented")

    def get_mad(self):
        if self.mad is None:
            try:
                self.snr, self.mad = self._calc_snr()
            except ValueError:
                self.mad = median_abs_deviation(self.ca.data)
        return self.mad

    def get_snr(self):
        if self.snr is None:
            self.snr, self.mad = self._calc_snr()
        return self.snr

    def _calc_snr(self):
        spk_inds = np.nonzero(self.sp.data)[0]
        mad = median_abs_deviation(self.ca.data)
        if len(spk_inds) > 0:
            sn = np.mean(self.ca.data[spk_inds]) / mad
            if np.isnan(sn):
                raise ValueError("Error in snr calculation")
        else:
            raise ValueError("No spikes found!")

        return sn, mad

    def get_t_off(self):
        if self.t_off is None:
            self.t_off, self.noise_ampl = self._fit_t_off()

        return self.t_off

    def get_noise_ampl(self):
        if self.noise_ampl is None:
            self.t_off, self.noise_ampl = self._fit_t_off()

        return self.noise_ampl

    def _fit_t_off(self):

        # TODO: fit for an arbitrary kernel form.
        # TODO: add nonlinear summation fit if needed

        res = minimize(
            Neuron.ca_mse_error,
            (np.array([self.default_t_off])),
            args=(self.ca.data, self.sp.data, self.default_t_rise),
        )
        opt_t_off = res.x[0]
        noise_amplitude = res.fun

        if opt_t_off > self.default_t_off * 5:
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Calculated t_off={int(opt_t_off)} for neuron {self.cell_id} is suspiciously high, "
                f"check signal quality. t_off has been automatically lowered to {self.default_t_off*5}"
            )

        return min(opt_t_off, self.default_t_off * 5), noise_amplitude

    def get_shuffled_calcium(self, method="roll_based", return_array=True, **kwargs):
        """Get shuffled calcium signal.
        
        Parameters
        ----------
        method : {'roll_based', 'waveform_based', 'chunks_based'}, default='roll_based'
            Shuffling method to use
        return_array : bool, default=True
            If True, return numpy array. If False, return TimeSeries object.
        **kwargs
            Additional arguments passed to shuffling method
            
        Returns
        -------
        ndarray or TimeSeries
            Shuffled calcium signal
        """
        fn = getattr(self, f"_shuffle_calcium_data_{method}")

        sh_ca = fn(**kwargs)
        sh_ca = Neuron.calcium_preprocessing(sh_ca)
        if not return_array:
            sh_ca = TimeSeries(sh_ca, discrete=False)

        return sh_ca

    def _shuffle_calcium_data_waveform_based(self, **kwargs):

        shuf_ca = np.zeros(self.n_frames)
        opt_t_off = self.get_t_off()
        # noise_amplitude = self.get_noise_ampl()  # For future use with noise addition
        # noise = np.random.normal(loc = 0, scale = noise_amplitude, size = len(self.ca))

        conv = Neuron.get_restored_calcium(self.sp.data, self.default_t_rise, opt_t_off)
        background = self.ca.data - conv[: len(self.ca.data)]

        pspk = self._shuffle_spikes_data_isi_based()
        psconv = Neuron.get_restored_calcium(pspk, self.default_t_rise, opt_t_off)

        # shuf_ca = conv[:len(self.ca.data)] + noise
        shuf_ca = psconv[: len(self.ca.data)] + background
        return shuf_ca

    def _shuffle_calcium_data_chunks_based(self, **kwargs):
        if "n" not in kwargs:
            n = 100
        else:
            n = kwargs["n"]

        shuf_ca = np.zeros(self.n_frames)
        ca = self.ca.data
        chunks = np.array_split(ca, n)
        inds = np.arange(len(chunks))
        np.random.shuffle(inds)

        shuf_ca[:] = np.concatenate([chunks[i] for i in inds])

        return shuf_ca

    def _shuffle_calcium_data_roll_based(self, **kwargs):
        opt_t_off = self.get_t_off()
        if "shift" in kwargs:
            shift = kwargs["shift"]
        else:
            # Ensure valid range for random shift
            min_shift = int(3 * opt_t_off)
            max_shift = self.n_frames - int(3 * opt_t_off)
            if min_shift >= max_shift:
                raise ValueError(
                    f"Signal too short for roll-based shuffling. "
                    f"Need at least {2 * int(3 * opt_t_off)} frames, but have {self.n_frames}"
                )
            shift = np.random.randint(min_shift, max_shift)

        shuf_ca = np.roll(self.ca.data, shift)

        return shuf_ca

    def get_shuffled_spikes(self, method="isi_based", return_array=True, **kwargs):
        """Get shuffled spike train.
        
        Parameters
        ----------
        method : {'isi_based'}, default='isi_based'
            Shuffling method to use
        return_array : bool, default=True
            If True, return numpy array. If False, return TimeSeries object.
        **kwargs
            Additional arguments passed to shuffling method
            
        Returns
        -------
        ndarray or TimeSeries
            Shuffled spike train
            
        Raises
        ------
        AttributeError
            If no spike data is available
        """
        if self.sp is None:
            raise AttributeError("Unable to shuffle spikes without spikes data")

        fn = getattr(self, f"_shuffle_spikes_data_{method}")

        sh_data = fn(**kwargs)
        if not return_array:
            return TimeSeries(sh_data, discrete=True)
        else:
            return sh_data

    def _shuffle_spikes_data_isi_based(self):
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
