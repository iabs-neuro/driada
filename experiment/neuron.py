import numpy as np
from numba import njit
from scipy.stats import median_abs_deviation
from scipy.optimize import minimize
from ..signals.sig_base import TimeSeries

DEFAULT_T_RISE = 0.25 #sec
DEFAULT_T_OFF = 2.0 #sec

DEFAULT_FPS = 20.0 #frames per sec
DEFAULT_MIN_BEHAVIOUR_TIME = 0.25 #sec

#TODO: add numba decorators where possible
class Neuron():
    """
    Class for representing all information about a single neuron.

    Attributes
    ----------
    test: str
        description

    Methods
    -------
    test(arg=None)
        description
    """

    @staticmethod
    @njit()
    def spike_form(t, t_rise, t_off):
        form = (1-np.exp(-t/t_rise))*np.exp(-t/t_off)
        return form/max(form)


    @staticmethod
    def get_restored_calcium(sp, t_rise, t_off):
        x = np.linspace(0, 1000, num = 1000)
        spform = Neuron.spike_form(x, t_rise=t_rise, t_off=t_off)
        conv = np.convolve(sp, spform)
        return conv


    @staticmethod
    def ca_mse_error(t_off, ca, spk, t_rise):
        re_ca = Neuron.get_restored_calcium(spk, t_rise, t_off)
        return np.sqrt(np.sum(np.abs(ca - re_ca[:len(ca)])**2)/len(ca))


    @staticmethod
    def calcium_preprocessing(ca):
        ca[np.where(ca < 0)[0]] = 0
        ca += np.random.random(size=len(ca))*1e-8
        return ca


    def __init__(self, cell_id, ca, sp, default_t_rise=DEFAULT_T_RISE, default_t_off=DEFAULT_T_OFF, fps=DEFAULT_FPS):

        if default_t_rise is None:
            default_t_rise = DEFAULT_T_RISE
        if default_t_off is None:
            default_t_off = DEFAULT_T_OFF
        if fps is None:
            fps = DEFAULT_FPS

        self.cell_id = cell_id
        self.ca = TimeSeries(Neuron.calcium_preprocessing(ca), discrete = False)
        if sp is None:
            self.sp = None
        else:
            self.sp = TimeSeries(sp, discrete = False)
        self.n_frames = len(ca.data)

        self.sp_count = np.sum(self.sp.data.astype(bool).astype(int))
        self.t_off = None
        self.noise_ampl = None
        self.mad = None
        self.snr = None

        self.default_t_off = default_t_off*fps
        self.default_t_rise = default_t_rise*fps

        self.get_t_off()


    def reconstruct_spikes(**kwargs):
        raise AttributeError('Spike reconstruction not implemented')


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
            sn = np.mean(self.sp.data[spk_inds])/mad
            if np.isnan(sn):
                raise ValueError('Error in snr calculation')
        else:
            raise ValueError('No spikes found!')

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

        #TODO: fit for an arbitrary kernel form.
        #TODO: add nonlinear summation fit if needed

        res = minimize(Neuron.ca_mse_error, (np.array([self.default_t_off])), args=(self.ca.data, self.sp.data, self.default_t_rise))
        opt_t_off = res.x[0]
        noise_amplitude = res.fun

        if opt_t_off > self.default_t_off*5:
            print(f'Calculated t_off={int(opt_t_off)} for neuron {self.cell_id} is suspiciously high, check signal quality. t_off has been automatically lowered to {self.default_t_off*5}')

        return min(opt_t_off, self.default_t_off*5), noise_amplitude


    def get_shuffled_calcium(self, method = 'roll_based', **kwargs):
        try:
            fn = getattr(self, f'_shuffle_calcium_data_{method}')
        except AttributeError():
            raise UserWarning('Unknown calcium data shuffling method')

        sh_ca = fn(**kwargs)
        sh_ca = TimeSeries(Neuron.calcium_preprocessing(sh_ca), discrete = False)

        return sh_ca


    def _shuffle_calcium_data_waveform_based(self, **kwargs):

        shuf_ca = np.zeros(self.n_frames)
        opt_t_off, noise_amplitude = self.get_t_off(), self.get_noise_ampl()

        #noise = np.random.normal(loc = 0, scale = noise_amplitude, size = len(self.ca))

        conv = Neuron.get_restored_calcium(self.sp.data, 5, opt_t_off)
        background = self.ca.data - conv[:len(self.ca.data)]

        pspk = self._shuffle_spikes_data_isi_based()
        psconv = Neuron.get_restored_calcium(pspk, 5, opt_t_off)

        #shuf_ca = conv[:len(self.ca.data)] + noise
        shuf_ca = psconv[:len(self.ca.data)] + background
        return shuf_ca


    def _shuffle_calcium_data_chunks_based(self, **kwargs):
        if 'n' not in kwargs:
            n = 100
        else:
            n = kwargs['n']

        shuf_ca = np.zeros(self.n_frames)
        ca = self.ca.data
        chunks = np.concatenate(np.split(ca[:-len(ca)%n], n), ca[-(len(ca)%n):])
        inds = np.arange(n)
        np.random.shuffle(inds)

        shuf_ca[:] = np.concatenate(tuple(np.array(chunks)[inds]))

        return shuf_ca


    def _shuffle_calcium_data_roll_based(self, **kwargs):
        opt_t_off = self.get_t_off()
        if 'shift' in kwargs:
            shift = kwargs['shift']
        else:
            shift = np.random.randint(3*opt_t_off, self.n_frames - 3*opt_t_off)

        shuf_ca = np.roll(self.ca.data, shift)

        return shuf_ca


    def get_shuffled_spikes(self, method = 'isi_based', **kwargs):
        if self.sp is None:
            raise AttributeError('Unable to shuffle spikes without spikes data')

        try:
            fn = getattr(self, f'_shuffle_spikes_data_{method}')
        except AttributeError():
            raise UserWarning('Unknown calcium data shuffling method')

        sh_data = fn(**kwargs)
        return TimeSeries(sh_data, discrete = False)


    def _shuffle_spikes_data_isi_based(self):
        nfr = self.n_frames

        pseudo_spikes = np.zeros(nfr)
        event_inds = np.where(self.sp.data != 0)[0]

        if len(event_inds) == 0: #if no events were detected, there is nothing to shuffle
            return self.sp.data

        event_vals = self.sp.data[event_inds]
        first_random_pos = np.random.choice(nfr - (max(event_inds) - min(event_inds)))

        interspike_intervals = np.diff(event_inds)
        rng = np.arange(len(interspike_intervals))
        np.random.shuffle(rng)
        disordered_interspike_intervals = interspike_intervals[rng]

        pseudo_event_inds = np.cumsum(np.insert(disordered_interspike_intervals,
                                                0, first_random_pos))

        pseudo_event_vals = event_vals
        np.random.shuffle(event_vals)
        pseudo_spikes[pseudo_event_inds] = pseudo_event_vals

        return pseudo_spikes
