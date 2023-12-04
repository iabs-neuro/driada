import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from numpy.fft import rfft, irfft
import pywt


def multiopt(x, thr):
    if thr == 0.0:
        return x

    m = np.abs(x)

    if m < thr:
        return 0

    return np.sign(x) * (m - 1.0 * thr * np.exp(1 - 1.0 * m / thr))


def truncate_sig(coeffs, thr):
    return np.array([multiopt(x, thr) for x in coeffs])


class Signal():

    def __init__(self, data):
        if not isinstance(data, np.ndarray):
            raise Exception('Wrong data type passed to time series constructor!')
        self.data = data
        self.mean = np.mean(data)
        self.std = np.std(data)

        self.ft = None
        self.cwt = None
        self.dwt = None
        self.hurst_exp = None

        self.mod_ft = None
        self.mod_cwt = None
        self.mod_dwt = None

        self.ift = None
        self.icwt = None
        self.idwt = None

        self.ap_en = None

    def get(self, fn, args=None):
        if not hasattr(self, fn):
            raise Exception('Wrong signal property!')

        d = getattr(self, fn)
        if d is None:
            constructor = 'get_' + fn
            getattr(self, constructor)(args)
            d = getattr(self, fn)

        return d

    def get_ft(self):
        sig = self.data
        ft = rfft(sig)
        self.ft = ft

    def get_ift(self, thr):
        ft = self.get('ft')
        mod_ft = ft.copy()
        # print(mod_ft.shape)
        mod_ft[(mod_ft > thr)] = 0
        self.ift = irfft(mod_ft)

    def modify_dwt(self, cd, thr):
        cd_mod = truncate_sig(cd, thr)
        return cd_mod

    def get_cwt(self, max_width=16):
        sig = self.data
        wavelet = signal.ricker
        # wavelet = signal.morlet
        widths = np.arange(1, max_width)
        cwt_mat = signal.cwt(sig, wavelet, widths)
        self.cwt = cwt_mat

    def get_dwt(self, wavelet):
        ca, cd = pywt.dwt(self.data, wavelet)
        self.dwt = (ca, cd)

    def get_idwt(self, wavelet, thr, level):
        n = len(self.data)
        max_level = pywt.dwt_max_level(len(self.data), wavelet)
        if not level is None and level > max_level:
            print('wrong level value, discarded to %s' % max_level)
            level = max_level

        all_coeffs = pywt.wavedec(self.data, wavelet, mode='smooth', level=level)
        # print(sum(all_coeffs[1]))
        for i in range(1, len(all_coeffs)):
            cd = all_coeffs[i]
            sigma = np.median(np.abs(cd)) / 0.6745
            thr = sigma * np.sqrt(2 * np.log(n))
            all_coeffs[i] = self.modify_dwt(cd, thr)

        # restored = pywt.idwt(all_coeffs[0], all_coeffs[1], wavelet, mode = 'smooth')
        # print(sum(all_coeffs[1]))
        restored = pywt.waverec(all_coeffs, wavelet, mode='smooth')
        # print(restored)
        self.idwt = restored

    def get_hurst_exp(self, wavelet, lvl_min, lvl_max):
        n = len(self.data)
        max_possible_level = pywt.dwt_max_level(len(self.data), wavelet)
        if not (lvl_min is None or lvl_max is None) and lvl_max > max_possible_level:
            print('wrong level value, discarded to %s' % max_possible_level)
            lvl_max = max_possible_level

        all_coeffs = pywt.wavedec(self.data, wavelet, mode='smooth', level=lvl_max)
        energies = []
        for i in range(1, len(all_coeffs)):
            if lvl_min <= i and lvl_max >= i:
                dlist = list(all_coeffs[i])
                energies.append(sum([d ** 2 for d in dlist]) / len(dlist))

        x = range(len(energies))[::-1]
        y = np.log2(energies)
        plt.plot(x, y)
        beta = np.polyfit(x, y, 1)[0]
        print('spectrum power = ', beta)
        H = (beta - 1) / 2
        return H

    def ap_en(self):
        if not self.ap_en is None:
            return self.ap_en

        return ApEn(self.data, 3, 1)


def ts_wavelet_denoise(d, denoising_params):
    new_d = np.zeros(len(d))
    sig = Signal(d)
    sig.get_idwt(**denoising_params)
    try:
        new_d = sig.idwt[:-1]
    except:
        new_d = sig.idwt

    return new_d

class TimeSeries():

    @staticmethod
    def define_ts_type(ts):
        if len(ts) < 100:
            warnings.warn('Time series is too short for accurate type (discrete/continuous) determination')
        unique_vals = np.unique(ts)
        sc1 = len(unique_vals) / len(ts)
        hist = np.histogram(ts, bins=len(ts))[0]
        ent = entropy(hist)
        maxent = entropy(np.ones(len(ts)))
        sc2 = ent / maxent

        if sc1 > 0.70 and sc2 > 0.70:
            return False  # both scores are high - the variable is most probably continuous
        elif sc1 < 0.25 and sc2 < 0.25:
            return True  # both scores are low - the variable is most probably discrete
        else:
            raise ValueError(f'Unable to determine time series type automatically: score 1 = {sc1}, score 2 = {sc2}')

    def _check_input(self):
        pass

    def __init__(self, data, discrete=None):
        self.data = data
        if discrete is None:
            #warnings.warn('Time series type not specified and will be inferred automatically')
            self.discrete = TimeSeries.define_ts_type(data)
        else:
            self.discrete = discrete

        scaler = MinMaxScaler()
        self.scdata = scaler.fit_transform(self.data.reshape(-1, 1)).reshape(1, -1)[0]
        self.copula_normal_data = None
        if not self.discrete:
            self.copula_normal_data = copnorm(self.data).ravel()

        self.entropy = dict()
        self.kdtree = None
        self.kdtree_query = None

    def get_kdtree(self):
        if self.kdtree is None:
            tree = self._compute_kdtree()
            self.kdtree = tree

        return self.kdtree

    def _compute_kdtree(self):
        d = self.scdata.reshape(self.scdata.shape[0], -1)
        return build_tree(d)

    def get_kdtree_query(self, k=DEFAULT_NN):
        if self.kdtree_query is None:
            q = self._compute_kdtree_query(k=k)
            self.kdtree_query = q

        return self.kdtree_query

    def _compute_kdtree_query(self, k=DEFAULT_NN):
        tree = self.get_kdtree()
        return tree.query(self.scdata, k=k + 1)

    def get_entropy(self, ds=1):
        if ds not in self.entropy.keys():
            self._compute_entropy(ds=ds)
        return self.entropy[ds]

    def _compute_entropy(self, ds=1):
        if self.discrete:
            counts = []
            for val in np.unique(self.data[::ds]):
                counts.append(len(np.where(self.data[::ds] == val)[0]))

            self.entropy[ds] = scipy.stats.entropy(counts, base=np.e)

        else:
            self.entropy[ds] = get_tdmi(self.scdata[::ds], min_shift=1, max_shift=2)[0]
            #raise AttributeError('Entropy for continuous variables is not yet implemented'
