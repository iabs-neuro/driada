from scipy import signal
import numpy as np
from numpy.fft import rfft, irfft
import pywt
from .complexity import ApEn
from ..information.info_base import TimeSeries

def multiopt(x, thr):
    if thr == 0.0:
        return x

    m = np.abs(x)

    if m < thr:
        return 0

    return np.sign(x) * (m - 1.0 * thr * np.exp(1 - 1.0 * m / thr))


def truncate_sig(coeffs, thr):
    return np.array([multiopt(x, thr) for x in coeffs])


def ts_wavelet_denoise(d, denoising_params):
    sig = Signal(d)
    sig.get_idwt(**denoising_params)
    try:
        new_d = sig.idwt[:-1]
    except:
        new_d = sig.idwt

    return new_d


# TODO: refactor Signal class into TimeSeries child class
class Signal(TimeSeries):
    '''
    Class implementing additional functionality for continuous time series
    '''
    def __init__(self, data):
        super().__init__(data, discrete=False)

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
            raise Exception('Wrong signals property!')

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
        # wavelet = signals.morlet
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
        beta = np.polyfit(x, y, 1)[0]
        print('spectrum power = ', beta)
        H = (beta - 1) / 2
        return H

    def ap_en(self):
        if not self.ap_en is None:
            return self.ap_en

        return ApEn(self.data, 3, 1)