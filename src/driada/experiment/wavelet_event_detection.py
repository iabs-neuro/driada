import numpy as np
import tqdm
import matplotlib.pyplot as plt

# Fix scipy compatibility issue for ssqueezepy
import scipy.integrate

if not hasattr(scipy.integrate, "trapz"):
    scipy.integrate.trapz = scipy.integrate.trapezoid

from ssqueezepy import cwt
from ssqueezepy.wavelets import Wavelet, time_resolution

from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelmax
from numba import njit

from .wavelet_ridge import Ridge, ridges_to_containers

WVT_EVENT_DETECTION_PARAMS = {
    "fps": 20,  # fps, frames
    "sigma": 8,  # smoothing parameter for peak detection, frames
    "beta": 2,  # Generalized Morse Wavelet parameter, FIXED
    "gamma": 3,  # Generalized Morse Wavelet parameter, FIXED
    "eps": 10,  # spacing between consecutive events, frames
    "manual_scales": np.logspace(2.5, 5.5, 50, base=2),
    # ridge filtering params
    "scale_length_thr": 40,  # min number of scales where ridge is present thr, higher = less events. max=len(manual_scales)
    "max_scale_thr": 7,  # index of a scale with max ridge intensity thr, higher = less events. < 5 = noise, > 20 = huge events
    "max_ampl_thr": 0.05,  # max ridge intensity thr, higher = less events. < 5 = noise, > 20 = huge events
    "max_dur_thr": 200,  # max event duration thr, higher = more events (but probably strange ones)
}
MIN_EVENT_DUR = 0.5  # sec
MAX_EVENT_DUR = 2.5  # sec


def wvt_viz(x, Wx):
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    axs[0].set_xlim(0, len(x))
    axs[0].plot(x, c="b")
    axs[1].imshow(np.abs(Wx), aspect="auto", cmap="turbo")


def get_cwt_ridges(
    sig, wavelet=None, fps=20, scmin=150, scmax=250, all_wvt_times=None, wvt_scales=None
):
    if wvt_scales is not None:
        scales = wvt_scales
    else:
        scales = "log-piecewise"
    W, wvt_scales = cwt(sig, wavelet=wavelet, fs=fps, scales=scales)

    # wvtdata = np.real(np.abs(W))
    wvtdata = np.real(W)
    scale_inds = np.arange(scmin, scmax)[::-1]
    if all_wvt_times is None:
        all_wvt_times = [
            time_resolution(wavelet, scale=wvt_scales[sc], nondim=False, min_decay=200)
            for sc in scale_inds
        ]

    # determine peak positions for all scales
    peaks = np.zeros((len(scale_inds), len(sig)))

    all_ridges = []
    for i, si in enumerate(scale_inds[:]):
        wvt_time = all_wvt_times[i]
        max_inds = argrelmax(wvtdata[si, :], order=10)[0]
        peaks[i, max_inds] = wvtdata[si, max_inds]
        # max_inds = np.nonzero(peaks[i,:])[0]
        # print(peaks[i, max_inds])

        if len(all_ridges) == 0:
            all_ridges = [
                Ridge(mi, peaks[i, mi], wvt_scales[si], wvt_time) for mi in max_inds
            ]
        else:
            # 1. extend old ridges
            prev_wvt_time = all_wvt_times[i - 1]
            live_ridges = [ridge for ridge in all_ridges if not ridge.terminated]
            maxima_used_for_prolongation = []

            for ridge in live_ridges:
                # 1.1 get ridge tip from previous scale
                last_max_index = ridge.tip()
                # 1.2 compute time window based on 68% of wavelet energy
                wlb, wrb = (
                    last_max_index - prev_wvt_time,
                    last_max_index + prev_wvt_time,
                )
                # 1.3 get list of candidate maxima of the current scale falling into the window
                candidates = [mi for mi in max_inds if (mi > wlb) and (mi < wrb)]
                # 1.4 extending ridges
                if len(candidates) == 0:
                    # gaps lead to ridge termination
                    # print(f'ridge with start time {ridge.indices[0]} terminated')
                    ridge.terminate()
                elif len(candidates) == 1:
                    # extend ridge
                    cand = candidates[0]
                    ridge.extend(cand, peaks[i, cand], wvt_scales[si], wvt_time)
                    maxima_used_for_prolongation.append(cand)
                    # print(f'ridge with start time {ridge.indices[0]} extended')
                else:
                    # extend ridge with the best maximum, others will later form new ridges
                    best_cand = candidates[np.argmax(peaks[i, np.array(candidates)])]
                    ridge.extend(
                        best_cand, peaks[i, best_cand], wvt_scales[si], wvt_time
                    )
                    maxima_used_for_prolongation.append(best_cand)
                    # maxima_used_for_prolongation.extend(candidates)

            # 2. generate new ridges
            new_ridges = [
                Ridge(mi, peaks[i, mi], wvt_scales[si], wvt_time)
                for mi in max_inds
                if mi not in maxima_used_for_prolongation
            ]
            all_ridges.extend(new_ridges)

    for r in all_ridges:
        r.terminate()

    return all_ridges


# TODO: add support for numba >0.59.0 or "numba_acceleration" flag
@njit()
def get_cwt_ridges_fast(wvtdata, peaks, wvt_times, wvt_scales):
    # determine peak positions for all scales

    start = True
    for si in range(wvtdata.shape[0]):
        wvt_time = wvt_times[si]
        max_inds = np.nonzero(peaks[si, :])[0]

        if start:
            all_ridges = [
                Ridge(mi, peaks[si, mi], wvt_scales[si], wvt_time) for mi in max_inds
            ]
            start = False
        else:
            # 1. extend old ridges
            prev_wvt_time = wvt_times[si - 1]
            live_ridges = [ridge for ridge in all_ridges if not ridge.terminated]
            maxima_used_for_prolongation = []

            for ridge in live_ridges:
                # 1.1 get ridge tip from previous scale
                last_max_index = ridge.tip()
                # 1.2 compute time window based on 68% of wavelet energy
                wlb, wrb = (
                    last_max_index - prev_wvt_time,
                    last_max_index + prev_wvt_time,
                )
                # 1.3 get list of candidate maxima of the current scale falling into the window
                candidates = [mi for mi in max_inds if (mi > wlb) and (mi < wrb)]
                # 1.4 extending ridges
                if len(candidates) == 0:
                    # gaps lead to ridge termination
                    # print(f'ridge with start time {ridge.indices[0]} terminated')
                    ridge.terminate()
                elif len(candidates) == 1:
                    # extend ridge
                    cand = candidates[0]
                    ridge.extend(cand, peaks[si, cand], wvt_scales[si], wvt_time)
                    maxima_used_for_prolongation.append(cand)
                    # print(f'ridge with start time {ridge.indices[0]} extended')
                else:
                    # extend ridge with the best maximum, others will later form new ridges
                    best_cand = candidates[np.argmax(peaks[si, np.array(candidates)])]
                    ridge.extend(
                        best_cand, peaks[si, best_cand], wvt_scales[si], wvt_time
                    )
                    maxima_used_for_prolongation.append(best_cand)
                    # maxima_used_for_prolongation.extend(candidates)

            # 2. generate new ridges
            new_ridges = [
                Ridge(mi, peaks[si, mi], wvt_scales[si], wvt_time)
                for mi in max_inds
                if mi not in maxima_used_for_prolongation
            ]
            # Use += instead of extend() to fix Numba 0.60+ type inference issue
            all_ridges += new_ridges

    for r in all_ridges:
        r.terminate()

    return all_ridges


def passing_criterion(
    ridge, scale_length_thr=40, max_scale_thr=10, max_ampl_thr=0.05, max_dur_thr=100
):
    crit = (
        ridge.length >= scale_length_thr
        and ridge.max_scale >= max_scale_thr
        and ridge.max_ampl >= max_ampl_thr
        and ridge.duration <= max_dur_thr
    )
    return crit


def get_events_from_ridges(
    all_ridges,
    scale_length_thr=40,
    max_scale_thr=10,
    max_ampl_thr=0.05,
    max_dur_thr=100,
):
    event_ridges = [
        r
        for r in all_ridges
        if passing_criterion(
            r,
            scale_length_thr=scale_length_thr,
            max_scale_thr=max_scale_thr,
            max_ampl_thr=max_ampl_thr,
            max_dur_thr=max_dur_thr,
        )
    ]

    st_evinds = [r.indices[0] for r in event_ridges]
    end_evinds = [r.indices[-1] for r in event_ridges]
    return st_evinds, end_evinds


def events_from_trace(
    trace,
    wavelet,
    manual_scales,
    rel_wvt_times,
    fps=20,
    sigma=8,
    eps=10,
    scale_length_thr=40,
    max_scale_thr=7,
    max_ampl_thr=0.05,
    max_dur_thr=200,
):

    trace = (trace - min(trace)) / (max(trace) - min(trace))
    sig = gaussian_filter1d(trace, sigma=sigma)

    W, wvt_scales = cwt(sig, wavelet=wavelet, fs=fps, scales=manual_scales)
    rev_wvtdata = np.real(W)

    all_max_inds = argrelmax(rev_wvtdata, axis=1, order=eps)
    peaks = np.zeros(rev_wvtdata.shape)
    peaks[all_max_inds] = rev_wvtdata[all_max_inds]

    all_ridges = get_cwt_ridges_fast(rev_wvtdata, peaks, rel_wvt_times, manual_scales)

    st_evinds, end_evinds = get_events_from_ridges(
        all_ridges,
        scale_length_thr=scale_length_thr,
        max_scale_thr=max_scale_thr,
        max_ampl_thr=max_ampl_thr,
        max_dur_thr=max_dur_thr,
    )

    return all_ridges, st_evinds, end_evinds


def extract_wvt_events(traces, wvt_kwargs):
    fps = wvt_kwargs.get("fps", 20)
    beta = wvt_kwargs.get("beta", 2)
    gamma = wvt_kwargs.get("gamma", 3)
    sigma = wvt_kwargs.get("sigma", 8)
    eps = wvt_kwargs.get("eps", 10)
    manual_scales = wvt_kwargs.get("manual_scales", np.logspace(2.5, 5.5, 50, base=2))

    scale_length_thr = wvt_kwargs.get("scale_length_thr", 40)
    max_scale_thr = wvt_kwargs.get("max_scale_thr", 7)
    max_ampl_thr = wvt_kwargs.get("max_ampl_thr", 0.05)
    max_dur_thr = wvt_kwargs.get("max_dur_thr", 200)

    wavelet = Wavelet(
        ("gmw", {"gamma": gamma, "beta": beta, "centered_scale": True}), N=8196
    )

    rel_wvt_times = [
        time_resolution(wavelet, scale=sc, nondim=False, min_decay=200)
        for sc in manual_scales
    ]

    st_ev_inds = []
    end_ev_inds = []
    all_ridges = []
    for i, trace in tqdm.tqdm(enumerate(traces), total=len(traces)):
        ridges, st_ev, end_ev = events_from_trace(
            trace,
            wavelet,
            manual_scales,
            rel_wvt_times,
            fps=fps,
            sigma=sigma,
            eps=eps,
            scale_length_thr=scale_length_thr,
            max_scale_thr=max_scale_thr,
            max_ampl_thr=max_ampl_thr,
            max_dur_thr=max_dur_thr,
        )

        st_ev_inds.append(st_ev)
        end_ev_inds.append(end_ev)
        all_ridges.append(ridges)

    return st_ev_inds, end_ev_inds, all_ridges


@njit
def events_to_ts_array_numba(
    length,
    ncells,
    st_ev_inds_flat,
    end_ev_inds_flat,
    event_counts,
    fps,
    min_event_dur,
    max_event_dur,
):
    """Numba-optimized version of events_to_ts_array."""
    spikes = np.zeros((ncells, length))

    mindur = int(min_event_dur * fps)
    maxdur = int(max_event_dur * fps)

    event_idx = 0
    for i in range(ncells):
        for j in range(event_counts[i]):
            start = int(st_ev_inds_flat[event_idx])
            end = int(end_ev_inds_flat[event_idx])
            start_, end_ = min(start, end), max(start, end)
            dur = end_ - start_
            if mindur <= dur <= maxdur:
                spikes[i, start_:end_] = 1
            elif dur > maxdur:
                spikes[i, start_ : start_ + maxdur] = 1
            else:
                middle = (start_ + end_) // 2
                spikes[i, int(middle - mindur // 2) : int(middle + mindur // 2)] = 1
            event_idx += 1

    return spikes


def events_to_ts_array(length, st_ev_inds, end_ev_inds, fps):
    """Convert event indices to time series array with spike trains."""
    ncells = len(end_ev_inds)

    # Flatten the jagged arrays for numba
    event_counts = np.array([len(st_ev_inds[i]) for i in range(ncells)])
    st_ev_inds_flat = np.concatenate([st_ev_inds[i] for i in range(ncells)])
    end_ev_inds_flat = np.concatenate([end_ev_inds[i] for i in range(ncells)])

    # Call numba function
    return events_to_ts_array_numba(
        length,
        ncells,
        st_ev_inds_flat,
        end_ev_inds_flat,
        event_counts,
        fps,
        MIN_EVENT_DUR,
        MAX_EVENT_DUR,
    )
