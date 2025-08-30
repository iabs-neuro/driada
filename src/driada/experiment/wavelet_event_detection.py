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

from ..utils.jit import conditional_njit
from ..utils.data import check_positive, check_nonnegative
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
    """Visualize signal and its wavelet transform.
    
    Creates a two-panel plot showing the input signal and its wavelet
    transform magnitude.
    
    Parameters
    ----------
    x : array-like of shape (n_samples,)
        Input signal. Must be 1D.
    Wx : ndarray of shape (n_scales, n_time)
        Wavelet transform coefficients. Must have n_time == len(x).
        
    Raises
    ------
    ValueError
        If x is not 1D or Wx shape doesn't match x length.
        
    Notes
    -----
    The wavelet transform is displayed as magnitude (absolute value)
    using the 'turbo' colormap.
    
    DOC_VERIFIED
    """
    # Validate inputs
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"x must be 1D, got shape {x.shape}")
    
    Wx = np.asarray(Wx)
    if Wx.ndim != 2:
        raise ValueError(f"Wx must be 2D, got shape {Wx.shape}")
    
    if Wx.shape[1] != len(x):
        raise ValueError(f"Wx time dimension ({Wx.shape[1]}) must match x length ({len(x)})")
    
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    axs[0].set_xlim(0, len(x))
    axs[0].plot(x, c="b")
    axs[1].imshow(np.abs(Wx), aspect="auto", cmap="turbo")


def get_cwt_ridges(
    sig, wavelet=None, fps=20, scmin=150, scmax=250, all_wvt_times=None, wvt_scales=None
):
    """Extract ridges from continuous wavelet transform of a signal.
    
    Identifies ridges (connected paths of local maxima across scales) in the
    wavelet transform, which correspond to transient events in the signal.
    Ridges are tracked by connecting maxima at adjacent scales within a time
    window determined by wavelet support.
    
    Parameters
    ----------
    sig : array-like
        Input signal to analyze.
    wavelet : ssqueezepy.wavelets.Wavelet, optional
        Wavelet object to use. If None, uses default from cwt().
    fps : float, default=20
        Sampling rate in Hz.
    scmin : int, default=150
        Starting scale index (processes scales from scmax down to scmin).
    scmax : int, default=250
        Maximum scale index.
    all_wvt_times : list, optional
        Pre-computed wavelet time resolutions for each scale. If None,
        computes them using time_resolution().
    wvt_scales : array-like, optional
        Wavelet scales to use. If None, uses 'log-piecewise' default.
        
    Returns
    -------
    list of Ridge
        All detected ridges. Each Ridge object contains indices, amplitudes,
        scales, and time resolutions along the ridge path.
        
    Raises
    ------
    ValueError
        If fps is not positive.
        If scmin or scmax are negative.
        If scmin >= scmax.
        If sig is not 1-dimensional.
        
    Notes
    -----
    The algorithm processes scales from coarse (scmax) to fine (scmin),
    extending existing ridges when possible and creating new ones for
    unmatched maxima. Ridges are terminated when no maxima fall within
    the expected time window at the next scale.
    
    See Also
    --------
    get_cwt_ridges_fast : Numba-accelerated version
    Ridge : Ridge object storing ridge properties
    
    DOC_VERIFIED
    """
    # Validate parameters
    check_positive(fps=fps)
    check_nonnegative(scmin=scmin, scmax=scmax)
    if scmin >= scmax:
        raise ValueError(f"scmin ({scmin}) must be less than scmax ({scmax})")
    
    # Validate signal
    sig = np.asarray(sig)
    if sig.ndim != 1:
        raise ValueError(f"sig must be 1D, got shape {sig.shape}")
    
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


@conditional_njit()
def get_cwt_ridges_fast(wvtdata, peaks, wvt_times, wvt_scales):
    """Fast ridge extraction from pre-computed wavelet transform data.
    
    Numba-accelerated version of get_cwt_ridges that operates on pre-computed
    wavelet coefficients and peak locations. Implements the same ridge-tracking
    algorithm but requires pre-processing of the data.
    
    Parameters
    ----------
    wvtdata : ndarray
        Pre-computed wavelet transform coefficients, shape (n_scales, n_time).
        Real part of complex wavelet transform.
    peaks : ndarray
        Binary mask of peak locations, shape (n_scales, n_time).
        Non-zero values indicate local maxima positions.
    wvt_times : array-like
        Time resolutions for each scale, indicating the temporal support
        of the wavelet at that scale.
    wvt_scales : array-like
        Scale values corresponding to each row in wvtdata.
        
    Returns
    -------
    list of Ridge
        All detected ridges with their properties.
        
    Notes
    -----
    This function is decorated with @conditional_njit for optional JIT
    compilation. The algorithm processes all scales sequentially, extending
    ridges when maxima fall within the expected time window (±wvt_time).
    
    The main differences from get_cwt_ridges:
    - Operates on pre-computed data rather than raw signal
    - Processes all scales rather than a subset
    - Uses list concatenation (+=) for Numba 0.60+ compatibility
    
    See Also
    --------
    get_cwt_ridges : Original implementation
    events_from_trace : High-level function using this for event detection
    
    DOC_VERIFIED
    """
    # Validate inputs (before JIT compilation)
    wvtdata = np.asarray(wvtdata)
    peaks = np.asarray(peaks)
    wvt_times = np.asarray(wvt_times)
    wvt_scales = np.asarray(wvt_scales)
    
    if wvtdata.ndim != 2:
        raise ValueError(f"wvtdata must be 2D, got shape {wvtdata.shape}")
    if peaks.shape != wvtdata.shape:
        raise ValueError(f"peaks shape {peaks.shape} must match wvtdata shape {wvtdata.shape}")
    if len(wvt_times) != wvtdata.shape[0]:
        raise ValueError(f"wvt_times length ({len(wvt_times)}) must match number of scales ({wvtdata.shape[0]})")
    if len(wvt_scales) != wvtdata.shape[0]:
        raise ValueError(f"wvt_scales length ({len(wvt_scales)}) must match number of scales ({wvtdata.shape[0]})")
    
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
    """Check if a ridge meets criteria for being a valid event.
    
    Evaluates whether a detected ridge represents a significant calcium
    transient event based on multiple criteria including length, scale,
    amplitude, and duration.
    
    Parameters
    ----------
    ridge : Ridge
        Ridge object to evaluate.
    scale_length_thr : int, default=40
        Minimum number of scales the ridge must span. Higher values
        filter out shorter-lived events.
    max_scale_thr : int, default=10
        Minimum maximum scale index. Events must reach at least this
        scale. Lower values (<5) may include noise.
    max_ampl_thr : float, default=0.05
        Minimum maximum amplitude of the ridge. Higher values filter
        out low-amplitude events.
    max_dur_thr : int, default=100
        Maximum allowed duration in frames. Filters out unrealistically
        long events that may be artifacts.
        
    Returns
    -------
    bool
        True if ridge passes all criteria, False otherwise.
        
    Raises
    ------
    TypeError
        If ridge is not a Ridge instance.
    ValueError
        If any threshold parameter is invalid (negative when should be non-negative).
        
    Notes
    -----
    All criteria must be satisfied (AND logic):
    - ridge.length >= scale_length_thr
    - ridge.max_scale >= max_scale_thr
    - ridge.max_ampl >= max_ampl_thr
    - ridge.duration <= max_dur_thr
    
    Typical calcium transients have specific scale and duration
    characteristics that distinguish them from noise or artifacts.
    
    DOC_VERIFIED
    """
    # Validate parameters
    check_nonnegative(scale_length_thr=scale_length_thr, max_scale_thr=max_scale_thr, max_ampl_thr=max_ampl_thr)
    check_positive(max_dur_thr=max_dur_thr)
    
    # Validate ridge type
    if not isinstance(ridge, Ridge):
        raise TypeError(f"ridge must be Ridge instance, got {type(ridge)}")
    
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
    """Extract event start/end times from ridges that pass quality criteria.
    
    Filters a list of detected ridges based on quality criteria and extracts
    the temporal boundaries (start and end indices) of events that pass.
    
    Parameters
    ----------
    all_ridges : list of Ridge
        All detected ridges from wavelet transform analysis.
    scale_length_thr : int, default=40
        Minimum ridge length in scales. See passing_criterion().
    max_scale_thr : int, default=10
        Minimum maximum scale. See passing_criterion().
    max_ampl_thr : float, default=0.05
        Minimum maximum amplitude. See passing_criterion().
    max_dur_thr : int, default=100
        Maximum duration in frames. See passing_criterion().
        
    Returns
    -------
    st_evinds : list of int
        Start indices (frame numbers) of events that pass criteria.
    end_evinds : list of int
        End indices (frame numbers) of events that pass criteria.
        Paired with st_evinds (same length).
        
    Raises
    ------
    TypeError
        If all_ridges is not iterable.
        
    See Also
    --------
    passing_criterion : Function that evaluates ridge quality
    events_from_trace : High-level event detection function
    
    DOC_VERIFIED
    """
    # Validate parameters
    check_nonnegative(scale_length_thr=scale_length_thr, max_scale_thr=max_scale_thr, max_ampl_thr=max_ampl_thr)
    check_positive(max_dur_thr=max_dur_thr)
    
    # Validate ridges
    try:
        iter(all_ridges)
    except TypeError:
        raise TypeError(f"all_ridges must be iterable, got {type(all_ridges)}")
    
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
    """Detect calcium transient events in a single trace using wavelet ridges.
    
    Complete pipeline for calcium event detection: normalizes signal, applies
    Gaussian smoothing, computes wavelet transform, finds ridges, and filters
    them to identify significant calcium transients.
    
    Parameters
    ----------
    trace : array-like
        Raw calcium signal trace (e.g., ΔF/F or raw fluorescence).
    wavelet : ssqueezepy.wavelets.Wavelet
        Wavelet object (typically Generalized Morse Wavelet).
    manual_scales : array-like
        Wavelet scales to use for CWT. Should span expected event durations.
    rel_wvt_times : array-like
        Pre-computed relative wavelet time resolutions for each scale.
    fps : float, default=20
        Sampling rate in Hz.
    sigma : float, default=8
        Gaussian smoothing sigma in frames. Reduces noise before CWT.
    eps : int, default=10
        Minimum spacing between peaks (order parameter for argrelmax).
        Prevents detecting multiple peaks too close together.
    scale_length_thr : int, default=40
        Minimum ridge length. See passing_criterion().
    max_scale_thr : int, default=7
        Minimum maximum scale. See passing_criterion().
    max_ampl_thr : float, default=0.05
        Minimum amplitude threshold. See passing_criterion().
    max_dur_thr : int, default=200
        Maximum duration in frames. See passing_criterion().
        
    Returns
    -------
    all_ridges : list of Ridge
        All detected ridges (before filtering).
    st_evinds : list of int
        Start indices of detected events.
    end_evinds : list of int
        End indices of detected events.
        
    Notes
    -----
    Processing steps:
    1. Normalize trace to [0, 1] range
    2. Apply Gaussian smoothing to reduce noise
    3. Compute continuous wavelet transform
    4. Find local maxima across scales
    5. Track ridges connecting maxima
    6. Filter ridges based on quality criteria
    
    The default parameters are tuned for typical calcium imaging at 20 Hz
    with GCaMP-like indicators. May need adjustment for different indicators
    or sampling rates.
    
    Raises
    ------
    ValueError
        If trace is empty, not 1D, or has zero range (constant signal).
        If parameters are invalid or arrays have mismatched lengths.
    TypeError
        If wavelet is not a Wavelet instance.
        
    See Also
    --------
    extract_wvt_events : Batch processing for multiple neurons
    WVT_EVENT_DETECTION_PARAMS : Default parameter dictionary
    
    DOC_VERIFIED
    """
    # Validate parameters
    check_positive(fps=fps, max_dur_thr=max_dur_thr)
    check_nonnegative(sigma=sigma, eps=eps, scale_length_thr=scale_length_thr, 
                     max_scale_thr=max_scale_thr, max_ampl_thr=max_ampl_thr)
    
    # Validate trace
    trace = np.asarray(trace)
    if trace.ndim != 1:
        raise ValueError(f"trace must be 1D, got shape {trace.shape}")
    if trace.size == 0:
        raise ValueError("trace cannot be empty")
        
    # Validate wavelet
    if not isinstance(wavelet, Wavelet):
        raise TypeError(f"wavelet must be Wavelet instance, got {type(wavelet)}")
        
    # Validate scales and times
    manual_scales = np.asarray(manual_scales)
    rel_wvt_times = np.asarray(rel_wvt_times)
    if len(manual_scales) != len(rel_wvt_times):
        raise ValueError(f"manual_scales and rel_wvt_times must have same length, got {len(manual_scales)} and {len(rel_wvt_times)}")
    
    # Normalize trace with range check
    trace_min, trace_max = trace.min(), trace.max()
    if trace_max - trace_min == 0:
        raise ValueError("trace has zero range (constant signal)")
    trace = (trace - trace_min) / (trace_max - trace_min)
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
    """Extract calcium events from multiple traces using wavelet ridge detection.
    
    Detects calcium transient events by finding ridges in the continuous wavelet
    transform (CWT) of calcium signals. Uses Generalized Morse Wavelets and
    ridge filtering to identify significant events.
    
    Parameters
    ----------
    traces : ndarray
        2D array of calcium traces (neurons x time).
    wvt_kwargs : dict
        Wavelet detection parameters:
        - fps : float, frame rate in Hz (default: 20)
        - beta : float, GMW beta parameter (default: 2)
        - gamma : float, GMW gamma parameter (default: 3)
        - sigma : float, Gaussian smoothing sigma in frames (default: 8)
        - eps : int, minimum spacing between events in frames (default: 10)
        - manual_scales : array, wavelet scales to use
        - scale_length_thr : int, minimum ridge length (default: 40)
        - max_scale_thr : int, max scale index threshold (default: 7)
        - max_ampl_thr : float, minimum ridge amplitude (default: 0.05)
        - max_dur_thr : int, maximum event duration in frames (default: 200)
        
    Returns
    -------
    st_ev_inds : list of lists
        Start indices for each detected event per neuron.
    end_ev_inds : list of lists
        End indices for each detected event per neuron.
    all_ridges : list of lists
        Ridge objects containing detailed event information per neuron.
        
    Raises
    ------
    ValueError
        If traces is not 2D or empty.
    TypeError
        If wvt_kwargs is not a dictionary.
        
    Notes
    -----
    The algorithm:
    1. Smooths traces with Gaussian filter
    2. Computes CWT using Generalized Morse Wavelets
    3. Detects ridges (connected paths through scale-time plane)
    4. Filters ridges based on length, amplitude, and duration criteria
    5. Returns event start/end times
    
    Ridge filtering removes noise and artifacts by requiring events to:
    - Persist across multiple scales (scale_length_thr)
    - Have sufficient amplitude (max_ampl_thr)
    - Have reasonable duration (max_dur_thr)
    
    DOC_VERIFIED
    """
    # Validate inputs
    traces = np.asarray(traces)
    if traces.ndim != 2:
        raise ValueError(f"traces must be 2D (neurons x time), got shape {traces.shape}")
    if traces.shape[1] == 0:
        raise ValueError("traces cannot be empty (no time points)")
    
    if not isinstance(wvt_kwargs, dict):
        raise TypeError(f"wvt_kwargs must be dict, got {type(wvt_kwargs)}")
        
    # Extract parameters with validation
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
    
    # Validate extracted parameters
    check_positive(fps=fps, beta=beta, gamma=gamma, max_dur_thr=max_dur_thr)
    check_nonnegative(sigma=sigma, eps=eps, scale_length_thr=scale_length_thr,
                     max_scale_thr=max_scale_thr, max_ampl_thr=max_ampl_thr)

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


@conditional_njit
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
    """Numba-optimized version of events_to_ts_array.
    
    Low-level implementation for converting event indices to binary time series.
    Uses flattened arrays for compatibility with Numba JIT compilation.
    
    Parameters
    ----------
    length : int
        Length of output time series in frames.
    ncells : int
        Number of neurons/cells.
    st_ev_inds_flat : np.ndarray of shape (total_events,)
        Flattened array of all event start indices.
    end_ev_inds_flat : np.ndarray of shape (total_events,)
        Flattened array of all event end indices.
    event_counts : np.ndarray
        Number of events per neuron, shape (ncells,).
    fps : float
        Frames per second of the recording.
    min_event_dur : float
        Minimum event duration in seconds.
    max_event_dur : float
        Maximum event duration in seconds.
        
    Returns
    -------
    np.ndarray
        Binary array of shape (ncells, length) where 1 indicates active event.
        
    Raises
    ------
    ValueError
        If parameters are invalid or array lengths don't match.
        
    Notes
    -----
    Called internally by events_to_ts_array. Event duration constraints are
    enforced as described in the parent function.
    
    DOC_VERIFIED
    """
    # Note: Parameter validation is done in the wrapper function
    # Cannot use check_positive inside numba-compiled functions
    
    # Validate arrays
    st_ev_inds_flat = np.asarray(st_ev_inds_flat)
    end_ev_inds_flat = np.asarray(end_ev_inds_flat)
    event_counts = np.asarray(event_counts)
    
    if len(st_ev_inds_flat) != len(end_ev_inds_flat):
        raise ValueError(f"st_ev_inds_flat and end_ev_inds_flat must have same length, got {len(st_ev_inds_flat)} and {len(end_ev_inds_flat)}")
    
    if len(event_counts) != ncells:
        raise ValueError(f"event_counts length ({len(event_counts)}) must equal ncells ({ncells})")
        
    if np.sum(event_counts) != len(st_ev_inds_flat):
        raise ValueError(f"Sum of event_counts ({np.sum(event_counts)}) must equal length of flattened arrays ({len(st_ev_inds_flat)})")
    
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
                # Ensure indices stay within bounds
                start_idx = max(0, int(middle - mindur // 2))
                end_idx = min(length, int(middle + mindur // 2))
                spikes[i, start_idx:end_idx] = 1
            event_idx += 1

    return spikes


def events_to_ts_array(length, st_ev_inds, end_ev_inds, fps):
    """Convert event indices to binary time series array.
    
    Transforms lists of event start/end indices into a binary array where
    1 indicates an active event and 0 indicates no event. Events are
    constrained to reasonable durations based on calcium dynamics.
    
    Parameters
    ----------
    length : int
        Length of the output time series in frames.
    st_ev_inds : list of lists
        Start indices for events. Each sublist contains events for one neuron.
    end_ev_inds : list of lists
        End indices for events. Each sublist contains events for one neuron.
    fps : float
        Frame rate in Hz, used to convert duration constraints to frames.
        
    Returns
    -------
    ndarray
        Binary array of shape (n_neurons, length) where 1 indicates an event.
        
    Raises
    ------
    ValueError
        If length is not positive, fps is not positive, or st_ev_inds and 
        end_ev_inds have different structures.
        
    Notes
    -----
    Events are adjusted to have durations between MIN_EVENT_DUR (0.5s) and
    MAX_EVENT_DUR (2.5s). Events shorter than minimum are extended from their
    center, while events longer than maximum are truncated.
    
    DOC_VERIFIED
    """
    # Validate parameters
    from ..utils.data import check_positive
    check_positive(length=length, fps=fps)
    
    # Validate structure
    if not isinstance(st_ev_inds, list) or not isinstance(end_ev_inds, list):
        raise TypeError("st_ev_inds and end_ev_inds must be lists")
        
    if len(st_ev_inds) != len(end_ev_inds):
        raise ValueError(f"st_ev_inds and end_ev_inds must have same length, got {len(st_ev_inds)} and {len(end_ev_inds)}")
        
    ncells = len(end_ev_inds)
    if ncells == 0:
        raise ValueError("Must have at least one neuron")

    # Check that sublists have matching lengths
    for i in range(ncells):
        if len(st_ev_inds[i]) != len(end_ev_inds[i]):
            raise ValueError(f"Neuron {i}: st_ev_inds and end_ev_inds sublists must have same length, got {len(st_ev_inds[i])} and {len(end_ev_inds[i])}")
    
    # Flatten the jagged arrays for numba
    event_counts = np.array([len(st_ev_inds[i]) for i in range(ncells)])
    
    # Handle case where some neurons have no events
    if np.sum(event_counts) == 0:
        # No events, return zeros
        return np.zeros((ncells, length))
    
    st_ev_inds_flat = np.concatenate([st_ev_inds[i] for i in range(ncells) if len(st_ev_inds[i]) > 0])
    end_ev_inds_flat = np.concatenate([end_ev_inds[i] for i in range(ncells) if len(end_ev_inds[i]) > 0])

    # Validate parameters before calling numba function
    check_positive(length=length, ncells=ncells, fps=fps, 
                  min_event_dur=MIN_EVENT_DUR, max_event_dur=MAX_EVENT_DUR)

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
