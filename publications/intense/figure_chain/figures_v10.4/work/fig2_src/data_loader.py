"""Data loader for Figure 2: INTENSE vs classic place cell comparison.

Loads classic algorithm results from .mat file and INTENSE results
from IntenseResults .npz files, then computes overlap metrics across
p-value thresholds.
"""

import numpy as np
import scipy.io as sio
import scipy.stats
from pathlib import Path


# Paths
FIG2_DIR = Path(__file__).parent
# Classic results and the per-session SSIM caches are read from the repository copy (read-only)
import sys as _sys  # noqa: E402
_sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # figure_chain/
import _paths_fig as _F  # noqa: E402
MAT_FILE = _F.P.CLASSIC_MAT
RESULTS_DIR = _F.V.get('stage2_results')
CACHE_DIR = _F.data_dir(FIG2_DIR) / 'cache'

# Session list (64 sessions: 16 animals x 4 days) — order matches .mat file
SESSION_LIST = [
    'H01_1D', 'H02_1D', 'H03_1D', 'H06_1D', 'H07_1D', 'H08_1D',
    'H09_1D', 'H14_1D', 'H23_1D', 'H26_1D', 'H27_1D', 'H31_1D',
    'H32_1D', 'H33_1D', 'H36_1D', 'H39_1D',
    'H01_2D', 'H02_2D', 'H03_2D', 'H06_2D', 'H07_2D', 'H08_2D',
    'H09_2D', 'H14_2D', 'H23_2D', 'H26_2D', 'H27_2D', 'H31_2D',
    'H32_2D', 'H33_2D', 'H36_2D', 'H39_2D',
    'H01_3D', 'H02_3D', 'H03_3D', 'H06_3D', 'H07_3D', 'H08_3D',
    'H09_3D', 'H14_3D', 'H23_3D', 'H26_3D', 'H27_3D', 'H31_3D',
    'H32_3D', 'H33_3D', 'H36_3D', 'H39_3D',
    'H01_4D', 'H02_4D', 'H03_4D', 'H06_4D', 'H07_4D', 'H08_4D',
    'H09_4D', 'H14_4D', 'H23_4D', 'H26_4D', 'H27_4D', 'H31_4D',
    'H32_4D', 'H33_4D', 'H36_4D', 'H39_4D',
]


def load_classic_data(zthr=3):
    """Load classic place cell results from .mat file.

    Parameters
    ----------
    zthr : float, default 3
        Z-score threshold for recomputing significance.

    Returns
    -------
    dict[str, dict]
        Per-session classic stats keyed by session name (e.g. 'H01_1D').
        Each value is a dict with keys: significant, MI, mean_MI_shuffle,
        std_MI_shuffle, z_score, n_spikes, significant_recomputed, p_val_computed.
    """
    classic = sio.loadmat(str(MAT_FILE))
    cells_info = classic['cells_info']

    all_classic_info = {}
    for session in SESSION_LIST:
        mat_key = f'NOF_{session}'
        datapiece = cells_info[mat_key][0, 0]

        significant = datapiece[1, :].copy()
        significant[significant == -1] = 0

        stats = {
            'significant': significant,
            'MI': datapiece[2, :],
            'mean_MI_shuffle': datapiece[3, :],
            'std_MI_shuffle': datapiece[4, :],
            'z_score': datapiece[5, :],
            'n_spikes': datapiece[6, :],
            'significant_recomputed': (datapiece[5, :] > zthr).astype(int),
            'p_val_computed': scipy.stats.norm.sf(np.abs(datapiece[5, :])),
        }
        all_classic_info[session] = stats

    return all_classic_info


def load_intense_data(feature='place'):
    """Load INTENSE place cell results from IntenseResults .npz files.

    Uses populate_stage1_pvals() to reconstruct p-values for ALL neurons
    (including non-significant ones) from saved shuffle distributions.

    Parameters
    ----------
    feature : str, default 'place'
        Feature name to extract.

    Returns
    -------
    dict[str, dict]
        Per-session INTENSE stats keyed by session name (e.g. 'H01_1D').
        Each value is a dict with keys: significant, MI, p_val, r_val.
    """
    from driada.intense import load_results

    all_intense_info = {}

    for session in SESSION_LIST:
        expname = f'NOF_{session}'
        results = load_results(RESULTS_DIR / f'{expname}_results.npz')
        results.populate_stage1_pvals()

        all_neuron_names = list(results.intense_params['neurons'].values())
        # Skip neuron 0 to align with classic .mat data (1-based neuron indices)
        neuron_names = all_neuron_names[1:]
        nn = len(neuron_names)

        significant = np.zeros(nn)
        pvals = np.ones(nn)
        mi = np.zeros(nn)
        rvals = np.zeros(nn)

        for i, neuron_id in enumerate(neuron_names):
            nk = str(neuron_id)
            stats_data = results.stats.get(nk, {}).get(feature, {})
            sign_data = results.significance.get(nk, {}).get(feature, {})

            if not stats_data:
                continue

            if sign_data.get('stage1', False) and sign_data.get('stage2', False):
                significant[i] = 1
                pvals[i] = stats_data.get('pval', 1.0) or 1.0
                rvals[i] = stats_data.get('rval', 0) or 0
            else:
                pvals[i] = stats_data.get('pre_pval', 1.0) or 1.0
                rvals[i] = stats_data.get('pre_rval', 0) or 0

            mi[i] = stats_data.get('me', 0) or 0

        all_intense_info[session] = {
            'significant': significant,
            'MI': mi,
            'p_val': pvals,
            'r_val': rvals,
        }

    return all_intense_info


def merge_session_stats(all_info, session_order, stat_keys):
    """Merge per-session stats into single arrays across all neurons.

    Parameters
    ----------
    all_info : dict[str, dict]
        Per-session stats from load_classic_data or load_intense_data.
    session_order : list of str
        Session names in merge order.
    stat_keys : list of str
        Stat keys to merge.

    Returns
    -------
    dict[str, np.ndarray]
        Merged arrays keyed by stat name.
    """
    merged = {key: [] for key in stat_keys}
    for key in stat_keys:
        total = []
        for session in session_order:
            total.extend(all_info[session][key])
        merged[key] = np.array(total)
    return merged


def compute_overlap_metrics(merged_intense, merged_classic, p_threshold):
    """Compute overlap metrics between INTENSE and classic place cells.

    Parameters
    ----------
    merged_intense : dict
        Merged INTENSE stats with keys: significant, MI, p_val, r_val.
    merged_classic : dict
        Merged classic stats with keys: significant_recomputed, p_val_computed, etc.
    p_threshold : float
        P-value threshold for filtering neurons.

    Returns
    -------
    pct_intense : float
        % of INTENSE PC that are also classic PC.
    pct_classic : float
        % of classic PC that are also INTENSE PC.
    jaccard : float
        Jaccard coefficient (%) between the two sets.
    pct_remaining : float
        % of all jointly-significant PC remaining at this threshold
        relative to no p-value filtering.
    """
    vals_intense = merged_intense['p_val']
    vals_classic = merged_classic['p_val_computed']

    good = (vals_intense < p_threshold) & (vals_classic < p_threshold)

    flags_intense = merged_intense['significant'][good]
    flags_classic = merged_classic['significant_recomputed'][good]
    flags_both = flags_classic * flags_intense

    n_intense = np.sum(flags_intense)
    n_classic = np.sum(flags_classic)
    n_both = np.sum(flags_both)

    pct_intense = (n_both / n_intense * 100) if n_intense > 0 else 0.0
    pct_classic = (n_both / n_classic * 100) if n_classic > 0 else 0.0
    jaccard = (n_both / (n_intense + n_classic - n_both) * 100) if (n_intense + n_classic - n_both) > 0 else 0.0

    # % of all PC remaining relative to unfiltered baseline
    flags_intense_all = merged_intense['significant']
    flags_classic_all = merged_classic['significant_recomputed']
    n_both_all = np.sum(flags_classic_all * flags_intense_all)
    pct_remaining = (n_both / n_both_all * 100) if n_both_all > 0 else 0.0

    return (
        np.round(pct_intense, 2),
        np.round(pct_classic, 2),
        np.round(jaccard, 2),
        pct_remaining,
    )


def compute_metrics_over_prange(merged_intense=None, merged_classic=None,
                                prange=None, zthr=3, feature='place'):
    """Compute overlap metrics across a range of p-value thresholds.

    If merged stats are not provided, loads and merges data from files.

    Parameters
    ----------
    merged_intense : dict, optional
        Pre-merged INTENSE stats.
    merged_classic : dict, optional
        Pre-merged classic stats.
    prange : array-like, optional
        P-value thresholds to sweep. Default: np.logspace(0, -5, 50).
    zthr : float, default 3
        Z-score threshold for classic recomputation.
    feature : str, default 'place'
        INTENSE feature name.

    Returns
    -------
    xvals : np.ndarray
        -log10(p) values for the x-axis.
    pct_intense : np.ndarray
        % of INTENSE PC at each threshold.
    pct_classic : np.ndarray
        % of classic PC at each threshold.
    jaccard : np.ndarray
        Jaccard coefficient at each threshold.
    pct_remaining : np.ndarray
        % of all PC remaining at each threshold.
    """
    if merged_intense is None or merged_classic is None:
        all_classic = load_classic_data(zthr=zthr)
        all_intense = load_intense_data(feature=feature)

        merged_classic = merge_session_stats(
            all_classic, SESSION_LIST,
            ['significant', 'significant_recomputed', 'MI',
             'mean_MI_shuffle', 'std_MI_shuffle', 'z_score',
             'n_spikes', 'p_val_computed'],
        )
        merged_intense = merge_session_stats(
            all_intense, SESSION_LIST,
            ['significant', 'MI', 'p_val', 'r_val'],
        )

    if prange is None:
        prange = np.logspace(0, -5, 50)

    metr_intense = []
    metr_classic = []
    metr_jaccard = []
    metr_remaining = []

    for p in prange:
        m1, m2, m3, rs = compute_overlap_metrics(merged_intense, merged_classic, p)
        metr_intense.append(m1)
        metr_classic.append(m2)
        metr_jaccard.append(m3)
        metr_remaining.append(rs)

    xvals = np.linspace(0, 5, len(prange))

    return (
        xvals,
        np.array(metr_intense),
        np.array(metr_classic),
        np.array(metr_jaccard),
        np.array(metr_remaining),
    )


# =============================================================================
# Cache helpers
# =============================================================================

def compute_fig2a_data(animal_id='H31', day='1D', neuron_index=60,
                       fps=20, sigma=8, event_threshold=0.2):
    """Load experiment and compute wavelet transform for Panel A.

    Parameters
    ----------
    animal_id : str
        Animal identifier.
    day : str
        Experimental day.
    neuron_index : int
        Neuron index to visualize.
    fps : int
        Sampling rate in Hz.
    sigma : float
        Gaussian smoothing sigma.
    event_threshold : float
        Threshold for reconstructed events.

    Returns
    -------
    sig : np.ndarray
        Gaussian-smoothed calcium trace.
    sp : np.ndarray
        Binary event series (1 where sig >= threshold).
    Wx : np.ndarray
        CWT magnitude (|W|), shape (n_scales, n_frames).
    wvt_scales : np.ndarray
        Wavelet scales (frequencies in Hz).
    """
    from science.intense_figures.fig1.data_loader import load_nof_experiment
    from scipy.ndimage import gaussian_filter1d

    # scipy compat fix for ssqueezepy
    import scipy.integrate
    if not hasattr(scipy.integrate, 'trapz'):
        scipy.integrate.trapz = scipy.integrate.trapezoid

    from ssqueezepy import cwt
    from ssqueezepy.wavelets import Wavelet

    Exp = load_nof_experiment(animal_id=animal_id, day=day)
    trace = Exp.neurons[neuron_index].ca.scdata

    sig = gaussian_filter1d(trace, sigma=sigma)
    sp = np.zeros(len(trace))
    sp[np.where(sig >= event_threshold)[0]] = 1

    wavelet = Wavelet(('gmw', {'gamma': 3, 'beta': 2, 'centered_scale': True}),
                      N=8196)
    W, wvt_scales = cwt(sig, wavelet=wavelet, fs=fps)

    return sig, sp, np.abs(W), wvt_scales


def save_fig2a_cache(sig, sp, Wx, wvt_scales):
    """Save Panel A data to cache for fast figure regeneration."""
    CACHE_DIR.mkdir(exist_ok=True)
    np.savez_compressed(
        CACHE_DIR / 'fig2a_data.npz',
        sig=sig,
        sp=sp,
        Wx=Wx.astype(np.float32),
        wvt_scales=wvt_scales,
    )


def load_fig2a_cache():
    """Load cached Panel A data.

    Returns
    -------
    dict with keys: sig, sp, Wx, wvt_scales.

    Raises
    ------
    FileNotFoundError
        If cache doesn't exist.
    """
    cache_path = CACHE_DIR / 'fig2a_data.npz'
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cache not found: {cache_path}\n"
            "Run: python -m science.intense_figures.fig2.fig2a_wavelets --cache"
        )
    return dict(np.load(cache_path))


def compute_fig2b_data(animal_id='H31', day='1D', cell_ind=931, ds=10):
    """Load experiment and compute trajectory plot data for Panel B.

    Parameters
    ----------
    animal_id : str
        Animal identifier.
    day : str
        Experimental day.
    cell_ind : int
        Neuron index to visualize.
    ds : int
        Downsample factor for trajectory.

    Returns
    -------
    x_ds : np.ndarray
        Downsampled x positions.
    y_ds : np.ndarray
        Downsampled y positions.
    neur_ds : np.ndarray
        Rescaled log-calcium values (downsampled, len = len(x_ds) - 1).
    spike_x : np.ndarray
        X positions of spike events.
    spike_y : np.ndarray
        Y positions of spike events.
    """
    from science.intense_figures.fig1.data_loader import load_nof_experiment
    from driada.utils.data import rescale

    Exp = load_nof_experiment(animal_id=animal_id, day=day,
                              reconstruct_spikes='wavelet')

    neur = rescale(np.log(Exp.neurons[cell_ind].ca.data + 1e-10 + 0.05))
    spinds = np.where(Exp.neurons[cell_ind].sp.data != 0)[0]

    x_ds = Exp.x.data[::ds]
    y_ds = Exp.y.data[::ds]
    neur_ds = neur[::ds][:-1]  # one fewer than positions (segment colors)
    spike_x = Exp.x.data[spinds][::10]
    spike_y = Exp.y.data[spinds][::10]

    return x_ds, y_ds, neur_ds, spike_x, spike_y


def save_fig2b_cache(x_ds, y_ds, neur_ds, spike_x, spike_y):
    """Save Panel B data to cache."""
    CACHE_DIR.mkdir(exist_ok=True)
    np.savez(
        CACHE_DIR / 'fig2b_data.npz',
        x_ds=x_ds,
        y_ds=y_ds,
        neur_ds=neur_ds,
        spike_x=spike_x,
        spike_y=spike_y,
    )


def load_fig2b_cache():
    """Load cached Panel B data.

    Returns
    -------
    dict with keys: x_ds, y_ds, neur_ds, spike_x, spike_y.

    Raises
    ------
    FileNotFoundError
        If cache doesn't exist.
    """
    cache_path = CACHE_DIR / 'fig2b_data.npz'
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cache not found: {cache_path}\n"
            "Run: python -m science.intense_figures.fig2.fig2b_trajectory --cache"
        )
    return dict(np.load(cache_path))


def save_fig2f_cache(xvals, pct_intense, pct_classic, jaccard, pct_remaining):
    """Save Panel D data to cache for fast figure regeneration."""
    CACHE_DIR.mkdir(exist_ok=True)
    np.savez(
        CACHE_DIR / 'fig2f_data.npz',
        xvals=xvals,
        pct_intense=pct_intense,
        pct_classic=pct_classic,
        jaccard=jaccard,
        pct_remaining=pct_remaining,
    )


def load_fig2f_cache():
    """Load cached Panel D data.

    Returns
    -------
    dict with keys: xvals, pct_intense, pct_classic, jaccard, pct_remaining.

    Raises
    ------
    FileNotFoundError
        If cache doesn't exist.
    """
    cache_path = CACHE_DIR / 'fig2f_data.npz'
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cache not found: {cache_path}\n"
            "Run: python -m science.intense_figures.fig2.data_loader"
        )
    return dict(np.load(cache_path))


# =============================================================================
# Figure 2E: Place field metrics (MSE, PSNR, SSIM)
# =============================================================================

def threshold_signal(signal, thr, mode='percentile'):
    """Threshold a signal using various methods.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    thr : float
        Threshold value (meaning depends on mode).
    mode : str, default 'percentile'
        Thresholding mode: 'simple', 'percentile', or 'mad'.

    Returns
    -------
    np.ndarray
        Thresholded signal (values below threshold set to 0).
    """
    if mode == 'simple':
        return np.where(signal > thr, signal, 0)
    elif mode == 'percentile':
        return np.where(signal > np.percentile(signal, thr), signal, 0)
    elif mode == 'mad':
        median = np.median(signal)
        mad = np.median(np.abs(signal - median))
        threshold = median + thr * mad
        return np.where(signal > threshold, signal, 0)
    return signal


def compute_activity_map(x_coords, y_coords, activity, n=8, ds=1,
                         smooth_sigma=1.0):
    """Compute average activity map on an n x n spatial grid.

    Parameters
    ----------
    x_coords, y_coords : np.ndarray
        Animal position coordinates.
    activity : np.ndarray
        Neural activity values (same length as coords).
    n : int, default 8
        Grid resolution (n x n bins).
    ds : int, default 1
        Downsample factor.
    smooth_sigma : float or None, default 1.0
        Gaussian smoothing sigma (in bins). None disables smoothing.

    Returns
    -------
    np.ndarray
        Average activity map of shape (n, n).
    """
    from scipy.ndimage import gaussian_filter

    x_coords = x_coords[::ds]
    y_coords = y_coords[::ds]
    activity = activity[::ds]

    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)

    area_width = (max_x - min_x) / n
    area_height = (max_y - min_y) / n

    occupancy_grid = np.zeros((n, n), dtype=int)
    activity_grid = np.zeros((n, n), dtype=float)

    for x, y, act in zip(x_coords, y_coords, activity):
        area_x = int((x - min_x) / area_width)
        area_y = int((y - min_y) / area_height)
        if 0 <= area_x < n and 0 <= area_y < n:
            occupancy_grid[n - area_y - 1, area_x] += 1
            activity_grid[n - area_y - 1, area_x] += act

    rate_map = np.where(occupancy_grid > 0, activity_grid / occupancy_grid, 0)

    if smooth_sigma is not None and smooth_sigma > 0:
        rate_map = gaussian_filter(rate_map, sigma=smooth_sigma)

    return rate_map


def calculate_act_maps_sim(activity_map1, activity_map2):
    """Compute similarity metrics between two activity maps.

    Both maps are normalized to [0, 1] before comparison.

    Parameters
    ----------
    activity_map1, activity_map2 : np.ndarray
        2D activity maps of the same shape.

    Returns
    -------
    tuple of (mse, psnr, ssim)
        Mean squared error, peak signal-to-noise ratio,
        structural similarity index.
    """
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import mean_squared_error, peak_signal_noise_ratio

    def normalize_image(img):
        img = np.array(img, dtype=float)
        rng = np.max(img) - np.min(img)
        if rng == 0:
            return np.zeros_like(img)
        return (img - np.min(img)) / rng

    map1 = normalize_image(activity_map1)
    map2 = normalize_image(activity_map2)

    mse_value = mean_squared_error(map1, map2)
    psnr_value = peak_signal_noise_ratio(map1, map2, data_range=1.0)
    ssim_value = ssim(map1, map2, data_range=1.0)

    return mse_value, psnr_value, ssim_value


def compute_fig2e_session_data(session, n=8, ds=1, thr=95):
    """Compute place field metrics for all neurons in one session.

    For each neuron, compares calcium-based and spike-based activity maps
    using MSE, PSNR, and SSIM.

    Parameters
    ----------
    session : str
        Session name (e.g. 'H01_1D').
    n : int, default 8
        Grid resolution for activity maps.
    ds : int, default 1
        Downsample factor for positions.
    thr : float, default 95
        Percentile threshold for calcium signal.

    Returns
    -------
    ssim_vals : np.ndarray
        Shape (n_neurons, 3) with columns [MSE, PSNR, SSIM].
        Neuron 0 is skipped for alignment with classic data.
    """
    from science.intense_figures.fig1.data_loader import load_nof_experiment

    animal_id, day = session.split('_')

    try:
        Exp = load_nof_experiment(animal_id=animal_id, day=day,
                                  reconstruct_spikes='wavelet')
    except Exception as e:
        print(f"  Wavelet reconstruction failed for {session}: {e}")
        print(f"  Falling back to threshold reconstruction...")
        Exp = load_nof_experiment(animal_id=animal_id, day=day,
                                  reconstruct_spikes='threshold')

    n_neurons = len(Exp.neurons) - 1  # skip neuron 0
    ssim_vals = np.full((n_neurons, 3), np.nan)

    x_coords = Exp.x.data
    y_coords = Exp.y.data

    for i in range(n_neurons):
        nind = i + 1  # skip neuron 0
        neuron = Exp.neurons[nind]

        if neuron.sp.data is None:
            print(f"  Warning: sp.data is None for neuron {nind} in {session}")
            continue

        ca_signal = threshold_signal(neuron.ca.scdata, thr, mode='percentile')
        sp_signal = neuron.sp.scdata

        ca_map = compute_activity_map(x_coords, y_coords, ca_signal, n=n, ds=ds)
        sp_map = compute_activity_map(x_coords, y_coords, sp_signal, n=n, ds=ds)

        ssim_vals[i] = calculate_act_maps_sim(ca_map, sp_map)

    return ssim_vals


def merge_fig2e_data(session_list=None):
    """Merge per-session fig2e metrics into a single array.

    Parameters
    ----------
    session_list : list of str, optional
        Sessions to merge. Default: SESSION_LIST.

    Returns
    -------
    all_ssim_vals : np.ndarray
        Shape (n_total, 3) with columns [MSE, PSNR, SSIM].
    """
    if session_list is None:
        session_list = SESSION_LIST

    arrays = []
    for session in session_list:
        cache = load_fig2e_session_cache(session)
        arrays.append(cache['ssim_vals'])

    return np.concatenate(arrays, axis=0)


def compute_fig2e_curves(all_ssim_vals, merged_intense_info, merged_classic_info):
    """Compute MSE/PSNR/SSIM curves over p-value thresholds.

    Parameters
    ----------
    all_ssim_vals : np.ndarray
        Shape (n_total, 3) with columns [MSE, PSNR, SSIM].
    merged_intense_info : dict
        Merged INTENSE stats with keys: significant, p_val.
    merged_classic_info : dict
        Merged classic stats with keys: significant_recomputed, p_val_computed.

    Returns
    -------
    xvals : np.ndarray
        -log10(p) values for x-axis (0 to 7).
    av_mse, err_mse : np.ndarray
        Mean and SEM of MSE at each threshold.
    av_psnr, err_psnr : np.ndarray
        Mean and SEM of PSNR.
    av_ssim, err_ssim : np.ndarray
        Mean and SEM of SSIM.
    """
    prange = np.logspace(-1, -7, 50)
    xvals = np.linspace(1, 7, 50)

    n = len(all_ssim_vals)
    vals_intense = np.array(merged_intense_info['p_val'][:n])
    vals_classic = np.array(merged_classic_info['p_val_computed'][:n])

    av_list = []
    err_list = []

    for p in prange:
        good = (vals_intense < p) & (vals_classic < p)

        sims = all_ssim_vals[good]
        # Filter out NaN/inf values
        if len(sims) > 0:
            valid = np.all(np.isfinite(sims), axis=1)
            sims = sims[valid]

        if len(sims) > 0:
            av_sim = np.mean(sims, axis=0)
            err_sim = np.std(sims, axis=0) / np.sqrt(len(sims))
        else:
            av_sim = np.full(3, np.nan)
            err_sim = np.full(3, np.nan)

        av_list.append(av_sim)
        err_list.append(err_sim)

    av = np.array(av_list)
    err = np.array(err_list)

    return xvals, av[:, 0], err[:, 0], av[:, 1], err[:, 1], av[:, 2], err[:, 2]


# Cache helpers for Figure 2C

def compute_fig2d_data(session='H01_1D', neuron_idx=97, n=8, ds=1, thr=95,
                       smooth_sigma=None):
    """Compute activity maps for a single neuron (fig2d example).

    Parameters
    ----------
    session : str
        Session name.
    neuron_idx : int
        Neuron index (1-based, 0 is skipped).
    n : int
        Grid resolution.
    ds : int
        Downsample factor.
    thr : float
        Percentile threshold for calcium.
    smooth_sigma : float or None
        Gaussian smoothing sigma. None disables smoothing.

    Returns
    -------
    ca_map, sp_map : np.ndarray
        Activity maps of shape (n, n).
    """
    from science.intense_figures.fig1.data_loader import load_nof_experiment

    animal_id, day = session.split('_')
    Exp = load_nof_experiment(animal_id=animal_id, day=day,
                              reconstruct_spikes='wavelet')

    neuron = Exp.neurons[neuron_idx]
    x_coords = Exp.x.data
    y_coords = Exp.y.data

    ca_signal = threshold_signal(neuron.ca.scdata, thr, mode='percentile')
    sp_signal = neuron.sp.scdata

    ca_map = compute_activity_map(x_coords, y_coords, ca_signal, n=n, ds=ds,
                                  smooth_sigma=smooth_sigma)
    sp_map = compute_activity_map(x_coords, y_coords, sp_signal, n=n, ds=ds,
                                  smooth_sigma=smooth_sigma)

    return ca_map, sp_map


def save_fig2d_cache(ca_map, sp_map):
    """Save fig2d activity maps to cache."""
    CACHE_DIR.mkdir(exist_ok=True)
    np.savez_compressed(
        CACHE_DIR / 'fig2d_data.npz',
        ca_map=ca_map,
        sp_map=sp_map,
    )


def load_fig2d_cache():
    """Load cached fig2d data.

    Returns
    -------
    dict with keys: ca_map, sp_map.

    Raises
    ------
    FileNotFoundError
        If cache doesn't exist.
    """
    cache_path = CACHE_DIR / 'fig2d_data.npz'
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cache not found: {cache_path}\n"
            "Run: python -m science.intense_figures.fig2.fig2d_activity_maps --cache"
        )
    return dict(np.load(cache_path))


# Cache helpers for Figure 2F (Venn diagrams)

def save_merged_stats_cache(merged_intense, merged_classic):
    """Save merged p-values and flags to cache (shared by fig2f/e/f)."""
    CACHE_DIR.mkdir(exist_ok=True)
    n = len(merged_intense['p_val'])
    np.savez_compressed(
        CACHE_DIR / 'merged_stats.npz',
        p_intense=np.array(merged_intense['p_val'][:n]),
        p_classic=np.array(merged_classic['p_val_computed'][:n]),
        flag_intense=np.array(merged_intense['significant'][:n]).astype(bool),
        flag_classic=np.array(
            merged_classic['significant_recomputed'][:n]).astype(bool),
    )


def load_merged_stats_cache():
    """Load cached merged p-values and significance flags.

    Returns
    -------
    dict with keys: p_intense, p_classic, flag_intense, flag_classic.

    Raises
    ------
    FileNotFoundError
        If cache doesn't exist.
    """
    cache_path = CACHE_DIR / 'merged_stats.npz'
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cache not found: {cache_path}\n"
            "Run: python -m science.intense_figures.fig2.data_loader"
        )
    data = np.load(cache_path)
    return {k: data[k] for k in data.files}


def compute_fig2c_data():
    """Compute Venn diagram counts for two conditions.

    Uses the same approach as fig2f's compute_overlap_metrics:
    pre-filter neurons where BOTH p-values pass the threshold,
    then check significance flags within that pool.

    Condition 1 ('any PC'): no pre-filter, just significance flags.
    Condition 2 ('p < 0.05'): pre-filter both p < 0.05, then flags.

    Loads from merged_stats cache (fast) if available, otherwise
    falls back to full recomputation.

    Returns
    -------
    dict with keys:
        For each condition ('any_pc', 'p05'):
        - '{cond}_intense_only': int
        - '{cond}_classic_only': int
        - '{cond}_both': int
    """
    try:
        cached = load_merged_stats_cache()
        p_intense = cached['p_intense']
        p_classic = cached['p_classic']
        flag_intense = cached['flag_intense']
        flag_classic = cached['flag_classic']
    except FileNotFoundError:
        print("  Merged stats cache not found, loading from raw data...")
        all_classic = load_classic_data(zthr=3)
        all_intense = load_intense_data(feature='place')

        merged_classic = merge_session_stats(
            all_classic, SESSION_LIST,
            ['significant', 'significant_recomputed', 'MI',
             'mean_MI_shuffle', 'std_MI_shuffle', 'z_score',
             'n_spikes', 'p_val_computed'],
        )
        merged_intense = merge_session_stats(
            all_intense, SESSION_LIST,
            ['significant', 'MI', 'p_val', 'r_val'],
        )

        # Save for next time
        save_merged_stats_cache(merged_intense, merged_classic)

        n = len(merged_intense['p_val'])
        p_intense = np.array(merged_intense['p_val'][:n])
        p_classic = np.array(merged_classic['p_val_computed'][:n])
        flag_intense = np.array(merged_intense['significant'][:n]).astype(bool)
        flag_classic = np.array(
            merged_classic['significant_recomputed'][:n]).astype(bool)

    # Condition 1: any PC — significance flags, no pre-filter
    both_any = flag_intense & flag_classic

    # Condition 2: p < 0.05 — pre-filter both p < 0.05, then flags
    pool = (p_intense < 0.05) & (p_classic < 0.05)
    fi_pool = flag_intense[pool]
    fc_pool = flag_classic[pool]
    both_p05 = fi_pool & fc_pool

    return {
        'any_pc_intense_only': int(np.sum(flag_intense & ~flag_classic)),
        'any_pc_classic_only': int(np.sum(flag_classic & ~flag_intense)),
        'any_pc_both': int(np.sum(both_any)),
        'p05_intense_only': int(np.sum(fi_pool & ~fc_pool)),
        'p05_classic_only': int(np.sum(fc_pool & ~fi_pool)),
        'p05_both': int(np.sum(both_p05)),
    }


def save_fig2c_cache(data):
    """Save fig2c Venn diagram counts to cache."""
    CACHE_DIR.mkdir(exist_ok=True)
    np.savez_compressed(CACHE_DIR / 'fig2c_data.npz', **data)


def load_fig2c_cache():
    """Load cached fig2c Venn diagram counts.

    Returns
    -------
    dict with Venn diagram counts.

    Raises
    ------
    FileNotFoundError
        If cache doesn't exist.
    """
    cache_path = CACHE_DIR / 'fig2c_data.npz'
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cache not found: {cache_path}\n"
            "Run: python -m science.intense_figures.fig2.fig2c_circles --cache"
        )
    loaded = dict(np.load(cache_path))
    return {k: int(v) for k, v in loaded.items()}


# Cache helpers for Figure 2E

def save_fig2e_session_cache(session, ssim_vals):
    """Save per-session fig2e metrics to cache."""
    CACHE_DIR.mkdir(exist_ok=True)
    np.savez_compressed(
        CACHE_DIR / f'fig2e_{session}_metrics.npz',
        ssim_vals=ssim_vals,
    )


def load_fig2e_session_cache(session):
    """Load cached per-session fig2e metrics.

    Returns
    -------
    dict with key 'ssim_vals': shape (n_neurons, 3).

    Raises
    ------
    FileNotFoundError
        If cache doesn't exist.
    """
    cache_path = CACHE_DIR / f'fig2e_{session}_metrics.npz'
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}")
    return dict(np.load(cache_path))


def save_fig2e_merged_cache(all_ssim_vals):
    """Save merged fig2e metrics to cache."""
    CACHE_DIR.mkdir(exist_ok=True)
    np.savez_compressed(
        CACHE_DIR / 'fig2e_merged.npz',
        all_ssim_vals=all_ssim_vals,
    )


def load_fig2e_merged_cache():
    """Load cached merged fig2e metrics.

    Returns
    -------
    dict with key 'all_ssim_vals': shape (n_total, 3).

    Raises
    ------
    FileNotFoundError
        If cache doesn't exist.
    """
    cache_path = CACHE_DIR / 'fig2e_merged.npz'
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cache not found: {cache_path}\n"
            "Run: python -m science.intense_figures.fig2.fig2e_metrics --cache"
        )
    return dict(np.load(cache_path))


def save_fig2e_curves_cache(xvals, av_mse, err_mse, av_psnr, err_psnr,
                            av_ssim, err_ssim):
    """Save fig2e curves to cache."""
    CACHE_DIR.mkdir(exist_ok=True)
    np.savez_compressed(
        CACHE_DIR / 'fig2e_curves.npz',
        xvals=xvals,
        av_mse=av_mse, err_mse=err_mse,
        av_psnr=av_psnr, err_psnr=err_psnr,
        av_ssim=av_ssim, err_ssim=err_ssim,
    )


def load_fig2e_curves_cache():
    """Load cached fig2e curves.

    Returns
    -------
    dict with keys: xvals, av_mse, err_mse, av_psnr, err_psnr,
    av_ssim, err_ssim.

    Raises
    ------
    FileNotFoundError
        If cache doesn't exist.
    """
    cache_path = CACHE_DIR / 'fig2e_curves.npz'
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cache not found: {cache_path}\n"
            "Run: python -m science.intense_figures.fig2.fig2e_metrics --cache"
        )
    return dict(np.load(cache_path))


if __name__ == '__main__':
    """Generate and cache Panel D data."""
    print("Loading classic data from .mat file...")
    all_classic = load_classic_data(zthr=3)
    print(f"  Loaded {len(all_classic)} sessions")

    print("Loading INTENSE data from IntenseResults...")
    all_intense = load_intense_data(feature='place')
    print(f"  Loaded {len(all_intense)} sessions")

    print("Merging stats across sessions...")
    merged_classic = merge_session_stats(
        all_classic, SESSION_LIST,
        ['significant', 'significant_recomputed', 'MI',
         'mean_MI_shuffle', 'std_MI_shuffle', 'z_score',
         'n_spikes', 'p_val_computed'],
    )
    merged_intense = merge_session_stats(
        all_intense, SESSION_LIST,
        ['significant', 'MI', 'p_val', 'r_val'],
    )
    print(f"  Classic neurons: {len(merged_classic['significant'])}")
    print(f"  INTENSE neurons: {len(merged_intense['significant'])}")

    print("Computing metrics over p-value range...")
    xvals, pct_intense, pct_classic, jaccard, pct_remaining = \
        compute_metrics_over_prange(merged_intense, merged_classic)

    print("Saving caches...")
    save_fig2f_cache(xvals, pct_intense, pct_classic, jaccard, pct_remaining)
    print(f"  fig2f cache saved to {CACHE_DIR / 'fig2f_data.npz'}")
    save_merged_stats_cache(merged_intense, merged_classic)
    print(f"  Merged stats cache saved to {CACHE_DIR / 'merged_stats.npz'}")
    print("Done!")
