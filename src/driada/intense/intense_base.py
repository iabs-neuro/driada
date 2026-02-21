import time
import multiprocessing
import warnings
from contextlib import contextmanager

import numpy as np
import tqdm
from dataclasses import dataclass
from typing import Callable, Optional
from joblib import Parallel, delayed

from .stats import (
    populate_nested_dict,
    get_table_of_stats,
    criterion1,
    criterion2,
    apply_stage_criterion,
    get_all_nonempty_pvals,
    merge_stage_stats,
    merge_stage_significance,
    DEFAULT_METRIC_DISTR_TYPE,
)
from ..information.info_base import (
    TimeSeries,
    MultiTimeSeries,
    get_multi_mi,
    get_sim,
)
from ..utils.data import nested_dict_to_seq_of_tables, add_names_to_nested_dict
from .io import IntenseResults
from .validation import (
    validate_time_series_bunches,
    validate_metric,
    validate_common_parameters,
)
from .fft import (
    MIN_SHUFFLES_FOR_FFT,
    MIN_SHIFTS_FOR_FFT_DELAYS,
    MAX_FFT_MTS_DIMENSIONS,
    MAX_MTS_MTS_FFT_DIMENSIONS,
    FFT_CONTINUOUS, FFT_DISCRETE, FFT_DISCRETE_DISCRETE,
    FFT_MULTIVARIATE, FFT_MTS_MTS, FFT_MTS_DISCRETE,
    FFT_PEARSON_CONTINUOUS, FFT_PEARSON_DISCRETE, FFT_AV_DISCRETE,
    FFTCacheEntry,
    get_fft_type,
    _extract_fft_data,
    _FFT_COMPUTE,
    _get_ts_key,
    _build_fft_cache,
)
from .correction import get_multicomp_correction_thr

# Import shared parallel utilities
# Note: _parallel_executor is now in utils.parallel for shared use across modules
from ..utils.parallel import parallel_executor as _parallel_executor, get_parallel_backend as _get_parallel_backend

# Default noise amplitude added to MI values for numerical stability
DEFAULT_NOISE_AMPLITUDE = 1e-3


@contextmanager
def _timed_section(timings, name):
    """Context manager for timing code sections. No-op if timings is None.

    Parameters
    ----------
    timings : dict or None
        Dictionary to store timing results. If None, timing is skipped (no-op).
    name : str
        Key under which the elapsed time (in seconds) will be stored in timings.
    """
    if timings is None:
        yield
    else:
        start = time.perf_counter()
        yield
        timings[name] = time.perf_counter() - start


def _build_shift_valid_map(ts_bunch1, ts_bunch2, optimal_delays, ds):
    """
    Build boolean map of valid shift indices per pair from shuffle masks.

    For each pair (i, j), combines the shuffle masks of ts_bunch1[i] and
    ts_bunch2[j], rolls by the optimal delay, and marks which downsampled
    shift indices are valid. The result is a 3D boolean array that can be
    used for vectorized validity checking.

    Cache keys use _get_ts_key() (ts.name) for stability across pickling,
    consistent with fft_cache keying throughout INTENSE.

    Parameters
    ----------
    ts_bunch1 : list
        First set of time series (e.g., neurons).
    ts_bunch2 : list
        Second set of time series (e.g., features).
    optimal_delays : np.ndarray
        Optimal delays of shape (len(ts_bunch1), len(ts_bunch2)).
    ds : int
        Downsampling factor.

    Returns
    -------
    valid_map : np.ndarray, dtype=bool
        Shape (n1, n2, n_shifts). True means shift index s is valid for pair (i,j).
    needs_correction : bool
        False if all masks are trivial (all shifts valid for all pairs).
    """
    n1, n2 = len(ts_bunch1), len(ts_bunch2)
    n_frames = ts_bunch1[0].data.shape[-1]
    n_shifts = n_frames // ds

    valid_map = np.ones((n1, n2, n_shifts), dtype=bool)
    needs_correction = False

    # Cache by (key1, key2, delay) — pairs sharing the same mask+delay
    # combination (common: all neurons share one mask, features unmasked)
    # only compute the valid set once.
    _cache = {}
    for i, ts1 in enumerate(ts_bunch1):
        key1 = _get_ts_key(ts1)
        for j, ts2 in enumerate(ts_bunch2):
            key2 = _get_ts_key(ts2)
            delay = int(optimal_delays[i, j])
            ck = (key1, key2, delay)
            if ck not in _cache:
                combined = ts1.shuffle_mask & ts2.shuffle_mask
                if delay != 0:
                    combined = np.roll(combined, delay)
                if np.all(combined):
                    _cache[ck] = None  # all shifts valid
                else:
                    _cache[ck] = np.unique(np.where(~combined)[0] // ds)
                    needs_correction = True
            inv = _cache[ck]
            if inv is not None:
                valid_map[i, j, inv] = False

    return valid_map, needs_correction


def _find_invalid_shifts(random_shifts, valid_map):
    """
    Find shifts that land on masked (invalid) positions.

    Uses advanced indexing to check all (n1 x n2 x nsh) shifts at once
    against the validity map.

    Parameters
    ----------
    random_shifts : np.ndarray, shape (n1, n2, nsh)
        Shift indices to check.
    valid_map : np.ndarray, shape (n1, n2, n_shifts), dtype=bool
        Validity map from _build_shift_valid_map.

    Returns
    -------
    bad : np.ndarray, shape (n1, n2, nsh), dtype=bool
        True where the shift is invalid.
    """
    ii = np.arange(valid_map.shape[0])[:, None, None]
    jj = np.arange(valid_map.shape[1])[None, :, None]
    return ~valid_map[ii, jj, random_shifts]


def _generate_random_shifts_grid(ts_bunch1, ts_bunch2, optimal_delays, nsh, seed, ds=1):
    """
    Generate all random shifts upfront for all pairs.

    Uses vectorized bulk generation followed by rejection resampling to
    respect shuffle masks. Much faster than per-pair RandomState construction.

    Parameters
    ----------
    ts_bunch1 : list
        First set of time series (e.g., neurons).
    ts_bunch2 : list
        Second set of time series (e.g., features).
    optimal_delays : np.ndarray
        Optimal delays of shape (len(ts_bunch1), len(ts_bunch2)).
    nsh : int
        Number of random shifts to generate per pair.
    seed : int
        Base random seed for reproducibility.
    ds : int, default=1
        Downsampling factor.

    Returns
    -------
    random_shifts : np.ndarray
        Array of shape (len(ts_bunch1), len(ts_bunch2), nsh) containing
        pre-generated random shifts for each pair.

    Notes
    -----
    - Uses _build_shift_valid_map + _find_invalid_shifts for mask correction
    - Respects shuffle masks via rejection resampling (converges in 2-3 rounds)
    """
    n1, n2 = len(ts_bunch1), len(ts_bunch2)
    n_frames = ts_bunch1[0].data.shape[-1]
    n_shifts = n_frames // ds

    rng = np.random.RandomState(seed)

    # Bulk generate all shifts uniformly
    random_shifts = rng.randint(0, n_shifts, size=(n1, n2, nsh))

    # Build validity map from shuffle masks (once)
    valid_map, needs_correction = _build_shift_valid_map(
        ts_bunch1, ts_bunch2, optimal_delays, ds
    )

    # Rejection loop: find invalid shifts, replace them, repeat until convergence
    if needs_correction:
        for _ in range(100):
            bad = _find_invalid_shifts(random_shifts, valid_map)
            n_bad = bad.sum()
            if n_bad == 0:
                break
            random_shifts[bad] = rng.randint(0, n_shifts, size=n_bad)

    return random_shifts


@dataclass
class StageConfig:
    """Configuration for a single stage of INTENSE computation.

    Encapsulates all stage-specific parameters to enable unified
    scan_stage() function for both Stage 1 and Stage 2.

    Attributes
    ----------
    stage_num : int
        Stage number (1 or 2).
    n_shuffles : int
        Number of shuffles for this stage.
    mask : np.ndarray
        Binary mask indicating which pairs to compute.
    topk : int
        True MI should rank in top k among shuffles.
    pval_thr : float, optional
        Base p-value threshold (Stage 2 only). Default 0.05.
    multicomp_correction : str, optional
        Multiple comparison correction method (Stage 2 only).
        Options: 'holm', 'bonferroni', etc.
    """
    stage_num: int
    n_shuffles: int
    mask: np.ndarray
    topk: int
    pval_thr: Optional[float] = None
    multicomp_correction: Optional[str] = None


def _extract_cache_subset(cache, ts_subset, ts_bunch2):
    """Extract cache entries for a specific subset of ts_bunch1.

    Instead of passing the entire cache to each worker, extract only
    the entries that worker needs. This avoids massive serialization
    overhead when using parallel processing with large caches.

    Parameters
    ----------
    cache : dict or None
        Full FFT cache mapping (key1, key2) -> FFTCacheEntry.
    ts_subset : list of TimeSeries
        Subset of ts_bunch1 that this worker will process.
    ts_bunch2 : list of TimeSeries
        Full set of ts_bunch2 (features).

    Returns
    -------
    dict or None
        Subset of cache containing only entries relevant to ts_subset,
        or None if input cache is None.
    """
    if cache is None:
        return None
    subset_cache = {}
    for ts1 in ts_subset:
        key1 = _get_ts_key(ts1)
        for ts2 in ts_bunch2:
            key2 = _get_ts_key(ts2)
            cache_key = (key1, key2)
            if cache_key in cache:
                subset_cache[cache_key] = cache[cache_key]
    return subset_cache


def get_calcium_feature_me_profile(
    exp,
    cell_id=None,
    feat_id=None,
    cbunch=None,
    fbunch=None,
    shift_window=2,
    ds=1,
    metric="mi",
    mi_estimator="gcmi",
    data_type="calcium",
) -> dict:
    """
    Compute metric profile between neurons and behavioral features across time shifts.

    Parameters
    ----------
    exp : Experiment
        Experiment object containing neurons and behavioral features.
    cell_id : int, optional
        Index of a single neuron in exp.neurons. Deprecated - use cbunch instead.
    feat_id : str or tuple of str, optional
        Single feature name(s) to analyze. Deprecated - use fbunch instead.
    cbunch : int, iterable or None, optional
        Neuron indices. If None (default), all neurons will be analyzed.
        Takes precedence over cell_id if both provided.
    fbunch : str, iterable or None, optional
        Feature names. If None (default), all single features will be analyzed.
        Takes precedence over feat_id if both provided.
    shift_window : int, optional
        Maximum shift to test in each direction (seconds). Default: 2.
        Converted to frames internally using exp.fps.
    ds : int, optional
        Downsampling factor. Default: 1 (no downsampling).
    metric : str, optional
        Similarity metric to compute. Default: 'mi'.
        - 'mi': Mutual information
        - 'spearman': Spearman correlation
        - Other metrics supported by get_sim function
    mi_estimator : str, optional
        Mutual information estimator to use when metric='mi'. Default: 'gcmi'.
        Options: 'gcmi' or 'ksg'
    data_type : str, optional
        Type of neural data to use. Default: 'calcium'.
        - 'calcium': Use calcium imaging data
        - 'spikes': Use spike data

    Returns
    -------
    dict
        If single cell_id and feat_id provided (backward compatibility):
            {'me0': float, 'shifted_me': list of float}
        If cbunch or fbunch used:
            Nested dictionary with structure:
            {cell_id: {feat_id: {'me0': float, 'shifted_me': list}}}
            where shifted_me contains metric values from -window to +window.

    Notes
    -----
    - shift_window is in seconds, converted to frames using exp.fps
    - Total number of shifts tested: 2 * shift_window * fps / ds
    - Multi-feature analysis (tuple feat_id) only supported for metric='mi'
    - Progress bar shows computation progress

    Examples
    --------
    This function requires an Experiment object, which contains neural recordings
    and behavioral features. Here's a conceptual example:

    >>> # Pseudo-code example (requires actual Experiment object):
    >>> # exp = load_experiment()  # Load your experiment data
    >>> #
    >>> # # Analyze MI between neuron 0 and speed feature
    >>> # me0, profile = get_calcium_feature_me_profile(exp, 0, 'speed',
    >>> #                                              window=100, ds=5)
    >>> #
    >>> # # Or analyze multiple neurons and features at once
    >>> # results = get_calcium_feature_me_profile(exp, cbunch=[0, 1],
    >>> #                                          fbunch=['speed', 'direction'],
    >>> #                                          window=50, ds=2)
    >>> # # Access results: results[neuron_id][feature_name]['me0']
    >>> pass  # Actual usage requires Experiment object"""
    # Validate inputs
    validate_common_parameters(ds=ds)
    validate_metric(metric)

    if shift_window <= 0:
        raise ValueError(f"shift_window must be positive, got {shift_window}")

    # Convert shift_window from seconds to frames
    window = int(shift_window * exp.fps)

    # Check if single cell/feature mode (backward compatibility)
    single_mode = cell_id is not None and feat_id is not None and cbunch is None and fbunch is None

    # Handle backward compatibility - if old-style single cell_id/feat_id provided
    if cbunch is None and cell_id is not None:
        cbunch = cell_id
    if fbunch is None and feat_id is not None:
        fbunch = feat_id

    # Process cbunch and fbunch using experiment's methods
    cell_ids = exp._process_cbunch(cbunch)
    feat_ids = exp._process_fbunch(fbunch, allow_multifeatures=True, mode=data_type)

    # Validate cell indices
    for cid in cell_ids:
        if not (0 <= cid < len(exp.neurons)):
            raise ValueError(f"cell_id {cid} out of range [0, {len(exp.neurons)-1}]")

    # Initialize results dictionary
    results = {}

    # Progress bar for all combinations
    total_combinations = len(cell_ids) * len(feat_ids)
    pbar = tqdm.tqdm(total=total_combinations, desc="Computing ME profiles")

    for cid in cell_ids:
        cell = exp.neurons[cid]
        ts1 = cell.ca if data_type == "calcium" else cell.spikes
        results[cid] = {}

        for fid in feat_ids:
            shifted_me = []

            if isinstance(fid, str):
                # Single feature
                ts2 = exp.dynamic_features[fid]
                me0 = get_sim(ts1, ts2, metric, ds=ds, estimator=mi_estimator)

                for shift in np.arange(-window, window + ds, ds) // ds:
                    lag_me = get_sim(ts1, ts2, metric, ds=ds, shift=shift, estimator=mi_estimator)
                    shifted_me.append(lag_me)

            else:
                # Multi-feature (tuple)
                if metric != "mi":
                    raise ValueError(
                        f"Multi-feature analysis only supported for metric='mi', got '{metric}'"
                    )
                feats = [exp.dynamic_features[f] for f in fid]
                me0 = get_multi_mi(feats, ts1, ds=ds, estimator=mi_estimator)

                for shift in np.arange(-window, window + ds, ds) // ds:
                    lag_me = get_multi_mi(feats, ts1, ds=ds, shift=shift, estimator=mi_estimator)
                    shifted_me.append(lag_me)

            results[cid][fid] = {"me0": me0, "shifted_me": shifted_me}
            pbar.update(1)

    pbar.close()

    # Return format based on usage mode
    if single_mode:
        # Backward compatibility - return simple format
        return (
            results[cell_ids[0]][feat_ids[0]]["me0"],
            results[cell_ids[0]][feat_ids[0]]["shifted_me"],
        )
    else:
        # New format - return full results dictionary
        return results


def scan_pairs(
    ts_bunch1,
    ts_bunch2,
    metric,
    nsh,
    optimal_delays,
    random_shifts=None,
    mi_estimator="gcmi",
    joint_distr=False,
    ds=1,
    mask=None,
    noise_const=DEFAULT_NOISE_AMPLITUDE,
    seed=None,
    allow_mixed_dimensions=True,
    enable_progressbar=True,
    engine="auto",
    fft_cache: dict = None,
    mi_estimator_kwargs=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate similarity metric and shuffled distributions for pairs of time series.

    This function computes the similarity metric between all pairs from ts_bunch1 and
    ts_bunch2, along with shuffled distributions for significance testing.

    Parameters
    ----------
    ts_bunch1 : list of TimeSeries or MultiTimeSeries
        First set of time series (typically neural signals).
    ts_bunch2 : list of TimeSeries or MultiTimeSeries
        Second set of time series (typically behavioral variables).
    metric : str
        Similarity metric to compute. See validate_metric for supported options.
    nsh : int
        Number of shuffles for significance testing.
    optimal_delays : np.ndarray
        Optimal delays array of shape (len(ts_bunch1), len(ts_bunch2)) or
        (len(ts_bunch1), 1) if joint_distr=True. Contains best shifts in frames.
    random_shifts : np.ndarray, optional
        Pre-generated random shifts of shape (len(ts_bunch1), len(ts_bunch2), nsh).
        If None, shifts will be generated using seed and stable keys.
    mi_estimator : str, default='gcmi'
        Mutual information estimator to use when metric='mi'.
        Options: 'gcmi' (Gaussian copula) or 'ksg' (k-nearest neighbors).
    joint_distr : bool, default=False
        If True, all TimeSeries in ts_bunch2 are treated as components of a
        single multivariate feature. Deprecated - use MultiTimeSeries instead.
    ds : int, default=1
        Downsampling factor. Every ds-th point is used from the time series.
    mask : np.ndarray, optional
        Binary mask array of shape (len(ts_bunch1), len(ts_bunch2)) or
        (len(ts_bunch1), 1) if joint_distr=True. 0 skips calculation, 1 proceeds.
    noise_const : float, default=1e-3
        Small noise amplitude added to improve numerical stability.
    seed : int, optional
        Random seed for reproducibility.
    allow_mixed_dimensions : bool, default=True
        Whether to allow mixed TimeSeries and MultiTimeSeries objects.

        .. deprecated:: 1.1
            This parameter is deprecated and will be removed in a future version.
            Mixed dimensions are now always allowed.
    enable_progressbar : bool, default=True
        Whether to show progress bar during computation.
    engine : {'auto', 'fft', 'loop'}, default='auto'
        Computation engine for MI shuffles:
        - 'auto': Use FFT when applicable (univariate continuous GCMI with nsh >= 50)
        - 'fft': Force FFT (raises error if not applicable)
        - 'loop': Force per-shift loop (original behavior)
    fft_cache : dict, optional
        Pre-computed FFT cache from _build_fft_cache. Keys are (key1, key2) tuples
        using stable identifiers from _get_ts_key(). If provided, avoids redundant
        data extraction.
    mi_estimator_kwargs : dict, optional
        Additional keyword arguments passed to the MI estimator function.

    Returns
    -------
    random_shifts : np.ndarray
        Array of shape (len(ts_bunch1), len(ts_bunch2), nsh) containing
        random shifts used for shuffled distribution computation.
    me_total : np.ndarray
        Array of shape (len(ts_bunch1), len(ts_bunch2), nsh+1) or
        (len(ts_bunch1), 1, nsh+1) if joint_distr=True. Contains true metric
        values at index 0 and shuffled values at indices 1:nsh+1.

    Notes
    -----
    - True metric values: me_total[:,:,0]
    - Shuffled values: me_total[:,:,1:]
    - Random shifts are drawn uniformly from time series length
    - Noise is added as: value * (1 + noise_const * U(-1,1))
    - FFT optimization provides ~100x speedup for univariate continuous GCMI"""

    # Validate inputs
    validate_time_series_bunches(
        ts_bunch1, ts_bunch2, allow_mixed_dimensions=allow_mixed_dimensions
    )
    validate_metric(metric)
    validate_common_parameters(ds=ds, nsh=nsh, noise_const=noise_const)

    # Validate optimal_delays shape
    n1 = len(ts_bunch1)
    n2 = 1 if joint_distr else len(ts_bunch2)

    if optimal_delays.shape != (n1, n2):
        raise ValueError(
            f"optimal_delays shape {optimal_delays.shape} doesn't match expected ({n1}, {n2})"
        )

    if seed is None:
        seed = 0

    # Note: Per-pair deterministic seeding uses pair_seed = seed + hash((key1, key2)) % 1000000
    # This ensures reproducibility without polluting global RNG state.
    # Only used for non-cached paths; cached paths use pre-generated noise arrays.

    lengths1 = [
        len(ts.data) if isinstance(ts, TimeSeries) else ts.data.shape[1] for ts in ts_bunch1
    ]
    lengths2 = [
        len(ts.data) if isinstance(ts, TimeSeries) else ts.data.shape[1] for ts in ts_bunch2
    ]
    if len(set(lengths1)) == 1 and len(set(lengths2)) == 1 and set(lengths1) == set(lengths2):
        t = lengths1[0]  # full length is the same for all time series
    else:
        raise ValueError("Lenghts of TimeSeries do not match!")

    if mask is None:
        mask = np.ones((n1, n2))

    me_table = np.zeros((n1, n2))
    me_table_shuffles = np.zeros((n1, n2, nsh))

    # Generate random shifts if not provided
    if random_shifts is None:
        if joint_distr:
            # DEPRECATED joint_distr path — per-pair shift generation
            random_shifts = np.zeros((n1, n2, nsh), dtype=int)
            for i, ts1 in enumerate(ts_bunch1):
                key1 = _get_ts_key(ts1)
                key2 = _get_ts_key(ts_bunch2[0]) if ts_bunch2 else 0
                pair_seed = seed + hash((key1, key2)) % 1000000 if seed is not None else None
                pair_rng = np.random.RandomState(pair_seed)

                combined_shuffle_mask = ts1.shuffle_mask
                for ts2 in ts_bunch2:
                    combined_shuffle_mask = combined_shuffle_mask & ts2.shuffle_mask
                combined_shuffle_mask = np.roll(combined_shuffle_mask, int(optimal_delays[i, 0]))
                indices_to_select = np.arange(t)[combined_shuffle_mask]
                random_shifts[i, 0, :] = pair_rng.choice(indices_to_select, size=nsh) // ds
        else:
            # Vectorized bulk generation + rejection resampling
            n_shifts = t // ds
            _rng = np.random.RandomState(seed if seed is not None else 0)
            random_shifts = _rng.randint(0, n_shifts, size=(n1, n2, nsh))
            valid_map, needs_correction = _build_shift_valid_map(
                ts_bunch1, ts_bunch2, optimal_delays, ds
            )
            if needs_correction:
                for _ in range(100):
                    bad = _find_invalid_shifts(random_shifts, valid_map)
                    n_bad = bad.sum()
                    if n_bad == 0:
                        break
                    random_shifts[bad] = _rng.randint(0, n_shifts, size=n_bad)

    # Pre-generate noise for FFT cache path (avoids per-pair RandomState)
    if fft_cache is not None:
        _noise_rng = np.random.RandomState(seed)
        _noise_true = _noise_rng.random(size=(n1, n2)) * noise_const
        _noise_shuffles = _noise_rng.random(size=(n1, n2, nsh)) * noise_const

    # calculate similarity metric arrays
    for i, ts1 in tqdm.tqdm(
        enumerate(ts_bunch1),
        total=len(ts_bunch1),
        position=0,
        leave=True,
        disable=not enable_progressbar,
    ):
        # DEPRECATED: This joint_distr branch is deprecated and will be removed in v2.0
        # Use MultiTimeSeries for joint distribution handling instead
        # FUTURE: Remove this entire branch in v2.0
        if joint_distr:
            if metric != "mi":
                raise ValueError("joint_distr mode works with metric = 'mi' only")
            if mask[i, 0] == 1:
                # default metric without shuffling, minus due to different order
                me0 = get_multi_mi(
                    ts_bunch2, ts1, ds=ds, shift=-optimal_delays[i, 0] // ds, estimator=mi_estimator,
                    mi_estimator_kwargs=mi_estimator_kwargs,
                )
                # Use deterministic RNG for this pair (stable key seeding)
                key1 = _get_ts_key(ts1)
                key2 = _get_ts_key(ts_bunch2[0]) if ts_bunch2 else 0
                pair_seed = seed + hash((key1, key2)) % 1000000 if seed is not None else None
                pair_rng = np.random.RandomState(pair_seed)

                me_table[i, 0] = (
                    me0 + pair_rng.random() * noise_const
                )  # add small noise for better fitting

                random_noise = (
                    pair_rng.random(size=nsh) * noise_const
                )  # add small noise for better fitting
                for k, shift in enumerate(random_shifts[i, 0, :]):
                    mi = get_multi_mi(ts_bunch2, ts1, ds=ds, shift=shift, estimator=mi_estimator,
                                      mi_estimator_kwargs=mi_estimator_kwargs)
                    me_table_shuffles[i, 0, k] = mi + random_noise[k]

            else:
                me_table[i, 0] = None
                me_table_shuffles[i, 0, :] = np.full(shape=nsh, fill_value=None)

        else:
            if fft_cache is not None:
                # Vectorized cache path: batch all cached pairs for this neuron
                key1 = _get_ts_key(ts1)
                cached_js = []
                mi_all_list = []
                uncached_js = []

                for j in range(n2):
                    if mask[i, j] == 1:
                        entry = fft_cache.get((key1, _get_ts_key(ts_bunch2[j])))
                        if entry is not None:
                            cached_js.append(j)
                            mi_all_list.append(entry.mi_all)
                        else:
                            uncached_js.append(j)

                # Batch process all cached pairs at once
                if cached_js:
                    cached_js_arr = np.array(cached_js)
                    mi_stack = np.array(mi_all_list)  # (n_cached, n_shifts)
                    arange_cached = np.arange(len(cached_js))

                    # Vectorized true MI lookup
                    opt_shifts = (optimal_delays[i, cached_js_arr] // ds).astype(int)
                    me0_vals = mi_stack[arange_cached, opt_shifts]

                    # Vectorized shuffle MI lookup
                    shifts_batch = random_shifts[i, cached_js_arr, :]  # (n_cached, nsh)
                    shuffle_vals = mi_stack[arange_cached[:, None], shifts_batch]

                    # Write results with pre-generated noise
                    me_table[i, cached_js_arr] = me0_vals + _noise_true[i, cached_js_arr]
                    me_table_shuffles[i, cached_js_arr, :] = shuffle_vals + _noise_shuffles[i, cached_js_arr, :]

                # Non-FFT-able pairs (cache entry was None): loop fallback
                for j in uncached_js:
                    ts2 = ts_bunch2[j]
                    key2 = _get_ts_key(ts2)
                    pair_seed = seed + hash((key1, key2)) % 1000000 if seed is not None else None
                    pair_rng = np.random.RandomState(pair_seed)
                    me0 = get_sim(
                        ts1, ts2, metric, ds=ds,
                        shift=optimal_delays[i, j] // ds,
                        estimator=mi_estimator,
                        check_for_coincidence=True,
                        mi_estimator_kwargs=mi_estimator_kwargs,
                    )
                    me_table[i, j] = me0 + pair_rng.random() * noise_const
                    random_noise = pair_rng.random(size=nsh) * noise_const
                    for k, shift in enumerate(random_shifts[i, j, :]):
                        me = get_sim(
                            ts1, ts2, metric, ds=ds, shift=shift,
                            estimator=mi_estimator,
                            mi_estimator_kwargs=mi_estimator_kwargs,
                        )
                        me_table_shuffles[i, j, k] = me + random_noise[k]

            else:
                # No cache — per-pair loop with fresh FFT or loop computation
                for j, ts2 in enumerate(ts_bunch2):
                    if mask[i, j] == 1:
                        key1 = _get_ts_key(ts1)
                        key2 = _get_ts_key(ts2)
                        pair_seed = seed + hash((key1, key2)) % 1000000 if seed is not None else None
                        pair_rng = np.random.RandomState(pair_seed)
                        fft_type = get_fft_type(ts1, ts2, metric, mi_estimator, nsh, engine)

                        if fft_type is not None:
                            # Unified FFT-accelerated path
                            data1, data2 = _extract_fft_data(ts1, ts2, fft_type, ds)
                            compute_fn = _FFT_COMPUTE[fft_type]

                            opt_shift = optimal_delays[i, j] // ds
                            me0 = compute_fn(data1, data2, np.array([opt_shift]))[0]
                            shuffle_mis = compute_fn(data1, data2, random_shifts[i, j, :])

                            me_table[i, j] = me0 + pair_rng.random() * noise_const
                            random_noise = pair_rng.random(size=nsh) * noise_const
                            me_table_shuffles[i, j, :] = shuffle_mis + random_noise

                        else:
                            # Original loop path (no FFT available)
                            me0 = get_sim(
                                ts1, ts2, metric, ds=ds,
                                shift=optimal_delays[i, j] // ds,
                                estimator=mi_estimator,
                                check_for_coincidence=True,
                                mi_estimator_kwargs=mi_estimator_kwargs,
                            )
                            me_table[i, j] = me0 + pair_rng.random() * noise_const
                            random_noise = pair_rng.random(size=nsh) * noise_const
                            for k, shift in enumerate(random_shifts[i, j, :]):
                                me = get_sim(
                                    ts1, ts2, metric, ds=ds, shift=shift,
                                    estimator=mi_estimator,
                                    mi_estimator_kwargs=mi_estimator_kwargs,
                                )
                                me_table_shuffles[i, j, k] = me + random_noise[k]

    me_total = np.dstack((me_table, me_table_shuffles))

    return random_shifts, me_total


def scan_pairs_parallel(
    ts_bunch1,
    ts_bunch2,
    metric,
    nsh,
    optimal_delays,
    mi_estimator="gcmi",
    joint_distr=False,
    allow_mixed_dimensions=True,
    ds=1,
    mask=None,
    noise_const=DEFAULT_NOISE_AMPLITUDE,
    seed=None,
    n_jobs=-1,
    engine="auto",
    fft_cache: dict = None,
    mi_estimator_kwargs=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate metric values and shuffles for time series pairs using parallel processing.

    Parameters
    ----------
    ts_bunch1 : list of TimeSeries
        First set of time series.
    ts_bunch2 : list of TimeSeries
        Second set of time series.
    metric : str
        Similarity metric to compute:
        - 'mi': Mutual information
        - 'spearman': Spearman correlation
        - Other metrics supported by get_sim function
    nsh : int
        Number of shuffles to perform.
    optimal_delays : np.ndarray of shape (len(ts_bunch1), len(ts_bunch2))
        Pre-computed optimal delays for each pair.
    mi_estimator : str, default='gcmi'
        Mutual information estimator to use when metric='mi'.
        Options: 'gcmi' (Gaussian copula) or 'ksg' (k-nearest neighbors).
    joint_distr : bool, default=False
        If True, treats all ts_bunch2 as components of a single multifeature.
    allow_mixed_dimensions : bool, default=True
        Whether to allow mixed TimeSeries and MultiTimeSeries objects.

        .. deprecated:: 1.1
            This parameter is deprecated and will be removed in a future version.
            Mixed dimensions are now always allowed.
    ds : int, default=1
        Downsampling factor.
    mask : np.ndarray, optional
        Binary mask of shape (len(ts_bunch1), len(ts_bunch2)).
        0 = skip computation, 1 = compute. Default: all ones.
    noise_const : float, default=1e-3
        Small noise added to improve numerical stability.
    seed : int, optional
        Random seed for reproducibility.
    n_jobs : int, default=-1
        Number of parallel jobs. -1 uses all cores.
    engine : {'auto', 'fft', 'loop'}, default='auto'
        Computation engine for MI shuffles:
        - 'auto': Use FFT when applicable (univariate continuous GCMI with nsh >= 50)
        - 'fft': Force FFT (raises error if not applicable)
        - 'loop': Force per-shift loop (original behavior)
    fft_cache : dict, optional
        Pre-computed FFT cache mapping (key1, key2) tuples to FFTCacheEntry objects.
        Keys are stable identifiers from _get_ts_key(). If provided, avoids redundant
        data extraction. If None, FFT type is computed fresh for each pair.
    mi_estimator_kwargs : dict, optional
        Additional keyword arguments passed to the MI estimator function.

    Returns
    -------
    random_shifts : np.ndarray of shape (len(ts_bunch1), len(ts_bunch2), nsh)
        Random shifts used for shuffling.
    me_total : np.ndarray of shape (len(ts_bunch1), len(ts_bunch2), nsh+1)
        Metric values. [:,:,0] contains true values, [:,:,1:] contains shuffles.

    Raises
    ------
    ValueError
        If input validation fails or parameters are invalid.

    Notes
    -----
    - Parallelization is done by splitting ts_bunch1 across workers
    - Each worker handles a subset of ts_bunch1 against all of ts_bunch2
    - Uses threading backend if PyTorch present (checked lazily), else loky
    - Random seeding ensures reproducibility across different mask configurations
    - FFT optimization provides ~100x speedup for univariate continuous GCMI

    See Also
    --------
    ~driada.intense.intense_base.scan_pairs : Sequential version of this function
    ~driada.intense.intense_base.scan_pairs_router : Wrapper that chooses between parallel and sequential

    Examples
    --------
    >>> # Minimal example with 2x2 pairs
    >>> import numpy as np
    >>> from driada.information.info_base import TimeSeries
    >>> np.random.seed(42)  # For reproducibility
    >>> # Small data: 2 neurons, 2 behaviors, 50 timepoints
    >>> neurons = [TimeSeries(np.random.randn(50), discrete=False) for _ in range(2)]
    >>> behaviors = [TimeSeries(np.random.randn(50), discrete=False) for _ in range(2)]
    >>> delays = np.zeros((2, 2), dtype=int)  # No delays
    >>> # Just 5 shuffles for demonstration
    >>> shifts, metrics = scan_pairs_parallel(neurons, behaviors, 'mi',
    ...                                      5, delays, n_jobs=1, seed=42)
    >>> shifts.shape
    (2, 2, 5)
    >>> metrics.shape  # Original + 5 shuffles = 6 total
    (2, 2, 6)"""

    # Validate inputs
    validate_time_series_bunches(
        ts_bunch1, ts_bunch2, allow_mixed_dimensions=allow_mixed_dimensions
    )
    validate_metric(metric)
    validate_common_parameters(ds=ds, nsh=nsh, noise_const=noise_const)

    n1 = len(ts_bunch1)
    n2 = 1 if joint_distr else len(ts_bunch2)

    # Validate optimal_delays shape
    if optimal_delays.shape != (n1, n2):
        raise ValueError(
            f"optimal_delays shape {optimal_delays.shape} doesn't match expected ({n1}, {n2})"
        )

    me_total = np.zeros((n1, n2, nsh + 1))

    if n_jobs == -1:
        n_jobs = min(multiprocessing.cpu_count(), n1)

    # Initialize mask if None
    if mask is None:
        n1 = len(ts_bunch1)
        n2 = 1 if joint_distr else len(ts_bunch2)
        mask = np.ones((n1, n2))

    # Pre-generate ALL random shifts upfront using stable key seeding
    random_shifts = _generate_random_shifts_grid(
        ts_bunch1, ts_bunch2, optimal_delays, nsh, seed if seed is not None else 0, ds
    )

    # Limit n_jobs to number of items to avoid empty worker splits
    n_jobs_effective = min(n_jobs, len(ts_bunch1))
    if n_jobs_effective < n_jobs:
        import warnings
        warnings.warn(
            f"Requested {n_jobs} parallel jobs but only {len(ts_bunch1)} items to process. "
            f"Using {n_jobs_effective} workers to avoid empty splits.",
            UserWarning
        )

    # Split work across workers
    split_ts_bunch1_inds = np.array_split(np.arange(len(ts_bunch1)), n_jobs_effective)
    split_ts_bunch1 = [np.array(ts_bunch1)[idxs] for idxs in split_ts_bunch1_inds]
    split_optimal_delays = [optimal_delays[idxs] for idxs in split_ts_bunch1_inds]
    split_random_shifts = [random_shifts[idxs] for idxs in split_ts_bunch1_inds]
    split_mask = [mask[idxs] for idxs in split_ts_bunch1_inds]

    # Split cache per worker - each worker only gets entries it needs
    # This avoids serializing the entire cache (potentially GBs) to each worker
    split_caches = [
        _extract_cache_subset(fft_cache, subset, ts_bunch2)
        for subset in split_ts_bunch1
    ]

    # Parallel execution with backend-specific config
    with _parallel_executor(n_jobs_effective) as parallel:
        parallel_result = parallel(
            delayed(scan_pairs)(
                small_ts_bunch,
                ts_bunch2,
                metric,
                nsh,
                split_optimal_delays[worker_idx],
                split_random_shifts[worker_idx],  # Pre-generated, pre-split shifts
                mi_estimator,
                joint_distr=joint_distr,
                allow_mixed_dimensions=allow_mixed_dimensions,
                ds=ds,
                mask=split_mask[worker_idx],
                noise_const=noise_const,
                seed=seed,
                enable_progressbar=False,
                engine=engine,
                fft_cache=split_caches[worker_idx],
                mi_estimator_kwargs=mi_estimator_kwargs,
            )
            for worker_idx, small_ts_bunch in enumerate(split_ts_bunch1)
        )

    for i in range(n_jobs_effective):
        inds_of_interest = split_ts_bunch1_inds[i]
        random_shifts[inds_of_interest, :, :] = parallel_result[i][0][:, :, :]
        me_total[inds_of_interest, :, :] = parallel_result[i][1][:, :, :]

    return random_shifts, me_total


def scan_pairs_router(
    ts_bunch1,
    ts_bunch2,
    metric,
    nsh,
    optimal_delays,
    mi_estimator="gcmi",
    joint_distr=False,
    allow_mixed_dimensions=True,
    ds=1,
    mask=None,
    noise_const=DEFAULT_NOISE_AMPLITUDE,
    seed=None,
    enable_parallelization=True,
    n_jobs=-1,
    engine="auto",
    fft_cache: dict = None,
    mi_estimator_kwargs=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Route metric computation to parallel or sequential implementation.

    Parameters
    ----------
    ts_bunch1 : list of TimeSeries
        First set of time series.
    ts_bunch2 : list of TimeSeries
        Second set of time series.
    metric : str
        Similarity metric to compute:
        - 'mi': Mutual information
        - 'spearman': Spearman correlation
        - Other metrics supported by get_sim function
    nsh : int
        Number of shuffles to perform.
    optimal_delays : np.ndarray of shape (len(ts_bunch1), len(ts_bunch2))
        Pre-computed optimal delays for each pair.
    mi_estimator : str, default='gcmi'
        Mutual information estimator to use when metric='mi'.
        Options: 'gcmi' (Gaussian copula) or 'ksg' (k-nearest neighbors).
    joint_distr : bool, default=False
        If True, treats all ts_bunch2 as components of a single multifeature.
    allow_mixed_dimensions : bool, default=True
        Whether to allow mixed TimeSeries and MultiTimeSeries objects.

        .. deprecated:: 1.1
            This parameter is deprecated and will be removed in a future version.
            Mixed dimensions are now always allowed.
    ds : int, default=1
        Downsampling factor.
    mask : np.ndarray, optional
        Binary mask of shape (len(ts_bunch1), len(ts_bunch2)).
        0 = skip computation, 1 = compute. Default: all ones.
    noise_const : float, default=1e-3
        Small noise added to improve numerical stability.
    seed : int, optional
        Random seed for reproducibility.
    enable_parallelization : bool, default=True
        Whether to use parallel processing.
    n_jobs : int, default=-1
        Number of parallel jobs if parallelization enabled. -1 uses all cores.
    engine : {'auto', 'fft', 'loop'}, default='auto'
        Computation engine for MI shuffles:
        - 'auto': Use FFT when applicable (univariate continuous GCMI with nsh >= 50)
        - 'fft': Force FFT (raises error if not applicable)
        - 'loop': Force per-shift loop (original behavior)
    fft_cache : dict, optional
        Pre-computed FFT cache mapping (global_i, j) tuples to FFTCacheEntry objects.
        If provided, avoids redundant data extraction. If None, FFT type is computed
        fresh for each pair. Use _build_fft_cache() to create this cache.
    mi_estimator_kwargs : dict, optional
        Additional keyword arguments passed to the MI estimator function.

    Returns
    -------
    random_shifts : np.ndarray of shape (len(ts_bunch1), len(ts_bunch2), nsh)
        Random shifts used for shuffling.
    me_total : np.ndarray of shape (len(ts_bunch1), len(ts_bunch2), nsh+1)
        Metric values. [:,:,0] contains true values, [:,:,1:] contains shuffles.

    Notes
    -----
    This function automatically chooses between sequential and parallel
    implementations based on the enable_parallelization flag. It's the
    recommended entry point for scan_pairs functionality.

    FFT optimization provides ~100x speedup for univariate continuous GCMI.

    See Also
    --------
    ~driada.intense.intense_base.scan_pairs : Sequential implementation
    ~driada.intense.intense_base.scan_pairs_parallel : Parallel implementation

    Examples
    --------
    >>> # Router example - chooses sequential or parallel execution
    >>> import numpy as np
    >>> from driada.information.info_base import TimeSeries
    >>> np.random.seed(42)
    >>> # Minimal data for fast execution
    >>> neurons = [TimeSeries(np.random.randn(30), discrete=False) for _ in range(2)]
    >>> behaviors = [TimeSeries(np.random.randn(30), discrete=False) for _ in range(2)]
    >>> delays = np.zeros((2, 2), dtype=int)  # No delays
    >>> # Use sequential mode (enable_parallelization=False)
    >>> shifts, metrics = scan_pairs_router(neurons, behaviors, 'mi',
    ...                                    3, delays, enable_parallelization=False, seed=42)
    >>> metrics.shape  # 1 original + 3 shuffles = 4 total
    (2, 2, 4)
    >>> # First slice contains actual MI values
    >>> metrics[:, :, 0].shape
    (2, 2)"""

    if enable_parallelization:
        random_shifts, me_total = scan_pairs_parallel(
            ts_bunch1,
            ts_bunch2,
            metric,
            nsh,
            optimal_delays,
            mi_estimator,
            joint_distr=joint_distr,
            allow_mixed_dimensions=allow_mixed_dimensions,
            ds=ds,
            mask=mask,
            noise_const=noise_const,
            seed=seed,
            n_jobs=n_jobs,
            engine=engine,
            fft_cache=fft_cache,
            mi_estimator_kwargs=mi_estimator_kwargs,
        )

    else:
        random_shifts, me_total = scan_pairs(
            ts_bunch1,
            ts_bunch2,
            metric,
            nsh,
            optimal_delays,
            random_shifts=None,  # Generate shifts inside scan_pairs
            mi_estimator=mi_estimator,
            joint_distr=joint_distr,
            allow_mixed_dimensions=allow_mixed_dimensions,
            ds=ds,
            mask=mask,
            seed=seed,
            noise_const=noise_const,
            engine=engine,
            fft_cache=fft_cache,
            mi_estimator_kwargs=mi_estimator_kwargs,
        )

    return random_shifts, me_total


def scan_stage(
    ts_bunch1: list,
    ts_bunch2: list,
    config: StageConfig,
    optimal_delays: np.ndarray,
    metric: str,
    mi_estimator: str,
    metric_distr_type: str,
    noise_const: float,
    ds: int,
    seed: int,
    joint_distr: bool,
    allow_mixed_dimensions: bool,
    enable_parallelization: bool,
    n_jobs: int,
    engine: str,
    fft_cache: dict = None,
    verbose: bool = True,
    mi_estimator_kwargs=None,
) -> tuple[dict, dict, dict]:
    """
    Execute a single stage of INTENSE computation.

    This function encapsulates the common logic between Stage 1 and Stage 2:
    1. Scan pairs to compute metric values and shuffle distributions
    2. Compute statistical tables from the results
    3. Apply the appropriate criterion:
       - Stage 1: criterion1 (rank-based filtering using topk)
       - Stage 2: criterion2 (p-value based with multiple comparison correction)

    For Stage 2, the multiple comparison correction threshold is computed
    internally from the stage statistics using config.pval_thr and
    config.multicomp_correction.

    Parameters
    ----------
    ts_bunch1 : list of TimeSeries or MultiTimeSeries
        First set of time series (typically neural signals).
    ts_bunch2 : list of TimeSeries or MultiTimeSeries
        Second set of time series (typically behavioral variables).
    config : StageConfig
        Configuration for this stage (stage number, n_shuffles, mask, topk, etc.).
    optimal_delays : np.ndarray
        Optimal delays array of shape (len(ts_bunch1), len(ts_bunch2)).
    metric : str
        Similarity metric to compute.
    mi_estimator : str
        Mutual information estimator ('gcmi' or 'ksg').
    metric_distr_type : str
        Distribution type for fitting shuffled metric values.
    noise_const : float
        Small noise amplitude added for numerical stability.
    ds : int
        Downsampling factor.
    seed : int
        Random seed for reproducibility.
    joint_distr : bool
        If True, all ts_bunch2 are treated as single multivariate feature.
    allow_mixed_dimensions : bool
        Whether to allow mixed TimeSeries and MultiTimeSeries objects.
    enable_parallelization : bool
        Whether to use parallel processing.
    n_jobs : int
        Number of parallel jobs if parallelization enabled.
    engine : str
        Computation engine ('auto', 'fft', 'loop').
    fft_cache : dict, optional
        Pre-computed FFT cache for accelerated computation.
    verbose : bool, default=True
        Whether to print stage information.
    mi_estimator_kwargs : dict, optional
        Additional keyword arguments passed to the MI estimator function.

    Returns
    -------
    stage_stats : dict
        Statistical results for all pairs from get_table_of_stats.
    stage_significance : dict
        Significance results for all pairs from apply_stage_criterion.
    stage_info : dict
        Additional information including:
        - 'random_shifts': Random shifts array used for shuffling
        - 'me_total': Full metric values array (true + shuffles)
        - 'pass_mask': Binary mask of pairs that passed the criterion
        - 'multicorr_thr': Multiple comparison threshold (Stage 2 only, None for Stage 1)
    """
    n1 = len(ts_bunch1)
    n2 = 1 if joint_distr else len(ts_bunch2)

    if verbose:
        print(f"Stage {config.stage_num}: {config.n_shuffles} shuffles")

    # 1. Scan pairs
    random_shifts, me_total = scan_pairs_router(
        ts_bunch1,
        ts_bunch2,
        metric,
        config.n_shuffles,
        optimal_delays,
        mi_estimator,
        joint_distr=joint_distr,
        allow_mixed_dimensions=allow_mixed_dimensions,
        ds=ds,
        mask=config.mask,
        noise_const=noise_const,
        seed=seed,
        enable_parallelization=enable_parallelization,
        n_jobs=n_jobs,
        engine=engine,
        fft_cache=fft_cache,
        mi_estimator_kwargs=mi_estimator_kwargs,
    )

    # 2. Compute stats
    stage_stats = get_table_of_stats(
        me_total,
        optimal_delays,
        precomputed_mask=config.mask,
        metric_distr_type=metric_distr_type,
        nsh=config.n_shuffles,
        stage=config.stage_num,
    )

    # 3. Apply criterion
    if config.stage_num == 1:
        stage_significance, pass_mask = apply_stage_criterion(
            stage_stats,
            stage_num=1,
            n1=n1,
            n2=n2,
            n_shuffles=config.n_shuffles,
            topk=config.topk,
        )
        multicorr_thr = None
    else:
        # Stage 2: compute multicorr_thr from p-values, then apply criterion
        nhyp = int(np.sum(config.mask))
        all_pvals = get_all_nonempty_pvals(stage_stats, range(n1), range(n2))
        multicorr_thr = get_multicomp_correction_thr(
            config.pval_thr,
            mode=config.multicomp_correction,
            all_pvals=all_pvals,
            nhyp=nhyp,
        )
        stage_significance, pass_mask = apply_stage_criterion(
            stage_stats,
            stage_num=2,
            n1=n1,
            n2=n2,
            n_shuffles=config.n_shuffles,
            topk=config.topk,
            multicorr_thr=multicorr_thr,
        )

    stage_info = {
        "random_shifts": random_shifts,
        "me_total": me_total,
        "pass_mask": pass_mask,
        "multicorr_thr": multicorr_thr,
    }

    return stage_stats, stage_significance, stage_info


def compute_me_stats(
    ts_bunch1,
    ts_bunch2,
    names1=None,
    names2=None,
    mode="two_stage",
    metric="mi",
    mi_estimator="gcmi",
    mi_estimator_kwargs=None,
    precomputed_mask_stage1=None,
    precomputed_mask_stage2=None,
    n_shuffles_stage1=100,
    n_shuffles_stage2=10000,
    joint_distr=False,
    allow_mixed_dimensions=True,
    metric_distr_type=DEFAULT_METRIC_DISTR_TYPE,
    noise_ampl=DEFAULT_NOISE_AMPLITUDE,
    ds=1,
    topk1=1,
    topk2=5,
    multicomp_correction="holm",
    pval_thr=0.01,
    find_optimal_delays=False,
    skip_delays=[],
    shift_window=100,
    verbose=True,
    seed=None,
    enable_parallelization=True,
    n_jobs=-1,
    duplicate_behavior="ignore",
    engine="auto",
    store_random_shifts=False,
    profile=False,
):
    """
    Calculates similarity metric statistics for TimeSeries or MultiTimeSeries pairs

    Parameters
    ----------
    ts_bunch1 : list of TimeSeries objects
        First set of time series

    ts_bunch2 : list of TimeSeries objects
        Second set of time series

    names1 : list of str, optional
        names than will be given to time series from tsbunch1 in final results

    names2 : list of str, optional
        names than will be given to time series from tsbunch2 in final results

    mode : str, default='two_stage'
        Computation mode. 3 modes are available:
        'stage1': perform preliminary scanning with "n_shuffles_stage1" shuffles only.
                  Rejects strictly non-significant neuron-feature pairs, does not give definite results
                  about significance of the others.
        'stage2': skip stage 1 and perform full-scale scanning ("n_shuffles_stage2" shuffles) of all neuron-feature pairs.
                  Gives definite results, but can be very time-consuming. Also reduces statistical power
                  of multiple comparison tests, since the number of hypotheses is very high.
        'two_stage': prune non-significant pairs during stage 1 and perform thorough testing for the rest during stage 2.
                     Recommended mode.

    metric : str, default='mi'
        similarity metric between TimeSeries

    mi_estimator : str, default='gcmi'
        Mutual information estimator to use when metric='mi'. Options: 'gcmi' or 'ksg'

    mi_estimator_kwargs : dict, optional
        Additional keyword arguments passed to the MI estimator function.

    precomputed_mask_stage1 : np.array, optional
        precomputed mask for skipping some of possible pairs in stage 1.
        Shape: (len(ts_bunch1), len(ts_bunch2)) or (len(ts_bunch), 1) if joint_distr=True
        0 in mask values means calculation will be skipped.
        1 in mask values means calculation will proceed.

    precomputed_mask_stage2 : np.array, optional
        precomputed mask for skipping some of possible pairs in stage 2.
        Shape: (len(ts_bunch1), len(ts_bunch2)) or (len(ts_bunch), 1) if joint_distr=True
        0 in mask values means calculation will be skipped.
        1 in mask values means calculation will proceed.

    n_shuffles_stage1 : int, default=100
        number of shuffles for first stage

    n_shuffles_stage2 : int, default=10000
        number of shuffles for second stage

    joint_distr : bool, default=False
        if joint_distr=True, ALL features in feat_bunch will be treated as components of a single multifeature
        For example, 'x' and 'y' features will be put together into ('x','y') multifeature.

    allow_mixed_dimensions : bool, default=True
        if True, both TimeSeries and MultiTimeSeries can be provided as signals.
        This parameter overrides "joint_distr"

        .. deprecated:: 1.1
            This parameter is deprecated and will be removed in a future version.
            Mixed dimensions are now always allowed.

    metric_distr_type : str, default="gamma_zi"
        Distribution type for shuffled metric null distribution. Options:

        - 'gamma_zi' (default): Zero-inflated gamma distribution. Explicitly models the probability
          mass at zero that commonly occurs in MI null distributions. Provides superior goodness-of-fit
          and accurate parameter estimation without requiring artificial noise.

        - 'gamma': Standard gamma distribution with small noise added (noise_ampl) to handle zeros.
          Provided for backward compatibility. Less statistically principled than 'gamma_zi'.

        - Other scipy.stats distributions: 'lognorm', 'norm', etc. are supported but not recommended
          for MI distributions.

    noise_ampl : float, default=1e-3
        Small noise amplitude, which is added to metrics to improve numerical fit

    ds : int, default=1
        Downsampling constant. Every "ds" point will be taken from the data time series.

    topk1 : int, default=1
        true MI for stage 1 should be among topk1 MI shuffles

    topk2 : int, default=5
        true MI for stage 2 should be among topk2 MI shuffles

    multicomp_correction : str or None, default='holm'
        type of multiple comparisons correction. Supported types are None (no correction),
        "bonferroni", "holm", and "fdr_bh".

    pval_thr : float, default=0.01
        pvalue threshold. if multicomp_correction=None, this is a p-value for a single pair.
        For FWER methods (bonferroni, holm), this is the family-wise error rate.
        For FDR methods (fdr_bh), this is the false discovery rate.

    find_optimal_delays : bool, default=False
        Allows slight shifting (not more than +- shift_window) of time series,
        selects a shift with the highest MI as default.

    skip_delays : list, default=[]
        List of indices from ts_bunch2 for which delays are not applied (set to 0).
        Has no effect if find_optimal_delays = False

    shift_window : int, default=100
        Window for optimal shift search (frames). Optimal shift will lie in the range
        -shift_window <= opt_shift <= shift_window

    verbose : bool, default=True
        whether to print intermediate information

    seed : int, optional
        random seed for reproducibility

    enable_parallelization : bool, default=True
        whether to use parallel processing for computations

    n_jobs : int, default=-1
        number of parallel jobs to use. -1 means use all available processors

    duplicate_behavior : str, default='ignore'
        How to handle duplicate TimeSeries in ts_bunch1 or ts_bunch2.
        - 'ignore': Process duplicates normally (default)
        - 'raise': Raise an error if duplicates are found
        - 'warn': Print a warning but continue processing

    engine : {'auto', 'fft', 'loop'}, default='auto'
        Computation engine for MI shuffles:
        - 'auto': Use FFT when applicable (univariate continuous GCMI with nsh >= 50)
        - 'fft': Force FFT (raises error if not applicable)
        - 'loop': Force per-shift loop (original behavior)
        FFT optimization provides ~100x speedup for Stage 2.

    store_random_shifts : bool, default=False
        Whether to store the random shift indices used during shuffle computation.
        When False (default), random_shifts1 and random_shifts2 arrays are not stored
        in accumulated_info, saving significant memory (e.g., ~400MB for typical datasets).
        Set to True if you need the shift indices for debugging or reproducibility analysis.

    profile : bool, default=False
        Whether to collect internal timing information. When True, accumulated_info
        will include a 'timings' dict with execution times (in seconds) for:
        - 'stage1_delay_optimization': delay optimization (if find_optimal_delays=True)
        - 'stage1_pair_scanning': stage 1 pair scanning
        - 'stage2_pair_scanning': stage 2 pair scanning (if applicable)
        - 'total': sum of all timing sections

    Returns
    -------
    stats : dict of dict of dicts
        Outer dict keys: indices of tsbunch1 or names1, if given
        Inner dict keys: indices or tsbunch2 or names2, if given
        Last dict: dictionary of stats variables.
        Can be easily converted to pandas DataFrame by pd.DataFrame(stats)

    significance : dict of dict of dicts
        Outer dict keys: indices of tsbunch1 or names1, if given
        Inner dict keys: indices or tsbunch2 or names2, if given
        Last dict: dictionary of significance-related variables.
        Can be easily converted to pandas DataFrame by pd.DataFrame(significance)

    accumulated_info : dict
        Data collected during computation.

    Raises
    ------
    ValueError
        If mode is not 'stage1', 'stage2', or 'two_stage'.
        If multicomp_correction is not None, 'bonferroni', 'holm', or 'fdr_bh'.
        If pval_thr is not between 0 and 1.
        If duplicate_behavior is not 'ignore', 'raise', or 'warn'.
        If allow_mixed_dimensions=False but mixed types are provided.
        If duplicate TimeSeries found and duplicate_behavior='raise'.

    Notes
    -----
    - When comparing the same bunch (ts_bunch1 is ts_bunch2), the diagonal
      of masks is automatically set to 0 to avoid self-comparisons.
    - In 'stage2' mode, dummy stage1 structures are created with placeholder values
      to maintain consistency in the return format.
    - For stage2, the final mask combines stage1 results with precomputed_mask_stage2
      using logical AND.
    - Input masks are never modified; copies are created when needed."""

    # FUTURE: add automatic min_shifts from autocorrelation time

    # Validate inputs
    validate_time_series_bunches(
        ts_bunch1, ts_bunch2, allow_mixed_dimensions=allow_mixed_dimensions
    )
    validate_metric(metric)
    validate_common_parameters(shift_window=shift_window, ds=ds, noise_const=noise_ampl)

    # Validate mode
    if mode not in ["stage1", "stage2", "two_stage"]:
        raise ValueError(f"mode must be 'stage1', 'stage2', or 'two_stage', got '{mode}'")

    # Validate multicomp_correction
    if multicomp_correction not in [None, "bonferroni", "holm", "fdr_bh"]:
        raise ValueError(f"Unknown multiple comparison correction method: '{multicomp_correction}'")

    # Validate pval_thr
    if not 0 < pval_thr < 1:
        raise ValueError(f"pval_thr must be between 0 and 1, got {pval_thr}")

    # Validate stage-specific parameters
    validate_common_parameters(nsh=n_shuffles_stage1)
    validate_common_parameters(nsh=n_shuffles_stage2)

    accumulated_info = dict()
    timings = {} if profile else None

    # Temporary naming: Save original names and assign temporary names if missing
    # This ensures all TimeSeries have names for FFT cache keys
    original_names1 = [ts.name if hasattr(ts, 'name') else None for ts in ts_bunch1]
    original_names2 = [ts.name if hasattr(ts, 'name') else None for ts in ts_bunch2]

    # Assign temporary names if missing
    for i, ts in enumerate(ts_bunch1):
        if not hasattr(ts, 'name') or ts.name is None or ts.name == '':
            # Use names1 if provided, else generate temp name
            ts.name = str(names1[i]) if names1 and i < len(names1) else f"_ts1_{i}"

    for j, ts in enumerate(ts_bunch2):
        if not hasattr(ts, 'name') or ts.name is None or ts.name == '':
            # Use names2 if provided, else generate temp name
            ts.name = str(names2[j]) if names2 and j < len(names2) else f"_ts2_{j}"

    try:
        # Check if we're comparing the same bunch with itself
        same_data_bunch = ts_bunch1 is ts_bunch2

        n1 = len(ts_bunch1)
        n2 = len(ts_bunch2)
        if not allow_mixed_dimensions:
            n2 = 1 if joint_distr else len(ts_bunch2)

            tsbunch1_is_1d = np.all([isinstance(ts, TimeSeries) for ts in ts_bunch1])
            tsbunch2_is_1d = np.all([isinstance(ts, TimeSeries) for ts in ts_bunch2])
            if not (tsbunch1_is_1d and tsbunch2_is_1d):
                raise ValueError(
                    "Multiple time series types found, but allow_mixed_dimensions=False."
                    "Consider setting it to True"
                )

        if precomputed_mask_stage1 is None:
            precomputed_mask_stage1 = np.ones((n1, n2))
        else:
            # Create a copy to avoid modifying the input
            precomputed_mask_stage1 = precomputed_mask_stage1.copy()

        if precomputed_mask_stage2 is None:
            precomputed_mask_stage2 = np.ones((n1, n2))
        else:
            # Create a copy to avoid modifying the input
            precomputed_mask_stage2 = precomputed_mask_stage2.copy()

        # If comparing the same bunch with itself, mask out the diagonal
        # to avoid computing MI of a TimeSeries with itself at zero shift
        if same_data_bunch:
            np.fill_diagonal(precomputed_mask_stage1, 0)
            np.fill_diagonal(precomputed_mask_stage2, 0)

        # Handle duplicate TimeSeries based on duplicate_behavior parameter
        if duplicate_behavior in ["raise", "warn"]:
            # Check for duplicates in ts_bunch1
            ts1_ids = []
            for ts in ts_bunch1:
                ts_id = id(ts.data) if hasattr(ts, "data") else id(ts)
                ts1_ids.append(ts_id)

            if len(set(ts1_ids)) < len(ts1_ids):
                msg = "Duplicate TimeSeries objects found in ts_bunch1"
                if duplicate_behavior == "raise":
                    raise ValueError(msg)
                else:  # warn
                    print(f"Warning: {msg}")

            # Check for duplicates in ts_bunch2 (if not joint_distr)
            if not joint_distr:
                ts2_ids = []
                for ts in ts_bunch2:
                    ts_id = id(ts.data) if hasattr(ts, "data") else id(ts)
                    ts2_ids.append(ts_id)

                if len(set(ts2_ids)) < len(ts2_ids):
                    msg = "Duplicate TimeSeries objects found in ts_bunch2"
                    if duplicate_behavior == "raise":
                        raise ValueError(msg)
                    else:  # warn
                        print(f"Warning: {msg}")

        optimal_delays = np.zeros((n1, n2), dtype=int)

        # Validate skip_delays indices before use
        if skip_delays:
            invalid_indices = [i for i in skip_delays if i < 0 or i >= len(ts_bunch2)]
            if invalid_indices:
                raise ValueError(
                    f"skip_delays contains invalid indices {invalid_indices}. "
                    f"Valid range: [0, {len(ts_bunch2)-1}] for {len(ts_bunch2)} features."
                )

        ts_with_delays = [ts for _, ts in enumerate(ts_bunch2) if not skip_delays or _ not in skip_delays]
        ts_with_delays_inds = np.array([_ for _, ts in enumerate(ts_bunch2) if not skip_delays or _ not in skip_delays])

        # Build FFT cache once at the start for reuse across delays + stages
        if verbose:
            print(f"Building FFT cache for {len(ts_bunch1)}x{len(ts_bunch2)} pairs (engine={engine})...")
        with _timed_section(timings, 'fft_cache_building'):
            fft_cache, fft_type_counts = _build_fft_cache(
                ts_bunch1, ts_bunch2, metric, mi_estimator, ds, engine, joint_distr,
                n_jobs=n_jobs if enable_parallelization else 1,
            )

        # Store FFT type counts for profiling
        if profile and fft_type_counts:
            timings['fft_type_counts'] = fft_type_counts

        with _timed_section(timings, 'stage1_delay_optimization'):
            if find_optimal_delays and len(ts_with_delays) > 0:
                # Local import to avoid circular dependency (delay.py imports from this module)
                from .delay import calculate_optimal_delays, calculate_optimal_delays_parallel

                # Use unified fft_cache - no need for separate delay cache
                # Since cache uses stable keys, ts_with_delays objects are already cached
                if enable_parallelization:
                    optimal_delays_res = calculate_optimal_delays_parallel(
                        ts_bunch1,
                        ts_with_delays,
                        metric,
                        shift_window,
                        ds,
                        verbose=verbose,
                        n_jobs=n_jobs,
                        mi_estimator=mi_estimator,
                        engine=engine,
                        fft_cache=fft_cache,
                        mi_estimator_kwargs=mi_estimator_kwargs,
                    )
                else:
                    optimal_delays_res = calculate_optimal_delays(
                        ts_bunch1,
                        ts_with_delays,
                        metric,
                        shift_window,
                        ds,
                        verbose=verbose,
                        mi_estimator=mi_estimator,
                        engine=engine,
                        fft_cache=fft_cache,
                        mi_estimator_kwargs=mi_estimator_kwargs,
                    )

                optimal_delays[:, ts_with_delays_inds] = optimal_delays_res

        accumulated_info["optimal_delays"] = optimal_delays

        # Initialize masks based on mode
        if mode == "stage2":
            # For stage2-only mode, assume all pairs pass stage 1
            mask_from_stage1 = np.ones((n1, n2))
        else:
            mask_from_stage1 = np.zeros((n1, n2))

        mask_from_stage2 = np.zeros((n1, n2))
        nhyp = n1 * n2

        # Conditional noise based on distribution type
        # ZIG handles zeros explicitly, so no noise needed
        noise_const = 0 if metric_distr_type == "gamma_zi" else noise_ampl

        if mode in ["two_stage", "stage1"]:
            npairs_to_check1 = int(np.sum(precomputed_mask_stage1))
            if verbose:
                print(f"Starting stage 1 scanning for {npairs_to_check1}/{nhyp} possible pairs")

            with _timed_section(timings, 'stage1_pair_scanning'):
                # STAGE 1 - primary scanning using scan_stage abstraction
                config_stage1 = StageConfig(
                    stage_num=1,
                    n_shuffles=n_shuffles_stage1,
                    mask=precomputed_mask_stage1,
                    topk=topk1,
                )

                stage_1_stats, stage_1_significance, stage_1_info = scan_stage(
                    ts_bunch1,
                    ts_bunch2,
                    config_stage1,
                    optimal_delays,
                    metric=metric,
                    mi_estimator=mi_estimator,
                    metric_distr_type=metric_distr_type,
                    noise_const=noise_const,
                    ds=ds,
                    seed=seed,
                    joint_distr=joint_distr,
                    allow_mixed_dimensions=allow_mixed_dimensions,
                    enable_parallelization=enable_parallelization,
                    n_jobs=n_jobs,
                    engine=engine,
                    fft_cache=fft_cache,
                    verbose=False,  # We handle verbose output here
                    mi_estimator_kwargs=mi_estimator_kwargs,
                )

                # Extract results from scan_stage
                random_shifts1 = stage_1_info["random_shifts"]
                me_total1 = stage_1_info["me_total"]
                mask_from_stage1 = stage_1_info["pass_mask"]

            # Convert to per-quantity tables for accumulated_info
            stage_1_stats_per_quantity = nested_dict_to_seq_of_tables(
                stage_1_stats, ordered_names1=range(n1), ordered_names2=range(n2)
            )
            stage_1_significance_per_quantity = nested_dict_to_seq_of_tables(
                stage_1_significance, ordered_names1=range(n1), ordered_names2=range(n2)
            )

            stage1_info = {
                "stage_1_significance": stage_1_significance_per_quantity,
                "stage_1_stats": stage_1_stats_per_quantity,
                "me_total1": me_total1,
            }
            if store_random_shifts:
                stage1_info["random_shifts1"] = random_shifts1
            accumulated_info.update(stage1_info)

            nhyp = int(np.sum(mask_from_stage1))  # number of hypotheses for further statistical testing
            if verbose:
                print("Stage 1 results:")
                print(
                    f"{nhyp/n1/n2*100:.2f}% ({nhyp}/{n1*n2}) of possible pairs identified as candidates"
                )

        if mode == "stage1" or nhyp == 0:
            final_stats = add_names_to_nested_dict(stage_1_stats, names1, names2)
            final_significance = add_names_to_nested_dict(stage_1_significance, names1, names2)

            if profile:
                timings['total'] = sum(v for v in timings.values() if isinstance(v, (int, float)))
                accumulated_info['timings'] = timings

            return final_stats, final_significance, accumulated_info

        elif mode == "stage2":
            # For stage2-only mode, create empty stage 1 structures
            stage_1_stats = populate_nested_dict(dict(), range(n1), range(n2))
            stage_1_significance = populate_nested_dict(dict(), range(n1), range(n2))
            # Set all pairs as passing stage 1 with placeholder values
            for i in range(n1):
                for j in range(n2):
                    stage_1_stats[i][j] = {"pre_rval": None, "pre_pval": None}
                    stage_1_significance[i][j]["stage1"] = True

        # Now proceed with stage 2
        if mode in ["two_stage", "stage2"]:
            # STAGE 2 - full-scale scanning
            combined_mask_for_stage_2 = np.ones((n1, n2))
            combined_mask_for_stage_2[np.where(mask_from_stage1 == 0)] = (
                0  # exclude non-significant pairs from stage1
            )
            combined_mask_for_stage_2[np.where(precomputed_mask_stage2 == 0)] = (
                0  # exclude precomputed stage 2 pairs
            )

            npairs_to_check2 = int(np.sum(combined_mask_for_stage_2))
            if verbose:
                print(f"Starting stage 2 scanning for {npairs_to_check2}/{nhyp} possible pairs")

            with _timed_section(timings, 'stage2_pair_scanning'):
                # STAGE 2 using scan_stage abstraction
                config_stage2 = StageConfig(
                    stage_num=2,
                    n_shuffles=n_shuffles_stage2,
                    mask=combined_mask_for_stage_2,
                    topk=topk2,
                    pval_thr=pval_thr,
                    multicomp_correction=multicomp_correction,
                )

                stage_2_stats, stage_2_significance, stage_2_info = scan_stage(
                    ts_bunch1,
                    ts_bunch2,
                    config_stage2,
                    optimal_delays,
                    metric=metric,
                    mi_estimator=mi_estimator,
                    metric_distr_type=metric_distr_type,
                    noise_const=noise_const,
                    ds=ds,
                    seed=seed,
                    joint_distr=joint_distr,
                    allow_mixed_dimensions=allow_mixed_dimensions,
                    enable_parallelization=enable_parallelization,
                    n_jobs=n_jobs,
                    engine=engine,
                    fft_cache=fft_cache,
                    verbose=False,  # We handle verbose output here
                    mi_estimator_kwargs=mi_estimator_kwargs,
                )

                # Extract results from scan_stage
                random_shifts2 = stage_2_info["random_shifts"]
                me_total2 = stage_2_info["me_total"]
                mask_from_stage2 = stage_2_info["pass_mask"]
                multicorr_thr = stage_2_info["multicorr_thr"]

            # Convert to per-quantity tables for accumulated_info
            stage_2_stats_per_quantity = nested_dict_to_seq_of_tables(
                stage_2_stats, ordered_names1=range(n1), ordered_names2=range(n2)
            )
            stage_2_significance_per_quantity = nested_dict_to_seq_of_tables(
                stage_2_significance, ordered_names1=range(n1), ordered_names2=range(n2)
            )

            stage2_info = {
                "stage_2_significance": stage_2_significance_per_quantity,
                "stage_2_stats": stage_2_stats_per_quantity,
                "me_total2": me_total2,
                "corrected_pval_thr": multicorr_thr,
                "group_pval_thr": pval_thr,
            }
            if store_random_shifts:
                stage2_info["random_shifts2"] = random_shifts2
            accumulated_info.update(stage2_info)

            num2 = int(np.sum(mask_from_stage2))
            if verbose:
                print("Stage 2 results:")
                print(
                    f"{num2/n1/n2*100:.2f}% ({num2}/{n1*n2}) of possible pairs identified as significant"
                )

            # Always merge stats for consistency
            merged_stats = merge_stage_stats(stage_1_stats, stage_2_stats)
            merged_significance = merge_stage_significance(stage_1_significance, stage_2_significance)
            final_stats = add_names_to_nested_dict(merged_stats, names1, names2)
            final_significance = add_names_to_nested_dict(merged_significance, names1, names2)

            if profile:
                timings['total'] = sum(v for v in timings.values() if isinstance(v, (int, float)))
                accumulated_info['timings'] = timings

            return final_stats, final_significance, accumulated_info
    finally:
        # Free FFT cache memory explicitly to prevent accumulation
        if 'fft_cache' in locals() and fft_cache is not None:
            fft_cache.clear()
            del fft_cache

        # Restore original names to leave objects unchanged
        for i, ts in enumerate(ts_bunch1):
            ts.name = original_names1[i]
        for j, ts in enumerate(ts_bunch2):
            ts.name = original_names2[j]
