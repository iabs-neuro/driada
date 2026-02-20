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
    compute_mi_batch_fft,
    compute_mi_gd_fft,
    compute_mi_mts_fft,
    compute_mi_mts_mts_fft,
    compute_mi_mts_discrete_fft,
    compute_mi_dd_fft,
    compute_pearson_batch_fft,
    compute_av_batch_fft,
)
from ..utils.data import nested_dict_to_seq_of_tables, add_names_to_nested_dict
from .io import IntenseResults

# Import shared parallel utilities
# Note: _parallel_executor is now in utils.parallel for shared use across modules
from ..utils.parallel import parallel_executor as _parallel_executor, get_parallel_backend as _get_parallel_backend

# Default noise amplitude added to MI values for numerical stability
DEFAULT_NOISE_AMPLITUDE = 1e-3

# Minimum number of shuffles to benefit from FFT optimization
# FFT is always beneficial due to high per-call overhead in loop fallback
MIN_SHUFFLES_FOR_FFT = 1

# Minimum number of shifts to benefit from FFT in delay optimization
# FFT is always beneficial due to high per-call overhead in loop fallback
MIN_SHIFTS_FOR_FFT_DELAYS = 1

# Maximum dimensions for FFT acceleration of MultiTimeSeries
MAX_FFT_MTS_DIMENSIONS = 3
MAX_MTS_MTS_FFT_DIMENSIONS = 6  # Total d1+d2 limit for MTS-MTS pairs

# FFT type constants
FFT_CONTINUOUS = "cc"       # Continuous-continuous (univariate 1D-1D)
FFT_DISCRETE = "gd"         # Gaussian-discrete (one discrete, one continuous)
FFT_DISCRETE_DISCRETE = "dd"  # Discrete-discrete (both variables discrete)
FFT_MULTIVARIATE = "mts"    # MultiTimeSeries + univariate TimeSeries
FFT_MTS_MTS = "mts_mts"     # MultiTimeSeries + MultiTimeSeries
FFT_MTS_DISCRETE = "mts_discrete"  # MultiTimeSeries + discrete
FFT_PEARSON_CONTINUOUS = "pearson_cc"  # Pearson correlation (continuous-continuous)
FFT_PEARSON_DISCRETE = "pearson_gd"   # Pearson correlation (continuous-discrete, point-biserial)
FFT_AV_DISCRETE = "av_gd"             # Activity ratio (continuous-discrete, binary)


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


def _get_ts_key(ts):
    """
    Get stable identifier for TimeSeries based on its name attribute.

    This function generates a stable key for cache lookups that works across
    both threading and loky (pickling) joblib backends.

    All TimeSeries/MultiTimeSeries objects MUST have a name attribute set.
    Names should be assigned at creation or by INTENSE pipelines before
    calling compute_me_stats.

    Parameters
    ----------
    ts : TimeSeries or MultiTimeSeries
        Time series object to get key for.

    Returns
    -------
    str
        The name attribute of the TimeSeries.

    Raises
    ------
    ValueError
        If the TimeSeries does not have a name attribute or the name is empty.

    Notes
    -----
    - Keys are stable across pickling (names are pickled with objects)
    - Assumes names are unique within a bundle
    - Used for FFT cache keys and random seed generation
    - INTENSE pipelines assign temporary names to unnamed objects on entry
    """
    if hasattr(ts, 'name') and ts.name:
        return ts.name

    # Should never reach here if naming strategy is complete
    raise ValueError(
        f"TimeSeries missing name attribute. "
        f"All TimeSeries/MultiTimeSeries must have names for FFT cache and seeding. "
        f"Shape: {ts.data.shape}, discrete: {ts.discrete}"
    )


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
class FFTCacheEntry:
    """Cache entry for pre-computed MI values.

    Stores MI values for ALL possible shifts, enabling O(1) lookup
    without redundant FFT computation. The FFT is computed once when
    building the cache, then MI for any shift is just array indexing.

    Attributes
    ----------
    fft_type : str
        FFT type constant (FFT_CONTINUOUS, FFT_DISCRETE, FFT_MULTIVARIATE).
    mi_all : np.ndarray
        MI values for ALL n shifts (shape: (n,)).
    """
    fft_type: str
    mi_all: np.ndarray


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


def get_fft_type(
    ts1,
    ts2,
    metric: str,
    mi_estimator: str,
    count: int,
    engine: str,
    for_delays: bool = False,
):
    """Determine which FFT optimization to use for a time series pair.

    Unified function that replaces _should_use_fft, _should_use_fft_gd,
    _should_use_fft_mts, and _should_use_fft_for_delays.

    Parameters
    ----------
    ts1 : TimeSeries or MultiTimeSeries
        First time series.
    ts2 : TimeSeries or MultiTimeSeries
        Second time series.
    metric : str
        Similarity metric being used.
    mi_estimator : str
        MI estimator ('gcmi' or 'ksg').
    count : int
        Number of shuffles (nsh) or shifts (for delay optimization).
    engine : str
        Computation engine: 'auto', 'fft', or 'loop'.
    for_delays : bool
        If True, use delay optimization threshold (MIN_SHIFTS_FOR_FFT_DELAYS).
        Default False uses shuffle threshold (MIN_SHUFFLES_FOR_FFT).

    Returns
    -------
    str or None
        FFT type constant (FFT_CONTINUOUS, FFT_DISCRETE, FFT_MULTIVARIATE),
        or None for loop fallback.

    Raises
    ------
    ValueError
        If engine='fft' but no FFT optimization is applicable.
    """
    # Early exit for loop engine
    if engine == "loop":
        return None

    # FFT works with MI (GCMI estimator), fast_pearsonr, and av
    if metric == "mi" and mi_estimator == "gcmi":
        pass  # proceed to type classification
    elif metric == "fast_pearsonr":
        pass  # proceed — continuous-continuous only
    elif metric == "av":
        pass  # proceed — binary-discrete + continuous only
    else:
        if engine == "fft":
            raise ValueError(
                f"engine='fft' requires metric='mi' (with mi_estimator='gcmi'), "
                f"'fast_pearsonr', or 'av'. "
                f"Got: metric={metric}, mi_estimator={mi_estimator}"
            )
        return None

    # Determine threshold based on context
    threshold = MIN_SHIFTS_FOR_FFT_DELAYS if for_delays else MIN_SHUFFLES_FOR_FFT

    # Classify the pair type and check applicability
    fft_type = None
    error_msg = None

    # For fast_pearsonr, only univariate continuous-continuous is supported via FFT
    is_pearson = (metric == "fast_pearsonr")

    is_av = (metric == "av")

    # Check for MultiTimeSeries + univariate TimeSeries pair
    if isinstance(ts1, MultiTimeSeries) and isinstance(ts2, TimeSeries) and not isinstance(ts2, MultiTimeSeries):
        if is_pearson or is_av:
            error_msg = f"FFT {metric} only supports univariate TimeSeries pairs"
        else:
            mts, ts = ts1, ts2
            # Check for MTS + discrete pair first
            if not mts.discrete and ts.discrete and mts.data.shape[0] <= MAX_FFT_MTS_DIMENSIONS:
                fft_type = FFT_MTS_DISCRETE
            # Then check for MTS + continuous pair
            elif not mts.discrete and not ts.discrete and mts.data.shape[0] <= MAX_FFT_MTS_DIMENSIONS:
                fft_type = FFT_MULTIVARIATE
            else:
                error_msg = (
                    f"MultiTimeSeries FFT requires continuous MTS with d <= {MAX_FFT_MTS_DIMENSIONS} "
                    f"paired with either continuous or discrete TimeSeries. "
                    f"Got: MTS discrete={mts.discrete}, TS discrete={ts.discrete}, MTS shape={mts.data.shape}"
                )
    elif isinstance(ts2, MultiTimeSeries) and isinstance(ts1, TimeSeries) and not isinstance(ts1, MultiTimeSeries):
        if is_pearson or is_av:
            error_msg = f"FFT {metric} only supports univariate TimeSeries pairs"
        else:
            mts, ts = ts2, ts1
            # Check for MTS + discrete pair first
            if not mts.discrete and ts.discrete and mts.data.shape[0] <= MAX_FFT_MTS_DIMENSIONS:
                fft_type = FFT_MTS_DISCRETE
            # Then check for MTS + continuous pair
            elif not mts.discrete and not ts.discrete and mts.data.shape[0] <= MAX_FFT_MTS_DIMENSIONS:
                fft_type = FFT_MULTIVARIATE
            else:
                error_msg = (
                    f"MultiTimeSeries FFT requires continuous MTS with d <= {MAX_FFT_MTS_DIMENSIONS} "
                    f"paired with either continuous or discrete TimeSeries. "
                    f"Got: MTS discrete={mts.discrete}, TS discrete={ts.discrete}, MTS shape={mts.data.shape}"
                )
    # Check for MultiTimeSeries + MultiTimeSeries pair
    elif isinstance(ts1, MultiTimeSeries) and isinstance(ts2, MultiTimeSeries):
        if is_pearson or is_av:
            error_msg = f"FFT {metric} only supports univariate TimeSeries pairs"
        else:
            d1 = ts1.data.shape[0]
            d2 = ts2.data.shape[0]
            if (not ts1.discrete and not ts2.discrete and
                d1 + d2 <= MAX_MTS_MTS_FFT_DIMENSIONS):
                fft_type = FFT_MTS_MTS
            else:
                error_msg = (
                    f"MTS-MTS FFT requires continuous variables with "
                    f"d1+d2 <= {MAX_MTS_MTS_FFT_DIMENSIONS}. "
                    f"Got: d1={d1}, d2={d2}, discrete=({ts1.discrete},{ts2.discrete})"
                )
    # Check for univariate TimeSeries pairs
    elif (isinstance(ts1, TimeSeries) and not isinstance(ts1, MultiTimeSeries) and
          isinstance(ts2, TimeSeries) and not isinstance(ts2, MultiTimeSeries)):
        is_av = (metric == "av")
        if is_av:
            # AV requires one binary-discrete and one continuous variable
            if ts1.discrete != ts2.discrete:
                discrete_ts = ts1 if ts1.discrete else ts2
                if discrete_ts.is_binary:
                    fft_type = FFT_AV_DISCRETE
                else:
                    error_msg = "FFT av requires binary discrete variable"
            else:
                error_msg = "FFT av requires one discrete and one continuous variable"
        elif is_pearson:
            if not ts1.discrete and not ts2.discrete:
                fft_type = FFT_PEARSON_CONTINUOUS
            elif ts1.discrete != ts2.discrete:
                fft_type = FFT_PEARSON_DISCRETE
            else:
                error_msg = "FFT Pearson does not support two discrete variables"
        else:
            # Discrete-continuous pair
            if ts1.discrete != ts2.discrete:
                fft_type = FFT_DISCRETE
            # Continuous-continuous pair
            elif not ts1.discrete and not ts2.discrete:
                fft_type = FFT_CONTINUOUS
            else:
                # Both discrete - use discrete-discrete FFT
                fft_type = FFT_DISCRETE_DISCRETE
    else:
        error_msg = (
            f"FFT requires univariate TimeSeries or MultiTimeSeries+TimeSeries pair. "
            f"Got: ts1={type(ts1).__name__}, ts2={type(ts2).__name__}"
        )

    # Handle engine='fft' validation
    if engine == "fft":
        if fft_type is None:
            raise ValueError(
                f"engine='fft' requested but no FFT optimization is applicable. {error_msg}"
            )
        return fft_type

    # engine='auto': check threshold
    if fft_type is not None and count >= threshold:
        return fft_type

    return None


def _extract_fft_data(ts1, ts2, fft_type, ds: int):
    """Extract and prepare data for FFT computation.

    Parameters
    ----------
    ts1 : TimeSeries or MultiTimeSeries
        First time series in the pair.
    ts2 : TimeSeries or MultiTimeSeries
        Second time series in the pair.
    fft_type : str
        FFT type constant (FFT_CONTINUOUS, FFT_DISCRETE, FFT_MULTIVARIATE).
    ds : int
        Downsampling factor.

    Returns
    -------
    tuple
        (data1, data2) ready for the corresponding FFT function.
    """
    if fft_type == FFT_CONTINUOUS:
        return ts1.copula_normal_data[::ds], ts2.copula_normal_data[::ds]
    elif fft_type == FFT_DISCRETE:
        if ts1.discrete:
            return ts2.copula_normal_data[::ds], ts1.int_data[::ds]
        else:
            return ts1.copula_normal_data[::ds], ts2.int_data[::ds]
    elif fft_type == FFT_MULTIVARIATE:
        if isinstance(ts1, MultiTimeSeries):
            return ts2.copula_normal_data[::ds], ts1.copula_normal_data[:, ::ds]
        else:
            return ts1.copula_normal_data[::ds], ts2.copula_normal_data[:, ::ds]
    elif fft_type == FFT_MTS_DISCRETE:
        # Handle both orientations (MTS, discrete) or (discrete, MTS)
        if isinstance(ts1, MultiTimeSeries):
            return ts1.copula_normal_data[:, ::ds], ts2.int_data[::ds]
        else:
            return ts2.copula_normal_data[:, ::ds], ts1.int_data[::ds]
    elif fft_type == FFT_MTS_MTS:
        return ts1.copula_normal_data[:, ::ds], ts2.copula_normal_data[:, ::ds]
    elif fft_type == FFT_DISCRETE_DISCRETE:
        return ts1.int_data[::ds], ts2.int_data[::ds]
    elif fft_type == FFT_PEARSON_CONTINUOUS:
        return ts1.data[::ds], ts2.data[::ds]
    elif fft_type == FFT_PEARSON_DISCRETE:
        # Treat discrete data as float for point-biserial correlation
        if ts1.discrete:
            return ts2.data[::ds], ts1.data[::ds].astype(float)
        else:
            return ts1.data[::ds], ts2.data[::ds].astype(float)
    elif fft_type == FFT_AV_DISCRETE:
        # AV uses RAW data (not copula-normalized): (continuous, binary)
        if ts1.discrete:
            return ts2.data[::ds], ts1.data[::ds].astype(float)
        else:
            return ts1.data[::ds], ts2.data[::ds].astype(float)
    else:
        raise ValueError(f"Unknown FFT type: {fft_type}")


# Dispatch table for FFT compute functions
_FFT_COMPUTE = {
    FFT_CONTINUOUS: compute_mi_batch_fft,
    FFT_DISCRETE: compute_mi_gd_fft,
    FFT_DISCRETE_DISCRETE: compute_mi_dd_fft,
    FFT_MULTIVARIATE: compute_mi_mts_fft,
    FFT_MTS_MTS: compute_mi_mts_mts_fft,
    FFT_MTS_DISCRETE: compute_mi_mts_discrete_fft,
    FFT_PEARSON_CONTINUOUS: compute_pearson_batch_fft,
    FFT_PEARSON_DISCRETE: compute_pearson_batch_fft,
    FFT_AV_DISCRETE: compute_av_batch_fft,
}


def _build_fft_cache(
    ts_bunch1: list,
    ts_bunch2: list,
    metric: str,
    mi_estimator: str,
    ds: int,
    engine: str,
    joint_distr: bool = False,
    n_jobs: int = 1,
) -> dict:
    """Build FFT cache for all pairs using stable keys.

    Pre-computes FFT data for all neuron-feature pairs that support FFT,
    enabling reuse across delay optimization, Stage 1, and Stage 2.

    When n_jobs != 1 and joint_distr=False, uses parallel processing to
    split ts_bunch1 across workers for faster cache building.

    Uses stable keys (TimeSeries names or data hashes) instead of positional
    indices, allowing cache to work correctly with both threading and loky
    joblib backends, and enabling automatic cache reuse when the same
    TimeSeries objects appear in different contexts (e.g., ts_with_delays subset).

    Parameters
    ----------
    ts_bunch1 : list
        First set of time series (e.g., neurons).
    ts_bunch2 : list
        Second set of time series (e.g., features).
    metric : str
        Similarity metric being used.
    mi_estimator : str
        MI estimator ('gcmi' or 'ksg').
    ds : int
        Downsampling factor.
    engine : str
        Computation engine: 'auto', 'fft', or 'loop'.
    joint_distr : bool
        If True, ts_bunch2 is treated as a single multifeature.
    n_jobs : int, optional
        Number of parallel jobs. Default: 1 (serial).
        Use -1 for all CPU cores. Parallelization is disabled
        when joint_distr=True or ts_bunch1 has only one element.

    Returns
    -------
    tuple
        A tuple of (cache, fft_type_counts):
        - cache: Dictionary mapping (key1, key2) tuple to FFTCacheEntry or None.
          Keys are stable identifiers from _get_ts_key() (names or hashes).
          None indicates loop fallback should be used for that pair.
        - fft_type_counts: Dictionary mapping FFT type strings to counts.
          Includes 'loop' for pairs that require loop fallback.
    """
    # Efficient duplicate name check: only validates when duplicate names exist
    all_ts = list(ts_bunch1) + list(ts_bunch2)
    # Cache (ts, name) pairs to avoid calling _get_ts_key() twice
    ts_name_pairs = [(ts, _get_ts_key(ts)) for ts in all_ts]  # Will raise if any unnamed
    all_names = [name for _, name in ts_name_pairs]

    # Fast path: if all names unique, no duplicates possible
    if len(set(all_names)) != len(all_ts):
        # Slow path: duplicates exist, check if they have different data
        # Use data equality instead of id() for pickle stability (loky backend)
        name_to_data = {}
        for ts, name in ts_name_pairs:
            if name not in name_to_data:
                # First occurrence of this name - store reference data
                name_to_data[name] = ts.data
            else:
                # Duplicate name - check if data matches
                if not np.array_equal(name_to_data[name], ts.data):
                    # Same name, different data = COLLISION!
                    raise ValueError(
                        f"Cache collision: TimeSeries name '{name}' maps to different data! "
                        f"Same names must have identical data to share FFT cache."
                    )

    # Delegate to parallel version if enabled and appropriate
    if n_jobs != 1 and not joint_distr and len(ts_bunch1) > 1:
        return _build_fft_cache_parallel(
            ts_bunch1, ts_bunch2, metric, mi_estimator, ds, engine, n_jobs
        )

    # Serial implementation
    cache = {}
    fft_type_counts = {}  # Track counts for profiling

    for ts1 in ts_bunch1:
        if joint_distr:
            # Joint distribution mode - no FFT support
            # Use first ts2 as representative (all treated as single multifeature)
            key1 = _get_ts_key(ts1)
            key2 = _get_ts_key(ts_bunch2[0]) if ts_bunch2 else 0
            cache[(key1, key2)] = None
            fft_type_counts['loop'] = fft_type_counts.get('loop', 0) + 1
        else:
            for ts2 in ts_bunch2:
                # Use count=1 since we just need type, not threshold check
                fft_type = get_fft_type(ts1, ts2, metric, mi_estimator, 1, engine)

                # Get stable keys for cache
                key1 = _get_ts_key(ts1)
                key2 = _get_ts_key(ts2)

                if fft_type is not None:
                    data1, data2 = _extract_fft_data(ts1, ts2, fft_type, ds)
                    compute_fn = _FFT_COMPUTE[fft_type]

                    # Precompute MI for ALL shifts (FFT done once here)
                    n = len(data1) if data1.ndim == 1 else data1.shape[1]
                    all_shifts = np.arange(n)
                    mi_all = compute_fn(data1, data2, all_shifts)

                    cache[(key1, key2)] = FFTCacheEntry(
                        fft_type=fft_type,
                        mi_all=mi_all,
                    )
                    fft_type_counts[fft_type] = fft_type_counts.get(fft_type, 0) + 1
                else:
                    cache[(key1, key2)] = None
                    fft_type_counts['loop'] = fft_type_counts.get('loop', 0) + 1

    return cache, fft_type_counts


def _build_fft_cache_worker(
    ts_bunch1_subset: list,
    ts_bunch2: list,
    metric: str,
    mi_estimator: str,
    ds: int,
    engine: str,
) -> tuple:
    """Build FFT cache for a subset of ts_bunch1 (worker function).

    This is a worker function used by _build_fft_cache_parallel to process
    a subset of the first time series bunch in parallel.

    Parameters
    ----------
    ts_bunch1_subset : list
        Subset of first time series to process.
    ts_bunch2 : list
        Full set of second time series (features).
    metric : str
        Similarity metric being used.
    mi_estimator : str
        MI estimator ('gcmi' or 'ksg').
    ds : int
        Downsampling factor.
    engine : str
        Computation engine: 'auto', 'fft', or 'loop'.

    Returns
    -------
    tuple
        (partial_cache, partial_fft_type_counts) for this subset.
    """
    cache = {}
    fft_type_counts = {}

    for ts1 in ts_bunch1_subset:
        for ts2 in ts_bunch2:
            fft_type = get_fft_type(ts1, ts2, metric, mi_estimator, 1, engine)
            key1 = _get_ts_key(ts1)
            key2 = _get_ts_key(ts2)

            if fft_type is not None:
                data1, data2 = _extract_fft_data(ts1, ts2, fft_type, ds)
                compute_fn = _FFT_COMPUTE[fft_type]
                n = len(data1) if data1.ndim == 1 else data1.shape[1]
                mi_all = compute_fn(data1, data2, np.arange(n))
                cache[(key1, key2)] = FFTCacheEntry(fft_type=fft_type, mi_all=mi_all)
                fft_type_counts[fft_type] = fft_type_counts.get(fft_type, 0) + 1
            else:
                cache[(key1, key2)] = None
                fft_type_counts['loop'] = fft_type_counts.get('loop', 0) + 1

    return cache, fft_type_counts


def _build_fft_cache_parallel(
    ts_bunch1: list,
    ts_bunch2: list,
    metric: str,
    mi_estimator: str,
    ds: int,
    engine: str,
    n_jobs: int = -1,
) -> tuple:
    """Parallel version of _build_fft_cache.

    Splits ts_bunch1 across workers, each builds a partial cache,
    then merges all results into a single cache dictionary.

    Parameters
    ----------
    ts_bunch1 : list
        First set of time series (e.g., neurons).
    ts_bunch2 : list
        Second set of time series (e.g., features).
    metric : str
        Similarity metric being used.
    mi_estimator : str
        MI estimator ('gcmi' or 'ksg').
    ds : int
        Downsampling factor.
    engine : str
        Computation engine: 'auto', 'fft', or 'loop'.
    n_jobs : int, optional
        Number of parallel jobs. Default: -1 (all CPU cores).

    Returns
    -------
    tuple
        (merged_cache, merged_fft_type_counts) from all workers.
    """
    if n_jobs == -1:
        n_jobs = min(multiprocessing.cpu_count(), len(ts_bunch1))
    n_jobs_effective = min(n_jobs, len(ts_bunch1))

    # Split ts_bunch1 across workers
    split_inds = np.array_split(np.arange(len(ts_bunch1)), n_jobs_effective)
    split_ts_bunch1 = [[ts_bunch1[i] for i in idxs] for idxs in split_inds if len(idxs) > 0]

    # Parallel execution with backend-specific config
    with _parallel_executor(n_jobs_effective) as parallel:
        results = parallel(
            delayed(_build_fft_cache_worker)(
                subset, ts_bunch2, metric, mi_estimator, ds, engine
            )
            for subset in split_ts_bunch1
        )

    # Merge results
    merged_cache = {}
    merged_counts = {}
    for partial_cache, partial_counts in results:
        merged_cache.update(partial_cache)
        for k, v in partial_counts.items():
            merged_counts[k] = merged_counts.get(k, 0) + v

    return merged_cache, merged_counts


def validate_time_series_bunches(ts_bunch1, ts_bunch2, allow_mixed_dimensions=True) -> None:
    """
    Validate time series bunches for INTENSE computations.

    Parameters
    ----------
    ts_bunch1 : list of TimeSeries or MultiTimeSeries
        First set of time series objects (e.g., neural activity).
    ts_bunch2 : list of TimeSeries or MultiTimeSeries
        Second set of time series objects (e.g., behavioral features).
    allow_mixed_dimensions : bool, default=True
        Whether to allow mixed TimeSeries and MultiTimeSeries objects.
        If False, all objects must be TimeSeries.

        .. deprecated:: 1.1
            This parameter is deprecated and will be removed in a future version.
            Mixed dimensions are now always allowed.

    Raises
    ------
    ValueError
        If bunches are empty, contain wrong types, or have mismatched lengths.

    Notes
    -----
    When allow_mixed_dimensions=False, both bunches must contain only TimeSeries
    objects. All time series within each bunch must have the same length, and
    both bunches must have matching lengths."""
    if len(ts_bunch1) == 0:
        raise ValueError("ts_bunch1 cannot be empty")
    if len(ts_bunch2) == 0:
        raise ValueError("ts_bunch2 cannot be empty")

    # Check time series types
    if not allow_mixed_dimensions:
        ts1_types = [type(ts) for ts in ts_bunch1]
        ts2_types = [type(ts) for ts in ts_bunch2]

        if not all(issubclass(t, TimeSeries) for t in ts1_types):
            if any(issubclass(t, MultiTimeSeries) for t in ts1_types):
                raise ValueError(
                    "MultiTimeSeries found in ts_bunch1 but allow_mixed_dimensions=False"
                )
            else:
                raise ValueError("ts_bunch1 must contain TimeSeries objects")

        if not all(issubclass(t, TimeSeries) for t in ts2_types):
            if any(issubclass(t, MultiTimeSeries) for t in ts2_types):
                raise ValueError(
                    "MultiTimeSeries found in ts_bunch2 but allow_mixed_dimensions=False"
                )
            else:
                raise ValueError("ts_bunch2 must contain TimeSeries objects")

    # Check lengths match
    lengths1 = [
        len(ts.data) if isinstance(ts, TimeSeries) else ts.data.shape[1] for ts in ts_bunch1
    ]
    lengths2 = [
        len(ts.data) if isinstance(ts, TimeSeries) else ts.data.shape[1] for ts in ts_bunch2
    ]

    if len(set(lengths1)) > 1:
        raise ValueError(f"All time series in ts_bunch1 must have same length, got {set(lengths1)}")
    if len(set(lengths2)) > 1:
        raise ValueError(f"All time series in ts_bunch2 must have same length, got {set(lengths2)}")
    if lengths1[0] != lengths2[0]:
        raise ValueError(f"Time series lengths don't match: {lengths1[0]} vs {lengths2[0]}")


def validate_metric(metric, allow_scipy=True) -> str:
    """
    Validate metric name and check if it's supported.

    Parameters
    ----------
    metric : str
        Metric name to validate. Supported metrics:
        - 'mi': Mutual information (supports multivariate data)
        - 'av': Activity ratio (requires one binary and one continuous variable)
        - 'fast_pearsonr': Fast Pearson correlation implementation
        - 'spearmanr', 'pearsonr', 'kendalltau': scipy.stats correlation functions
        - Any other callable from scipy.stats (if allow_scipy=True)
    allow_scipy : bool, default=True
        Whether to allow scipy.stats correlation functions.

    Returns
    -------
    metric_type : str
        Type of metric:
        - 'mi': Mutual information metric
        - 'special': Special metrics ('av', 'fast_pearsonr')
        - 'scipy': scipy.stats functions

    Raises
    ------
    ValueError
        If metric is not supported or not a callable function in scipy.stats.

    Notes
    -----
    The function validates that scipy.stats attributes are callable to prevent
    accepting non-function attributes like constants or data arrays."""
    # Built-in metrics
    if metric == "mi":
        return "mi"

    # Special metrics
    if metric in ["av", "fast_pearsonr"]:
        return "special"

    # Full scipy names
    scipy_correlation_metrics = ["spearmanr", "pearsonr", "kendalltau"]
    if metric in scipy_correlation_metrics:
        return "scipy"

    # Check if it's a scipy function
    if allow_scipy:
        try:
            import scipy.stats

            attr = getattr(scipy.stats, metric, None)
            if attr is not None and callable(attr):
                return "scipy"
        except ImportError:
            pass

    # If we get here, metric is not supported
    raise ValueError(
        f"Unsupported metric: {metric}. Supported metrics include: "
        f"'mi', 'av', 'fast_pearsonr', 'spearmanr', 'pearsonr', 'kendalltau', "
        f"and other scipy.stats functions."
    )


def validate_common_parameters(shift_window=None, ds=None, nsh=None, noise_const=None) -> None:
    """
    Validate common INTENSE parameters.

    Parameters
    ----------
    shift_window : int, optional
        Maximum shift window in frames. Must be non-negative.
    ds : int, optional
        Downsampling factor. Must be positive integer.
    nsh : int, optional
        Number of shuffles for significance testing. Must be positive integer.
    noise_const : float, optional
        Noise constant for numerical stability. Must be non-negative.

    Raises
    ------
    TypeError
        If parameters have incorrect types (non-integer for shift_window, ds, nsh;
        non-numeric for noise_const).
    ValueError
        If parameters have invalid values (negative shift_window or noise_const;
        non-positive ds or nsh).

    Notes
    -----
    This function validates parameter types using isinstance checks for numpy
    compatibility (accepts both Python int and numpy integer types)."""
    if shift_window is not None:
        if not isinstance(shift_window, (int, np.integer)):
            raise TypeError(f"shift_window must be integer, got {type(shift_window).__name__}")
        if shift_window < 0:
            raise ValueError(f"shift_window must be non-negative, got {shift_window}")

    if ds is not None:
        if not isinstance(ds, (int, np.integer)):
            raise TypeError(f"ds must be integer, got {type(ds).__name__}")
        if ds <= 0:
            raise ValueError(f"ds must be positive, got {ds}")

    if nsh is not None:
        if not isinstance(nsh, (int, np.integer)):
            raise TypeError(f"nsh must be integer, got {type(nsh).__name__}")
        if nsh <= 0:
            raise ValueError(f"nsh must be positive, got {nsh}")

    if noise_const is not None:
        if not isinstance(noise_const, (int, float, np.number)):
            raise TypeError(f"noise_const must be numeric, got {type(noise_const).__name__}")
        if noise_const < 0:
            raise ValueError(f"noise_const must be non-negative, got {noise_const}")


def calculate_optimal_delays(
    ts_bunch1,
    ts_bunch2,
    metric,
    shift_window,
    ds,
    verbose=True,
    enable_progressbar=True,
    mi_estimator="gcmi",
    engine="auto",
    fft_cache: dict = None,
    mi_estimator_kwargs=None,
) -> np.ndarray:
    """
    Calculate optimal temporal delays between pairs of time series.

    Finds the delay that maximizes the similarity metric between each pair of time series
    from ts_bunch1 and ts_bunch2. This accounts for temporal offsets in neural responses
    relative to behavioral variables.

    Parameters
    ----------
    ts_bunch1 : list of TimeSeries
        First set of time series (typically neural signals).
    ts_bunch2 : list of TimeSeries
        Second set of time series (typically behavioral variables).
    metric : str
        Similarity metric to maximize. See validate_metric for supported options.
    shift_window : int
        Maximum shift to test in each direction (frames).
        Will test shifts from -shift_window to +shift_window inclusive.
    ds : int
        Downsampling factor. Every ds-th point is used from the time series.
    verbose : bool, default=True
        Whether to print progress information.
    enable_progressbar : bool, default=True
        Whether to show progress bar.
    mi_estimator : str, default='gcmi'
        MI estimator to use when metric='mi'. Options: 'gcmi' or 'ksg'.
    engine : {'auto', 'fft', 'loop'}, default='auto'
        Computation engine for delay optimization:
        - 'auto': Use FFT when applicable (univariate continuous GCMI with >= 20 shifts)
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
    optimal_delays : np.ndarray of shape (len(ts_bunch1), len(ts_bunch2))
        Optimal delay (in frames) for each pair. Positive values indicate
        that ts2 leads ts1, negative values indicate ts1 leads ts2.

    Notes
    -----
    - With FFT engine: O(n1 * n2 * n log n) where n is downsampled time series length
    - With loop engine: O(n1 * n2 * shifts * n) where shifts = 2 * shift_window / ds
    - FFT provides ~10-20x speedup for typical delay windows (100-200 shifts)
    - The optimal delay is found by exhaustive search over all possible shifts
    - Memory efficient: only stores final optimal delays, not all tested values

    Examples
    --------
    >>> # Create minimal time series with known phase shift
    >>> import numpy as np
    >>> from driada.information.info_base import TimeSeries
    >>> # 100 points is enough to demonstrate functionality
    >>> t = np.linspace(0, 2*np.pi, 100)
    >>> # Create two sine waves with 5-sample phase shift
    >>> data1 = np.sin(t)
    >>> data2 = np.sin(t + np.pi/4)  # phase shifted signal
    >>> ts1 = TimeSeries(data1, discrete=False)
    >>> ts2 = TimeSeries(data2, discrete=False)
    >>> # Find optimal delay with small window for speed
    >>> delays = calculate_optimal_delays([ts1], [ts2], 'mi',
    ...                                   shift_window=5, ds=1, verbose=False)
    >>> delays.shape
    (1, 1)
    >>> # The delay captures the phase relationship
    >>> -5 <= delays[0, 0] <= 5
    True"""
    # Validate inputs
    validate_time_series_bunches(ts_bunch1, ts_bunch2, allow_mixed_dimensions=True)
    validate_metric(metric)
    validate_common_parameters(shift_window=shift_window, ds=ds)

    if verbose:
        print("Calculating optimal delays:")

    optimal_delays = np.zeros((len(ts_bunch1), len(ts_bunch2)), dtype=int)
    shifts = np.arange(-shift_window, shift_window + ds, ds) // ds

    # Pre-compute FFT types for each feature (only if no cache provided)
    if fft_cache is None and len(ts_bunch1) > 0:
        fft_types = [
            get_fft_type(
                ts_bunch1[0], ts2, metric, mi_estimator, len(shifts), engine, for_delays=True
            )
            for ts2 in ts_bunch2
        ]
    else:
        fft_types = None

    for i, ts1 in tqdm.tqdm(
        enumerate(ts_bunch1), total=len(ts_bunch1), disable=not enable_progressbar
    ):
        for j, ts2 in enumerate(ts_bunch2):
            # Check cache first (uses stable keys)
            cache_entry = fft_cache.get(
                (_get_ts_key(ts1), _get_ts_key(ts2))
            ) if fft_cache else None

            if cache_entry is not None:
                # Use cached MI values - just index!
                mi_all = cache_entry.mi_all
                n = len(mi_all)

                # Convert shifts to non-negative indices
                fft_shifts = np.where(shifts >= 0, shifts, n + shifts).astype(int)

                # Look up MI at requested shifts
                mi_values = mi_all[fft_shifts]

                best_idx = np.argmax(mi_values)
                optimal_delays[i, j] = int(shifts[best_idx] * ds)
            elif fft_types is not None and fft_types[j] is not None:
                # No cache but FFT applicable - extract data fresh
                fft_type = fft_types[j]
                data1, data2 = _extract_fft_data(ts1, ts2, fft_type, ds)
                compute_fn = _FFT_COMPUTE[fft_type]

                n = len(data1) if data1.ndim == 1 else data1.shape[-1]
                fft_shifts = np.where(shifts >= 0, shifts, n + shifts).astype(int)
                mi_values = compute_fn(data1, data2, fft_shifts)

                best_idx = np.argmax(mi_values)
                optimal_delays[i, j] = int(shifts[best_idx] * ds)
            else:
                # Loop fallback
                shifted_me = []
                for shift in shifts:
                    lag_me = get_sim(
                        ts1, ts2, metric, ds=ds, shift=int(shift), estimator=mi_estimator,
                        mi_estimator_kwargs=mi_estimator_kwargs,
                    )
                    shifted_me.append(lag_me)

                best_shift = shifts[np.argmax(shifted_me)]
                optimal_delays[i, j] = int(best_shift * ds)

    return optimal_delays


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


def calculate_optimal_delays_parallel(
    ts_bunch1,
    ts_bunch2,
    metric,
    shift_window,
    ds,
    verbose=True,
    n_jobs=-1,
    mi_estimator="gcmi",
    engine="auto",
    fft_cache: dict = None,
    mi_estimator_kwargs=None,
) -> np.ndarray:
    """
    Calculate optimal temporal delays between pairs of time series using parallel processing.

    Parallel version of calculate_optimal_delays that distributes computation across
    multiple CPU cores for improved performance with large datasets.

    Parameters
    ----------
    ts_bunch1 : list of TimeSeries
        First set of time series (typically neural signals).
    ts_bunch2 : list of TimeSeries
        Second set of time series (typically behavioral variables).
    metric : str
        Similarity metric to maximize. See validate_metric for supported options.
    shift_window : int
        Maximum shift to test in each direction (frames).
        Will test shifts from -shift_window to +shift_window inclusive.
    ds : int
        Downsampling factor. Every ds-th point is used from the time series.
    verbose : bool, default=True
        Whether to print progress information.
    n_jobs : int, default=-1
        Number of parallel jobs to run. -1 uses all available cores.
    mi_estimator : str, default='gcmi'
        MI estimator to use when metric='mi'. Options: 'gcmi' or 'ksg'.
    engine : {'auto', 'fft', 'loop'}, default='auto'
        Computation engine for delay optimization:
        - 'auto': Use FFT when applicable (univariate continuous GCMI with >= 20 shifts)
        - 'fft': Force FFT (raises error if not applicable)
        - 'loop': Force per-shift loop (original behavior)
    fft_cache : dict, optional
        Pre-computed FFT cache from _build_fft_cache. Keys are (key1, key2) tuples
        using stable identifiers from _get_ts_key(). Passed to each worker for cache reuse.
    mi_estimator_kwargs : dict, optional
        Additional keyword arguments passed to the MI estimator function.

    Returns
    -------
    optimal_delays : np.ndarray of shape (len(ts_bunch1), len(ts_bunch2))
        Optimal delay (in frames) for each pair. Positive values indicate
        that ts2 leads ts1, negative values indicate ts1 leads ts2.

    Notes
    -----
    - Parallelization is done by splitting ts_bunch1 across workers
    - Each worker processes a subset of ts_bunch1 against all of ts_bunch2
    - Memory usage scales with number of workers
    - Speedup is typically sublinear due to overhead and memory bandwidth
    - FFT optimization within each worker provides additional speedup

    See Also
    --------
    ~driada.intense.intense_base.calculate_optimal_delays : Sequential version of this function

    Examples
    --------
    >>> # Demonstrate parallel processing with minimal data
    >>> import numpy as np
    >>> from driada.information.info_base import TimeSeries
    >>> # Create 3 neurons and 2 behaviors with 50 timepoints each
    >>> np.random.seed(42)  # For reproducible example
    >>> neurons = [TimeSeries(np.random.randn(50), discrete=False) for _ in range(3)]
    >>> behaviors = [TimeSeries(np.random.randn(50), discrete=False) for _ in range(2)]
    >>> # Use 2 cores with small shift window
    >>> delays = calculate_optimal_delays_parallel(neurons, behaviors, 'mi',
    ...                                           shift_window=3, ds=1, n_jobs=2, verbose=False)
    >>> delays.shape
    (3, 2)
    >>> # All delays should be within the window
    >>> np.all(np.abs(delays) <= 3)
    True"""
    # Validate inputs
    validate_time_series_bunches(ts_bunch1, ts_bunch2, allow_mixed_dimensions=True)
    validate_metric(metric)
    validate_common_parameters(shift_window=shift_window, ds=ds)

    if verbose:
        print("Calculating optimal delays in parallel mode:")

    optimal_delays = np.zeros((len(ts_bunch1), len(ts_bunch2)), dtype=int)

    if n_jobs == -1:
        n_jobs = min(multiprocessing.cpu_count(), len(ts_bunch1))

    # Limit n_jobs to number of items to avoid empty worker splits
    n_jobs_effective = min(n_jobs, len(ts_bunch1))
    if n_jobs_effective < n_jobs and verbose:
        import warnings
        warnings.warn(
            f"Requested {n_jobs} parallel jobs but only {len(ts_bunch1)} items to process. "
            f"Using {n_jobs_effective} workers to avoid empty splits.",
            UserWarning
        )

    split_ts_bunch1_inds = np.array_split(np.arange(len(ts_bunch1)), n_jobs_effective)
    split_ts_bunch1 = [np.array(ts_bunch1)[idxs] for idxs in split_ts_bunch1_inds]

    # Split cache per worker - each worker only gets entries it needs
    # This avoids serializing the entire cache (potentially GBs) to each worker
    split_caches = [
        _extract_cache_subset(fft_cache, subset, ts_bunch2)
        for subset in split_ts_bunch1
    ]

    # Parallel execution with backend-specific config
    with _parallel_executor(n_jobs_effective) as parallel:
        parallel_delays = parallel(
            delayed(calculate_optimal_delays)(
                small_ts_bunch,
                ts_bunch2,
                metric,
                shift_window,
                ds,
                verbose=False,
                enable_progressbar=False,
                mi_estimator=mi_estimator,
                engine=engine,
                fft_cache=split_cache,
                mi_estimator_kwargs=mi_estimator_kwargs,
            )
            for small_ts_bunch, split_cache in zip(split_ts_bunch1, split_caches)
        )

    for i, pd in enumerate(parallel_delays):
        inds_of_interest = split_ts_bunch1_inds[i]
        optimal_delays[inds_of_interest, :] = pd

    return optimal_delays


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

    # Note: Per-pair deterministic seeding is used via np.random.RandomState(pair_seed)
    # where pair_seed = seed + global_i * 10000 + j * 100
    # This ensures reproducibility without polluting global RNG state

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
            for j, ts2 in enumerate(ts_bunch2):
                if mask[i, j] == 1:
                    key1 = _get_ts_key(ts1)
                    key2 = _get_ts_key(ts2)

                    # Check cache first, then compute FFT type if needed (using stable keys)
                    cache_entry = fft_cache.get((key1, key2)) if fft_cache else None

                    if cache_entry is not None:
                        # Use cached MI values - just index!
                        mi_all = cache_entry.mi_all

                        opt_shift = int(optimal_delays[i, j]) // ds
                        me0 = mi_all[opt_shift]
                        shuffle_mis = mi_all[random_shifts[i, j, :]]

                        # Add pre-generated noise for numerical stability
                        me_table[i, j] = me0 + _noise_true[i, j]
                        me_table_shuffles[i, j, :] = shuffle_mis + _noise_shuffles[i, j, :]

                    elif fft_cache is None:
                        # No cache provided — need per-pair RNG for noise
                        pair_seed = seed + hash((key1, key2)) % 1000000 if seed is not None else None
                        pair_rng = np.random.RandomState(pair_seed)
                        # Compute FFT type fresh
                        fft_type = get_fft_type(ts1, ts2, metric, mi_estimator, nsh, engine)

                        if fft_type is not None:
                            # Unified FFT-accelerated path
                            data1, data2 = _extract_fft_data(ts1, ts2, fft_type, ds)
                            compute_fn = _FFT_COMPUTE[fft_type]

                            # Compute true MI at optimal delay
                            opt_shift = optimal_delays[i, j] // ds
                            me0 = compute_fn(data1, data2, np.array([opt_shift]))[0]

                            # Compute all shuffle MIs at once
                            shuffle_mis = compute_fn(data1, data2, random_shifts[i, j, :])

                            # Add noise for numerical stability
                            me_table[i, j] = me0 + pair_rng.random() * noise_const
                            random_noise = pair_rng.random(size=nsh) * noise_const
                            me_table_shuffles[i, j, :] = shuffle_mis + random_noise

                        else:
                            # Original loop path (no FFT available)
                            me0 = get_sim(
                                ts1,
                                ts2,
                                metric,
                                ds=ds,
                                shift=optimal_delays[i, j] // ds,
                                estimator=mi_estimator,
                                check_for_coincidence=True,
                                mi_estimator_kwargs=mi_estimator_kwargs,
                            )  # default metric without shuffling

                            me_table[i, j] = (
                                me0 + pair_rng.random() * noise_const
                            )  # add small noise for better fitting

                            random_noise = (
                                pair_rng.random(size=nsh) * noise_const
                            )  # add small noise for better fitting

                            for k, shift in enumerate(random_shifts[i, j, :]):
                                me = get_sim(
                                    ts1, ts2, metric, ds=ds, shift=shift, estimator=mi_estimator,
                                    mi_estimator_kwargs=mi_estimator_kwargs,
                                )
                                me_table_shuffles[i, j, k] = me + random_noise[k]

                    else:
                        # Cache provided but entry is None — need per-pair RNG for noise
                        pair_seed = seed + hash((key1, key2)) % 1000000 if seed is not None else None
                        pair_rng = np.random.RandomState(pair_seed)
                        # Loop fallback required
                        me0 = get_sim(
                            ts1,
                            ts2,
                            metric,
                            ds=ds,
                            shift=optimal_delays[i, j] // ds,
                            estimator=mi_estimator,
                            check_for_coincidence=True,
                            mi_estimator_kwargs=mi_estimator_kwargs,
                        )

                        me_table[i, j] = me0 + pair_rng.random() * noise_const

                        random_noise = pair_rng.random(size=nsh) * noise_const

                        for k, shift in enumerate(random_shifts[i, j, :]):
                            me = get_sim(
                                ts1, ts2, metric, ds=ds, shift=shift, estimator=mi_estimator,
                                mi_estimator_kwargs=mi_estimator_kwargs,
                            )
                            me_table_shuffles[i, j, k] = me + random_noise[k]

                else:
                    me_table[i, j] = None
                    me_table_shuffles[i, j, :] = np.full(nsh, None)

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


def get_multicomp_correction_thr(fwer, mode="holm", **multicomp_kwargs) -> float:
    """
    Calculate p-value threshold for multiple hypothesis correction.

    Parameters
    ----------
    fwer : float
        Family-wise error rate or false discovery rate (e.g., 0.05).
        Must be between 0 and 1.
    mode : str or None, optional
        Multiple comparison correction method. Default: 'holm'.
        - None: No correction, threshold = fwer
        - 'bonferroni': Bonferroni correction (FWER control)
        - 'holm': Holm-Bonferroni correction (FWER control, more powerful)
        - 'fdr_bh': Benjamini-Hochberg FDR correction
    **multicomp_kwargs
        Additional arguments for correction method:
        - For 'bonferroni': nhyp (int) - number of hypotheses, must be > 0
        - For 'holm': all_pvals (list) - all p-values to be tested
        - For 'fdr_bh': all_pvals (list) - all p-values to be tested

    Returns
    -------
    threshold : float
        Adjusted p-value threshold for individual hypothesis testing.
        Returns 0 if no p-values pass the correction criteria (reject all).

    Raises
    ------
    ValueError
        If fwer is not between 0 and 1.
        If required arguments are missing or invalid.
        If unknown method specified.
        If nhyp <= 0 for bonferroni method.
        If all_pvals is empty for holm or fdr_bh methods.

    Notes
    -----
    - FWER methods (bonferroni, holm) control probability of ANY false positive
    - FDR methods control expected proportion of false positives among rejections
    - Holm is uniformly more powerful than Bonferroni
    - FDR typically allows more discoveries but with controlled false positive rate
    - If no p-values satisfy the correction criteria, threshold is set to 0
      (reject all hypotheses)

    Examples
    --------
    >>> # Compare different multiple comparison correction methods
    >>> pvals = [0.001, 0.01, 0.02, 0.03, 0.04]
    >>>
    >>> # No correction - uses raw threshold
    >>> thr_none = get_multicomp_correction_thr(0.05, mode=None)
    >>> thr_none
    0.05
    >>>
    >>> # Bonferroni correction - most conservative
    >>> thr_bonf = get_multicomp_correction_thr(0.05, mode='bonferroni', nhyp=5)
    >>> thr_bonf
    0.01
    >>>
    >>> # Holm correction - less conservative than Bonferroni
    >>> thr_holm = get_multicomp_correction_thr(0.05, mode='holm', all_pvals=pvals)
    >>> round(thr_holm, 4)
    0.0125
    >>>
    >>> # FDR correction - controls false discovery rate
    >>> thr_fdr = get_multicomp_correction_thr(0.05, mode='fdr_bh', all_pvals=pvals)
    >>> thr_fdr
    0.04
    >>>
    >>> # FDR is least conservative: bonf < holm < fdr < none
    >>> thr_bonf < thr_holm < thr_fdr < thr_none
    True"""
    # Validate fwer parameter
    if not 0 <= fwer <= 1:
        raise ValueError(f"fwer must be between 0 and 1, got {fwer}")

    if mode is None:
        threshold = fwer

    elif mode == "bonferroni":
        if "nhyp" not in multicomp_kwargs:
            raise ValueError("Number of hypotheses for Bonferroni correction not provided")
        nhyp = multicomp_kwargs["nhyp"]
        if nhyp <= 0:
            raise ValueError(f"Number of hypotheses must be positive, got {nhyp}")
        threshold = fwer / nhyp

    elif mode == "holm":
        if "all_pvals" not in multicomp_kwargs:
            raise ValueError("List of p-values for Holm correction not provided")
        all_pvals = multicomp_kwargs["all_pvals"]
        if len(all_pvals) == 0:
            raise ValueError("Empty p-value list provided for Holm correction")
        all_pvals = sorted(all_pvals)
        nhyp = len(all_pvals)
        threshold = 0  # Default if no discoveries (reject all)
        for i, pval in enumerate(all_pvals):
            cthr = fwer / (nhyp - i)
            if pval > cthr:
                break
            threshold = cthr

    elif mode == "fdr_bh":
        if "all_pvals" not in multicomp_kwargs:
            raise ValueError("List of p-values for FDR correction not provided")
        all_pvals = multicomp_kwargs["all_pvals"]
        if len(all_pvals) == 0:
            raise ValueError("Empty p-value list provided for FDR correction")
        all_pvals = sorted(all_pvals)
        nhyp = len(all_pvals)
        threshold = 0.0  # Default if no discoveries (reject all)

        # Benjamini-Hochberg procedure
        for i in range(nhyp - 1, -1, -1):
            if all_pvals[i] <= fwer * (i + 1) / nhyp:
                threshold = all_pvals[i]
                break

    else:
        raise ValueError("Unknown multiple comparisons correction method")

    return threshold
