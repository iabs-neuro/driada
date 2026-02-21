"""
FFT dispatch, data extraction, and caching for INTENSE.

Provides FFT type classification, data extraction helpers, and cache building
functions used by the INTENSE pipeline and delay optimization.
"""

import multiprocessing

import numpy as np
from dataclasses import dataclass
from joblib import delayed

from ..information.info_base import (
    TimeSeries,
    MultiTimeSeries,
    compute_mi_batch_fft,
    compute_mi_gd_fft,
    compute_mi_mts_fft,
    compute_mi_mts_mts_fft,
    compute_mi_mts_discrete_fft,
    compute_mi_dd_fft,
    compute_pearson_batch_fft,
    compute_av_batch_fft,
)
from ..utils.parallel import parallel_executor as _parallel_executor
from .validation import (
    validate_time_series_bunches,
    validate_metric,
    validate_common_parameters,
)

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
