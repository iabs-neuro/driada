"""
Delay optimization for INTENSE analysis.

Provides functions to find optimal temporal delays between pairs of time series
(e.g. neural signals and behavioral variables) by maximizing a similarity metric
across a range of shifts.
"""

import multiprocessing
import warnings

import numpy as np
import tqdm
from joblib import delayed

from ..information.info_base import get_sim
from ..utils.parallel import parallel_executor as _parallel_executor
from .validation import (
    validate_common_parameters,
    validate_metric,
    validate_time_series_bunches,
)
from .fft import (
    MIN_SHIFTS_FOR_FFT_DELAYS,
    _extract_fft_data,
    _FFT_COMPUTE,
    _get_ts_key,
    get_fft_type,
)
from .intense_base import _extract_cache_subset


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
    validate_time_series_bunches(ts_bunch1, ts_bunch2)
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
    ~driada.intense.delay.calculate_optimal_delays : Sequential version of this function

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
    validate_time_series_bunches(ts_bunch1, ts_bunch2)
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
