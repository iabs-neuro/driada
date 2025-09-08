import numpy as np
import tqdm
from joblib import Parallel, delayed
import multiprocessing

from .stats import (
    populate_nested_dict,
    get_table_of_stats,
    criterion1,
    criterion2,
    get_all_nonempty_pvals,
    merge_stage_stats,
    merge_stage_significance,
)
from ..information.info_base import TimeSeries, MultiTimeSeries, get_multi_mi, get_sim
from ..utils.data import write_dict_to_hdf5, nested_dict_to_seq_of_tables, add_names_to_nested_dict

# Configure joblib backend to avoid PyTorch forking issues
try:
    import torch

    # If PyTorch is available, use threading backend to avoid forking issues
    # This prevents "function '_has_torch_function' already has a docstring" errors
    JOBLIB_BACKEND = "threading"
except ImportError:
    # Default to loky if PyTorch not available
    JOBLIB_BACKEND = "loky"


def validate_time_series_bunches(ts_bunch1, ts_bunch2, allow_mixed_dimensions=False):
    """
    Validate time series bunches for INTENSE computations.

    Parameters
    ----------
    ts_bunch1 : list of TimeSeries or MultiTimeSeries
        First set of time series objects (e.g., neural activity).
    ts_bunch2 : list of TimeSeries or MultiTimeSeries
        Second set of time series objects (e.g., behavioral features).
    allow_mixed_dimensions : bool, default=False
        Whether to allow mixed TimeSeries and MultiTimeSeries objects.
        If False, all objects must be TimeSeries.

    Raises
    ------
    ValueError
        If bunches are empty, contain wrong types, or have mismatched lengths.
        
    Notes
    -----
    When allow_mixed_dimensions=False, both bunches must contain only TimeSeries
    objects. All time series within each bunch must have the same length, and
    both bunches must have matching lengths.    """
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
        len(ts.data) if isinstance(ts, TimeSeries) else ts.data.shape[1]
        for ts in ts_bunch1
    ]
    lengths2 = [
        len(ts.data) if isinstance(ts, TimeSeries) else ts.data.shape[1]
        for ts in ts_bunch2
    ]

    if len(set(lengths1)) > 1:
        raise ValueError(
            f"All time series in ts_bunch1 must have same length, got {set(lengths1)}"
        )
    if len(set(lengths2)) > 1:
        raise ValueError(
            f"All time series in ts_bunch2 must have same length, got {set(lengths2)}"
        )
    if lengths1[0] != lengths2[0]:
        raise ValueError(
            f"Time series lengths don't match: {lengths1[0]} vs {lengths2[0]}"
        )


def validate_metric(metric, allow_scipy=True):
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
    accepting non-function attributes like constants or data arrays.    """
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


def validate_common_parameters(shift_window=None, ds=None, nsh=None, noise_const=None):
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
    compatibility (accepts both Python int and numpy integer types).    """
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
):
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

    Returns
    -------
    optimal_delays : np.ndarray of shape (len(ts_bunch1), len(ts_bunch2))
        Optimal delay (in frames) for each pair. Positive values indicate
        that ts2 leads ts1, negative values indicate ts1 leads ts2.

    Notes
    -----
    - Computational complexity: O(n1 * n2 * shifts) where n1, n2 are lengths
      of ts_bunch1 and ts_bunch2, and shifts = 2 * shift_window / ds
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
    validate_time_series_bunches(ts_bunch1, ts_bunch2, allow_mixed_dimensions=False)
    validate_metric(metric)
    validate_common_parameters(shift_window=shift_window, ds=ds)

    if verbose:
        print("Calculating optimal delays:")

    optimal_delays = np.zeros((len(ts_bunch1), len(ts_bunch2)), dtype=int)
    shifts = np.arange(-shift_window, shift_window + ds, ds) // ds

    for i, ts1 in tqdm.tqdm(
        enumerate(ts_bunch1), total=len(ts_bunch1), disable=not enable_progressbar
    ):
        for j, ts2 in enumerate(ts_bunch2):
            shifted_me = []
            for shift in shifts:
                lag_me = get_sim(ts1, ts2, metric, ds=ds, shift=int(shift))
                shifted_me.append(lag_me)

            best_shift = shifts[np.argmax(shifted_me)]
            optimal_delays[i, j] = int(best_shift * ds)

    return optimal_delays


def calculate_optimal_delays_parallel(
    ts_bunch1, ts_bunch2, metric, shift_window, ds, verbose=True, n_jobs=-1
):
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
    validate_time_series_bunches(ts_bunch1, ts_bunch2, allow_mixed_dimensions=False)
    validate_metric(metric)
    validate_common_parameters(shift_window=shift_window, ds=ds)

    if verbose:
        print("Calculating optimal delays in parallel mode:")

    optimal_delays = np.zeros((len(ts_bunch1), len(ts_bunch2)), dtype=int)

    if n_jobs == -1:
        n_jobs = min(multiprocessing.cpu_count(), len(ts_bunch1))

    split_ts_bunch1_inds = np.array_split(np.arange(len(ts_bunch1)), n_jobs)
    split_ts_bunch1 = [np.array(ts_bunch1)[idxs] for idxs in split_ts_bunch1_inds]

    parallel_delays = Parallel(n_jobs=n_jobs, backend=JOBLIB_BACKEND, verbose=True)(
        delayed(calculate_optimal_delays)(
            small_ts_bunch,
            ts_bunch2,
            metric,
            shift_window,
            ds,
            verbose=False,
            enable_progressbar=False,
        )
        for small_ts_bunch in split_ts_bunch1
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
    window=1000,
    ds=1,
    metric="mi",
    mi_estimator="gcmi",
    data_type="calcium",
):
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
    window : int, optional
        Maximum shift to test in each direction (frames). Default: 1000.
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
    - Total number of shifts tested: 2 * window / ds
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

    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")

    # Check if single cell/feature mode (backward compatibility)
    single_mode = (
        cell_id is not None
        and feat_id is not None
        and cbunch is None
        and fbunch is None
    )

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
    mi_estimator="gcmi",
    joint_distr=False,
    ds=1,
    mask=None,
    noise_const=1e-3,
    seed=None,
    allow_mixed_dimensions=False,
    enable_progressbar=True,
    start_index=0,
):
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
    allow_mixed_dimensions : bool, default=False
        Whether to allow mixed TimeSeries and MultiTimeSeries objects.
    enable_progressbar : bool, default=True
        Whether to show progress bar during computation.
    start_index : int, default=0
        Global starting index for ts_bunch1. Used internally by parallel processing
        to ensure deterministic random number generation.

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
    - Noise is added as: value * (1 + noise_const * U(-1,1))    """

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

    np.random.seed(seed)

    lengths1 = [
        len(ts.data) if isinstance(ts, TimeSeries) else ts.data.shape[1]
        for ts in ts_bunch1
    ]
    lengths2 = [
        len(ts.data) if isinstance(ts, TimeSeries) else ts.data.shape[1]
        for ts in ts_bunch2
    ]
    if (
        len(set(lengths1)) == 1
        and len(set(lengths2)) == 1
        and set(lengths1) == set(lengths2)
    ):
        t = lengths1[0]  # full length is the same for all time series
    else:
        raise ValueError("Lenghts of TimeSeries do not match!")

    if mask is None:
        mask = np.ones((n1, n2))

    me_table = np.zeros((n1, n2))
    me_table_shuffles = np.zeros((n1, n2, nsh))
    random_shifts = np.zeros((n1, n2, nsh), dtype=int)

    # fill random shifts according to the allowed shuffles masks of both time series
    for i, ts1 in enumerate(ts_bunch1):
        # Use global index for deterministic seeding
        global_i = start_index + i
        
        if joint_distr:
            # Create deterministic seed for this specific (global_i, 0) pair
            # This ensures same results regardless of parallel execution
            pair_seed = seed + global_i * 10000 if seed is not None else None
            pair_rng = np.random.RandomState(pair_seed)
            
            # Combine shuffle masks from ts1 and all ts in tsbunch2
            combined_shuffle_mask = ts1.shuffle_mask.copy()
            for ts2 in ts_bunch2:
                combined_shuffle_mask = combined_shuffle_mask & ts2.shuffle_mask
            # move shuffle mask according to optimal shift
            combined_shuffle_mask = np.roll(
                combined_shuffle_mask, int(optimal_delays[i, 0])
            )
            indices_to_select = np.arange(t)[combined_shuffle_mask]
            random_shifts[i, 0, :] = pair_rng.choice(indices_to_select, size=nsh) // ds

        else:
            for j, ts2 in enumerate(ts_bunch2):
                # Create deterministic seed for this specific (global_i, j) pair
                # This ensures same results regardless of parallel execution
                pair_seed = seed + global_i * 10000 + j * 100 if seed is not None else None
                pair_rng = np.random.RandomState(pair_seed)
                
                combined_shuffle_mask = ts1.shuffle_mask & ts2.shuffle_mask
                # move shuffle mask according to optimal shift
                combined_shuffle_mask = np.roll(
                    combined_shuffle_mask, int(optimal_delays[i, j])
                )
                indices_to_select = np.arange(t)[combined_shuffle_mask]
                random_shifts[i, j, :] = (
                    pair_rng.choice(indices_to_select, size=nsh) // ds
                )

    # calculate similarity metric arrays
    for i, ts1 in tqdm.tqdm(
        enumerate(ts_bunch1),
        total=len(ts_bunch1),
        position=0,
        leave=True,
        disable=not enable_progressbar,
    ):
        # Use global index for deterministic seeding
        global_i = start_index + i

        # DEPRECATED: This joint_distr branch is deprecated and will be removed in v2.0
        # Use MultiTimeSeries for joint distribution handling instead
        # FUTURE: Remove this entire branch in v2.0
        if joint_distr:
            if metric != "mi":
                raise ValueError("joint_distr mode works with metric = 'mi' only")
            if mask[i, 0] == 1:
                # default metric without shuffling, minus due to different order
                me0 = get_multi_mi(
                    ts_bunch2, ts1, ds=ds, shift=-optimal_delays[i, 0] // ds, estimator=mi_estimator
                )
                # Use deterministic RNG for this pair
                pair_seed = seed + global_i * 10000 if seed is not None else None
                pair_rng = np.random.RandomState(pair_seed)
                
                me_table[i, 0] = (
                    me0 + pair_rng.random() * noise_const
                )  # add small noise for better fitting

                random_noise = (
                    pair_rng.random(size=len(random_shifts[i, 0, :])) * noise_const
                )  # add small noise for better fitting
                for k, shift in enumerate(random_shifts[i, 0, :]):
                    mi = get_multi_mi(ts_bunch2, ts1, ds=ds, shift=shift, estimator=mi_estimator)
                    me_table_shuffles[i, 0, k] = mi + random_noise[k]

            else:
                me_table[i, 0] = None
                me_table_shuffles[i, 0, :] = np.full(shape=nsh, fill_value=None)

        else:
            for j, ts2 in enumerate(ts_bunch2):
                if mask[i, j] == 1:
                    me0 = get_sim(
                        ts1,
                        ts2,
                        metric,
                        ds=ds,
                        shift=optimal_delays[i, j] // ds,
                        estimator=mi_estimator,
                        check_for_coincidence=True,
                    )  # default metric without shuffling

                    # Use deterministic RNG for this pair
                    pair_seed = seed + global_i * 10000 + j * 100 if seed is not None else None
                    pair_rng = np.random.RandomState(pair_seed)
                    
                    me_table[i, j] = (
                        me0 + pair_rng.random() * noise_const
                    )  # add small noise for better fitting

                    random_noise = (
                        pair_rng.random(size=len(random_shifts[i, j, :])) * noise_const
                    )  # add small noise for better fitting

                    for k, shift in enumerate(random_shifts[i, j, :]):
                        # mi = get_1d_mi(ts1, ts2, shift=shift, ds=ds)
                        me = get_sim(ts1, ts2, metric, ds=ds, shift=shift, estimator=mi_estimator)

                        me_table_shuffles[i, j, k] = me + random_noise[k]

                else:
                    me_table[i, j] = None
                    me_table_shuffles[i, j, :] = np.array([None for _ in range(nsh)])

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
    allow_mixed_dimensions=False,
    ds=1,
    mask=None,
    noise_const=1e-3,
    seed=None,
    n_jobs=-1,
):
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
    allow_mixed_dimensions : bool, default=False
        Whether to allow mixed TimeSeries and MultiTimeSeries objects.
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
    - Uses JOBLIB_BACKEND global variable (threading if PyTorch present, else loky)
    - Random seeding ensures reproducibility across different mask configurations

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
    random_shifts = np.zeros((n1, n2, nsh), dtype=int)

    if n_jobs == -1:
        n_jobs = min(multiprocessing.cpu_count(), n1)

    # Initialize mask if None
    if mask is None:
        n1 = len(ts_bunch1)
        n2 = 1 if joint_distr else len(ts_bunch2)
        mask = np.ones((n1, n2))

    split_ts_bunch1_inds = np.array_split(np.arange(len(ts_bunch1)), n_jobs)
    split_ts_bunch1 = [np.array(ts_bunch1)[idxs] for idxs in split_ts_bunch1_inds]
    split_optimal_delays = [optimal_delays[idxs] for idxs in split_ts_bunch1_inds]
    split_mask = [mask[idxs] for idxs in split_ts_bunch1_inds]

    parallel_result = Parallel(n_jobs=n_jobs, backend=JOBLIB_BACKEND, verbose=True)(
        delayed(scan_pairs)(
            small_ts_bunch,
            ts_bunch2,
            metric,
            nsh,
            split_optimal_delays[worker_idx],
            mi_estimator,
            joint_distr=joint_distr,
            allow_mixed_dimensions=allow_mixed_dimensions,
            ds=ds,
            mask=split_mask[worker_idx],
            noise_const=noise_const,
            seed=seed,
            enable_progressbar=False,
            start_index=split_ts_bunch1_inds[worker_idx][0],
        )
        for worker_idx, small_ts_bunch in enumerate(split_ts_bunch1)
    )

    for i in range(n_jobs):
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
    allow_mixed_dimensions=False,
    ds=1,
    mask=None,
    noise_const=1e-3,
    seed=None,
    enable_parallelization=True,
    n_jobs=-1,
):
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
    allow_mixed_dimensions : bool, default=False
        Whether to allow mixed TimeSeries and MultiTimeSeries objects.
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
        )

    else:
        random_shifts, me_total = scan_pairs(
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
            seed=seed,
            noise_const=noise_const,
        )

    return random_shifts, me_total


class IntenseResults(object):
    """
    Container for INTENSE computation results.

    Attributes
    ----------
    info : dict
        Metadata about the computation (optimal delays, thresholds, etc.).
    intense_params : dict
        Parameters used for the INTENSE computation.
    stats : dict
        Statistical results (p-values, metric values, etc.).
    significance : dict
        Significance test results for each neuron-feature pair.

    Methods
    -------
    update(property_name, data)
        Add or update a property with data.
    update_multiple(datadict)
        Update multiple properties from a dictionary.
    save_to_hdf5(fname)
        Save all results to an HDF5 file.

    Examples
    --------
    >>> # Create results container and add analysis outputs
    >>> results = IntenseResults()
    >>> # Add statistical results
    >>> results.update('stats', {'neuron1': {'feature1': {'me': 0.5, 'pval': 0.01}}})
    >>> # Add computation metadata
    >>> results.update('info', {'optimal_delays': [[0, 5], [10, 0]],
    ...                        'n_shuffles': 1000})
    >>> # Access stored data
    >>> results.stats['neuron1']['feature1']['me']
    0.5
    >>> # Save results (commented to avoid file creation in doctest)
    >>> # results.save_to_hdf5('analysis_results.h5')"""

    def __init__(self):
        """
        Initialize an empty IntenseResults container.
        
        Creates an IntenseResults object with no initial data. Properties are added
        dynamically using the update() or update_multiple() methods.
        
        Notes
        -----
        The IntenseResults class serves as a flexible container for storing INTENSE
        computation outputs. It allows dynamic addition of properties to accommodate
        different analysis configurations and results.
        
        Common properties added during INTENSE analysis:
        - 'stats': Statistical test results (p-values, metric values)
        - 'significance': Binary significance indicators
        - 'info': Computation metadata (delays, parameters used)
        - 'intense_params': Parameters used for the computation
        
        See Also
        --------
        ~driada.intense.pipelines.compute_cell_feat_significance : Main function that returns IntenseResults
        ~driada.intense.intense_base.update : Method to add properties to the results
        ~driada.intense.intense_base.save_to_hdf5 : Method to persist results to disk        """
        pass

    def update(self, property_name, data):
        """Add or update a property with data.
        
        Stores analysis results as attributes of the IntenseResults object,
        allowing flexible storage of various data types and structures.
        
        Parameters
        ----------
        property_name : str
            Name of the property to store. Will become an attribute of the
            object accessible via dot notation.
        data : any
            Data to store. Can be any Python object: arrays, dictionaries,
            dataframes, custom objects, etc.
            
        Examples
        --------
        >>> # Store different types of analysis results
        >>> import numpy as np
        >>> results = IntenseResults()
        >>> # Add mutual information matrix
        >>> results.update('mi_matrix', np.array([[0, 0.5], [0.5, 0]]))
        >>> # Add list of significant neuron-feature pairs
        >>> results.update('significant_pairs', [(0, 1), (2, 3)])
        >>> # Access via attribute notation
        >>> results.mi_matrix
        array([[0. , 0.5],
               [0.5, 0. ]])
        >>> results.significant_pairs
        [(0, 1), (2, 3)]
               
        Notes
        -----
        Property names should be valid Python identifiers. Existing properties
        will be overwritten without warning.        """
        setattr(self, property_name, data)

    def update_multiple(self, datadict):
        """Update multiple properties from a dictionary.
        
        Batch update of multiple properties at once, useful for storing
        related analysis results together.
        
        Parameters
        ----------
        datadict : dict
            Dictionary mapping property names to data values. Each key-value
            pair will be stored as an attribute.
            
        Examples
        --------
        >>> # Batch update multiple analysis results at once
        >>> import numpy as np
        >>> results = IntenseResults()
        >>> # Add multiple related results together
        >>> results.update_multiple({
        ...     'mi_values': np.array([0.1, 0.5, 0.3]),
        ...     'p_values': np.array([0.05, 0.001, 0.02]),
        ...     'significant': np.array([False, True, True]),
        ...     'parameters': {'metric': 'mi', 'correction': 'fdr'}
        ... })
        >>> # All properties are now accessible
        >>> results.mi_values
        array([0.1, 0.5, 0.3])
        >>> results.significant
        array([False,  True,  True])
        
        See Also
        --------
        ~driada.intense.intense_base.update : Add single property        """
        for dname, data in datadict.items():
            setattr(self, dname, data)

    def save_to_hdf5(self, fname):
        """Save all results to an HDF5 file.
        
        Exports all stored properties to an HDF5 file for persistent storage
        and later analysis. Handles numpy arrays, lists, and basic Python types.
        
        Parameters
        ----------
        fname : str
            Path to the output HDF5 file. Will be created or overwritten.
            
        Notes
        -----
        - Arrays are stored as HDF5 datasets
        - Lists are converted to numpy arrays before storage
        - Other types are stored as HDF5 attributes
        - Complex objects may need custom serialization
        - Nested dictionaries are preserved as HDF5 groups
        
        Examples
        --------
        >>> # Demonstrate saving analysis results to HDF5
        >>> import numpy as np
        >>> import tempfile
        >>> import os
        >>> results = IntenseResults()
        >>> # Add various types of data
        >>> results.update('mi_matrix', np.array([[0, 0.5], [0.5, 0]]))
        >>> results.update('settings', {'threshold': 0.05, 'method': 'mi'})
        >>> results.update('neuron_ids', [0, 1, 2])
        >>> # Create temporary file for demonstration
        >>> with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        ...     tmp_name = tmp.name
        >>> # Save all results to HDF5
        >>> results.save_to_hdf5(tmp_name)
        >>> # Verify file exists then clean up
        >>> os.path.exists(tmp_name)
        True
        >>> os.unlink(tmp_name)
        
        See Also
        --------
        ~driada.utils.data.write_dict_to_hdf5 : Underlying function used        """
        dict_repr = self.__dict__
        write_dict_to_hdf5(dict_repr, fname)


def compute_me_stats(
    ts_bunch1,
    ts_bunch2,
    names1=None,
    names2=None,
    mode="two_stage",
    metric="mi",
    mi_estimator="gcmi",
    precomputed_mask_stage1=None,
    precomputed_mask_stage2=None,
    n_shuffles_stage1=100,
    n_shuffles_stage2=10000,
    joint_distr=False,
    allow_mixed_dimensions=False,
    metric_distr_type="gamma",
    noise_ampl=1e-3,
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

    allow_mixed_dimensions : bool, default=False
        if True, both TimeSeries and MultiTimeSeries can be provided as signals.
        This parameter overrides "joint_distr"

    metric_distr_type : str, default="gamma"
        Distribution type for shuffled metric distribution fit. Supported options are distributions from scipy.stats
        Note: While 'gamma' is theoretically appropriate for MI distributions, empirical testing shows
        that 'norm' (normal distribution) often performs better due to its conservative p-values when
        fitting poorly to the skewed MI data. This conservatism reduces false positives.

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
    - Input masks are never modified; copies are created when needed.    """

    # FUTURE: add automatic min_shifts from autocorrelation time

    # Validate inputs
    validate_time_series_bunches(
        ts_bunch1, ts_bunch2, allow_mixed_dimensions=allow_mixed_dimensions
    )
    validate_metric(metric)
    validate_common_parameters(shift_window=shift_window, ds=ds, noise_const=noise_ampl)

    # Validate mode
    if mode not in ["stage1", "stage2", "two_stage"]:
        raise ValueError(
            f"mode must be 'stage1', 'stage2', or 'two_stage', got '{mode}'"
        )

    # Validate multicomp_correction
    if multicomp_correction not in [None, "bonferroni", "holm", "fdr_bh"]:
        raise ValueError(
            f"Unknown multiple comparison correction method: '{multicomp_correction}'"
        )

    # Validate pval_thr
    if not 0 < pval_thr < 1:
        raise ValueError(f"pval_thr must be between 0 and 1, got {pval_thr}")

    # Validate stage-specific parameters
    validate_common_parameters(nsh=n_shuffles_stage1)
    validate_common_parameters(nsh=n_shuffles_stage2)

    accumulated_info = dict()

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
    ts_with_delays = [ts for _, ts in enumerate(ts_bunch2) if _ not in skip_delays]
    ts_with_delays_inds = np.array(
        [_ for _, ts in enumerate(ts_bunch2) if _ not in skip_delays]
    )

    if find_optimal_delays:
        if enable_parallelization:
            optimal_delays_res = calculate_optimal_delays_parallel(
                ts_bunch1,
                ts_with_delays,
                metric,
                shift_window,
                ds,
                verbose=verbose,
                n_jobs=n_jobs,
            )
        else:
            optimal_delays_res = calculate_optimal_delays(
                ts_bunch1, ts_with_delays, metric, shift_window, ds, verbose=verbose
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

    if mode in ["two_stage", "stage1"]:
        npairs_to_check1 = int(np.sum(precomputed_mask_stage1))
        if verbose:
            print(
                f"Starting stage 1 scanning for {npairs_to_check1}/{nhyp} possible pairs"
            )

        # STAGE 1 - primary scanning
        random_shifts1, me_total1 = scan_pairs_router(
            ts_bunch1,
            ts_bunch2,
            metric,
            n_shuffles_stage1,
            optimal_delays,
            mi_estimator,
            joint_distr=joint_distr,
            allow_mixed_dimensions=allow_mixed_dimensions,
            ds=ds,
            mask=precomputed_mask_stage1,
            noise_const=noise_ampl,
            seed=seed,
            enable_parallelization=enable_parallelization,
            n_jobs=n_jobs,
        )

        # turn computed data tables from stage 1 and precomputed data into dict of stats dicts
        stage_1_stats = get_table_of_stats(
            me_total1,
            optimal_delays,
            metric_distr_type=metric_distr_type,
            nsh=n_shuffles_stage1,
            precomputed_mask=precomputed_mask_stage1,
            stage=1,
        )

        stage_1_stats_per_quantity = nested_dict_to_seq_of_tables(
            stage_1_stats, ordered_names1=range(n1), ordered_names2=range(n2)
        )
        # print(stage_1_stats_per_quantity)

        # select potentially significant pairs for stage 2
        # 0 in mask values means the pair MI is definitely insignificant, stage 2 calculation will be skipped.
        # 1 in mask values means the pair MI is potentially significant, stage 2 calculation will proceed.

        if verbose:
            print("Computing significance for all pairs in stage 1...")

        stage_1_significance = populate_nested_dict(dict(), range(n1), range(n2))
        for i in range(n1):
            for j in range(n2):
                pair_passes_stage1 = criterion1(
                    stage_1_stats[i][j], n_shuffles_stage1, topk=topk1
                )
                if pair_passes_stage1:
                    mask_from_stage1[i, j] = 1

                sig1 = {"stage1": pair_passes_stage1}
                stage_1_significance[i][j].update(sig1)

        stage_1_significance_per_quantity = nested_dict_to_seq_of_tables(
            stage_1_significance, ordered_names1=range(n1), ordered_names2=range(n2)
        )

        # print(stage_1_significance_per_quantity)
        accumulated_info.update(
            {
                "stage_1_significance": stage_1_significance_per_quantity,
                "stage_1_stats": stage_1_stats_per_quantity,
                "random_shifts1": random_shifts1,
                "me_total1": me_total1,
            }
        )

        nhyp = int(
            np.sum(mask_from_stage1)
        )  # number of hypotheses for further statistical testing
        if verbose:
            print("Stage 1 results:")
            print(
                f"{nhyp/n1/n2*100:.2f}% ({nhyp}/{n1*n2}) of possible pairs identified as candidates"
            )

    if mode == "stage1" or nhyp == 0:
        final_stats = add_names_to_nested_dict(stage_1_stats, names1, names2)
        final_significance = add_names_to_nested_dict(
            stage_1_significance, names1, names2
        )

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
            print(
                f"Starting stage 2 scanning for {npairs_to_check2}/{nhyp} possible pairs"
            )

        random_shifts2, me_total2 = scan_pairs_router(
            ts_bunch1,
            ts_bunch2,
            metric,
            n_shuffles_stage2,
            optimal_delays,
            mi_estimator,
            joint_distr=joint_distr,
            allow_mixed_dimensions=allow_mixed_dimensions,
            ds=ds,
            mask=combined_mask_for_stage_2,
            noise_const=noise_ampl,
            seed=seed,
            enable_parallelization=enable_parallelization,
            n_jobs=n_jobs,
        )

        # turn data tables from stage 2 to array of stats dicts
        stage_2_stats = get_table_of_stats(
            me_total2,
            optimal_delays,
            metric_distr_type=metric_distr_type,
            nsh=n_shuffles_stage2,
            precomputed_mask=combined_mask_for_stage_2,
            stage=2,
        )

        stage_2_stats_per_quantity = nested_dict_to_seq_of_tables(
            stage_2_stats, ordered_names1=range(n1), ordered_names2=range(n2)
        )
        # print(stage_2_stats_per_quantity)

        # select significant pairs after stage 2
        if verbose:
            print("Computing significance for all pairs in stage 2...")
        all_pvals = None
        if multicomp_correction in [
            "holm",
            "fdr_bh",
        ]:  # these procedures require all p-values
            all_pvals = get_all_nonempty_pvals(stage_2_stats, range(n1), range(n2))

        multicorr_thr = get_multicomp_correction_thr(
            pval_thr, mode=multicomp_correction, all_pvals=all_pvals, nhyp=nhyp
        )

        stage_2_significance = populate_nested_dict(dict(), range(n1), range(n2))
        for i in range(n1):
            for j in range(n2):
                pair_passes_stage2 = criterion2(
                    stage_2_stats[i][j], n_shuffles_stage2, multicorr_thr, topk=topk2
                )
                if pair_passes_stage2:
                    mask_from_stage2[i, j] = 1

                sig2 = {"stage2": pair_passes_stage2}
                stage_2_significance[i][j] = sig2

        stage_2_significance_per_quantity = nested_dict_to_seq_of_tables(
            stage_2_significance, ordered_names1=range(n1), ordered_names2=range(n2)
        )

        # print(stage_2_significance_per_quantity)
        accumulated_info.update(
            {
                "stage_2_significance": stage_2_significance_per_quantity,
                "stage_2_stats": stage_2_stats_per_quantity,
                "random_shifts2": random_shifts2,
                "me_total2": me_total2,
                "corrected_pval_thr": multicorr_thr,
                "group_pval_thr": pval_thr,
            }
        )

        num2 = int(np.sum(mask_from_stage2))
        if verbose:
            print("Stage 2 results:")
            print(
                f"{num2/n1/n2*100:.2f}% ({num2}/{n1*n2}) of possible pairs identified as significant"
            )

        # Always merge stats for consistency
        merged_stats = merge_stage_stats(stage_1_stats, stage_2_stats)
        merged_significance = merge_stage_significance(
            stage_1_significance, stage_2_significance
        )
        final_stats = add_names_to_nested_dict(merged_stats, names1, names2)
        final_significance = add_names_to_nested_dict(
            merged_significance, names1, names2
        )
        return final_stats, final_significance, accumulated_info


def get_multicomp_correction_thr(fwer, mode="holm", **multicomp_kwargs):
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
            raise ValueError(
                "Number of hypotheses for Bonferroni correction not provided"
            )
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
