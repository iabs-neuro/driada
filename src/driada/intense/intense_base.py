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
    ts_bunch1 : list
        First set of time series.
    ts_bunch2 : list
        Second set of time series.
    allow_mixed_dimensions : bool, optional
        Whether to allow mixed TimeSeries and MultiTimeSeries. Default: False.

    Raises
    ------
    ValueError
        If validation fails.
    """
    if len(ts_bunch1) == 0:
        raise ValueError("ts_bunch1 cannot be empty")
    if len(ts_bunch2) == 0:
        raise ValueError("ts_bunch2 cannot be empty")

    # Check time series types
    if not allow_mixed_dimensions:
        ts1_types = [type(ts) for ts in ts_bunch1]
        ts2_types = [type(ts) for ts in ts_bunch2]

        if not all(t == TimeSeries for t in ts1_types):
            if any(t == MultiTimeSeries for t in ts1_types):
                raise ValueError(
                    "MultiTimeSeries found in ts_bunch1 but allow_mixed_dimensions=False"
                )
            else:
                raise ValueError("ts_bunch1 must contain TimeSeries objects")

        if not all(t == TimeSeries for t in ts2_types):
            if any(t == MultiTimeSeries for t in ts2_types):
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
        Metric name to validate.
    allow_scipy : bool, optional
        Whether to allow scipy correlation metrics. Default: True.

    Returns
    -------
    metric_type : str
        Type of metric: 'mi', 'correlation', 'special', or 'scipy'.

    Raises
    ------
    ValueError
        If metric is not supported.
    """
    # Built-in metrics
    if metric == "mi":
        return "mi"

    # Special metrics
    if metric in ["av", "fast_pearsonr"]:
        return "special"

    # Common correlation metrics (shorthand names)
    correlation_metrics = ["spearman", "pearson", "kendall"]
    if metric in correlation_metrics:
        return "correlation"

    # Full scipy names
    scipy_correlation_metrics = ["spearmanr", "pearsonr", "kendalltau"]
    if metric in scipy_correlation_metrics:
        return "scipy"

    # Check if it's a scipy function
    if allow_scipy:
        try:
            import scipy.stats

            if hasattr(scipy.stats, metric):
                return "scipy"
        except ImportError:
            pass

    # If we get here, metric is not supported
    raise ValueError(
        f"Unsupported metric: {metric}. Supported metrics include: "
        f"'mi', 'av', 'fast_pearsonr', 'spearman', 'pearson', 'kendall', "
        f"'spearmanr', 'pearsonr', 'kendalltau', and other scipy.stats functions."
    )


def validate_common_parameters(shift_window=None, ds=None, nsh=None, noise_const=None):
    """
    Validate common INTENSE parameters.

    Parameters
    ----------
    shift_window : int, optional
        Maximum shift window in frames.
    ds : int, optional
        Downsampling factor.
    nsh : int, optional
        Number of shuffles.
    noise_const : float, optional
        Noise constant for numerical stability.

    Raises
    ------
    ValueError
        If any parameter is invalid.
    """
    if shift_window is not None and shift_window < 0:
        raise ValueError(f"shift_window must be non-negative, got {shift_window}")

    if ds is not None and ds <= 0:
        raise ValueError(f"ds must be positive, got {ds}")

    if nsh is not None and nsh <= 0:
        raise ValueError(f"nsh must be positive, got {nsh}")

    if noise_const is not None and noise_const < 0:
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
        Similarity metric to maximize. Options include:
        - 'mi': Mutual information
        - 'spearman': Spearman correlation
        - Other metrics supported by get_sim function
    shift_window : int
        Maximum shift to test in each direction (frames).
        Will test shifts from -shift_window to +shift_window.
    ds : int
        Downsampling factor. Every ds-th point is used from the time series.
        Default: 1 (no downsampling).
    verbose : bool, optional
        Whether to print progress information. Default: True.
    enable_progressbar : bool, optional
        Whether to show progress bar. Default: True.

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
    >>> neurons = [neuron1.ca, neuron2.ca]  # calcium signals
    >>> behaviors = [speed_ts, direction_ts]  # behavioral variables
    >>> delays = calculate_optimal_delays(neurons, behaviors, 'mi',
    ...                                   shift_window=100, ds=1)
    >>> print(f"Neuron 1 optimal delay with speed: {delays[0, 0]} frames")
    """
    # Validate inputs
    validate_time_series_bunches(ts_bunch1, ts_bunch2, allow_mixed_dimensions=False)
    validate_metric(metric)
    validate_common_parameters(shift_window=shift_window, ds=ds)

    if verbose:
        print("Calculating optimal delays:")

    optimal_delays = np.zeros((len(ts_bunch1), len(ts_bunch2)), dtype=int)
    shifts = np.arange(-shift_window, shift_window, ds) // ds

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
        Similarity metric to maximize. Options include:
        - 'mi': Mutual information
        - 'spearman': Spearman correlation
        - Other metrics supported by get_sim function
    shift_window : int
        Maximum shift to test in each direction (frames).
        Will test shifts from -shift_window to +shift_window.
    ds : int
        Downsampling factor. Every ds-th point is used from the time series.
    verbose : bool, optional
        Whether to print progress information. Default: True.
    n_jobs : int, optional
        Number of parallel jobs to run. Default: -1 (use all available cores).

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
    calculate_optimal_delays : Sequential version of this function

    Examples
    --------
    >>> neurons = [neuron.ca for neuron in exp.neurons[:100]]
    >>> behaviors = [exp.speed, exp.direction]
    >>> # Use 8 cores for faster computation
    >>> delays = calculate_optimal_delays_parallel(neurons, behaviors, 'mi',
    ...                                            shift_window=100, ds=1, n_jobs=8)
    """
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
    >>> # Backward compatibility - single cell and feature
    >>> mi_zero, mi_profile = get_calcium_feature_me_profile(exp, 0, 'speed')
    >>>
    >>> # New usage - analyze multiple cells and features
    >>> results = get_calcium_feature_me_profile(exp, cbunch=[0, 1, 2], fbunch=['speed', 'head_direction'])
    >>> # Access specific result: results[cell_id][feat_id]['me0'] and ['shifted_me']
    >>>
    >>> # Analyze all cells with all features
    >>> results = get_calcium_feature_me_profile(exp, cbunch=None, fbunch=None)
    >>>
    >>> # Multi-feature joint mutual information
    >>> results = get_calcium_feature_me_profile(exp, cbunch=[0], fbunch=[('x', 'y')])
    """
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
                me0 = get_sim(ts1, ts2, metric, ds=ds)

                for shift in np.arange(-window, window, ds) // ds:
                    lag_me = get_sim(ts1, ts2, metric, ds=ds, shift=shift)
                    shifted_me.append(lag_me)

            else:
                # Multi-feature (tuple)
                if metric != "mi":
                    raise ValueError(
                        f"Multi-feature analysis only supported for metric='mi', got '{metric}'"
                    )
                feats = [exp.dynamic_features[f] for f in fid]
                me0 = get_multi_mi(feats, ts1, ds=ds)

                for shift in np.arange(-window, window, ds) // ds:
                    lag_me = get_multi_mi(feats, ts1, ds=ds, shift=shift)
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
    joint_distr=False,
    ds=1,
    mask=None,
    noise_const=1e-3,
    seed=None,
    allow_mixed_dimensions=False,
    enable_progressbar=True,
):
    """
    Calculates MI shuffles for 2 given sets of TimeSeries
    This function is generally assumed to be used internally,
    but can be also called manually to "look inside" high-level computation routines

    Parameters
    ----------
    ts_bunch1: list of TimeSeries objects

    ts_bunch2: list of TimeSeries objects

    metric: similarity metric between TimeSeries

    nsh: int
        number of shuffles

    joint_distr: bool
        if joint_distr=True, ALL (sic!) TimeSeries in ts_bunch2 will be treated as components of a single multifeature
        default: False

    ds: int
        Downsampling constant. Every "ds" point will be taken from the data time series.
        default: 1

    mask: np.array of shape (len(ts_bunch1), len(ts_bunch2)) or (len(ts_bunch), 1) if joint_distr=True
          precomputed mask for skipping some of possible pairs.
          0 in mask values means calculation will be skipped.
          1 in mask values means calculation will proceed.

    noise_const: float
        Small noise amplitude, which is added to MI and shuffled MI to improve numerical fit
        default: 1e-3

    optimal_delays: np.array of shape (len(ts_bunch1), len(ts_bunch2)) or (len(ts_bunch), 1) if joint_distr=True
        best shifts from original time series alignment in terms of MI.

    seed: int
        Random seed for reproducibility

    Returns
    -------
    random_shifts: np.array of shape (len(ts_bunch1), len(ts_bunch2), nsh)
        signals shifts used for MI distribution computation

    me_total: np.array of shape (len(ts_bunch1), len(ts_bunch2)), nsh+1) or (len(ts_bunch1), 1, nsh+1) if joint_distr==True
        Aggregated array of true and shuffled MI values.
        True MI matrix can be obtained by me_total[:,:,0]
        Shuffled MI tensor of shape (len(ts_bunch1), len(ts_bunch2)), nsh) or (len(ts_bunch1), 1, nsh) if joint_distr==True
        can be obtained by me_total[:,:,1:]
    """

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
        if joint_distr:
            np.random.seed(seed)
            # Combine shuffle masks from ts1 and all ts in tsbunch2
            combined_shuffle_mask = ts1.shuffle_mask.copy()
            for ts2 in ts_bunch2:
                combined_shuffle_mask = combined_shuffle_mask & ts2.shuffle_mask
            # move shuffle mask according to optimal shift
            combined_shuffle_mask = np.roll(
                combined_shuffle_mask, int(optimal_delays[i, 0])
            )
            indices_to_select = np.arange(t)[combined_shuffle_mask]
            random_shifts[i, 0, :] = np.random.choice(indices_to_select, size=nsh) // ds

        else:
            for j, ts2 in enumerate(ts_bunch2):
                np.random.seed(seed)
                combined_shuffle_mask = ts1.shuffle_mask & ts2.shuffle_mask
                # move shuffle mask according to optimal shift
                combined_shuffle_mask = np.roll(
                    combined_shuffle_mask, int(optimal_delays[i, j])
                )
                indices_to_select = np.arange(t)[combined_shuffle_mask]
                random_shifts[i, j, :] = (
                    np.random.choice(indices_to_select, size=nsh) // ds
                )

    # calculate similarity metric arrays
    for i, ts1 in tqdm.tqdm(
        enumerate(ts_bunch1),
        total=len(ts_bunch1),
        position=0,
        leave=True,
        disable=not enable_progressbar,
    ):

        np.random.seed(seed)

        # DEPRECATED: This joint_distr branch is deprecated and will be removed in v2.0
        # Use MultiTimeSeries for joint distribution handling instead
        # TODO: Remove this entire branch in v2.0
        if joint_distr:
            if metric != "mi":
                raise ValueError("joint_distr mode works with metric = 'mi' only")
            if mask[i, 0] == 1:
                # default metric without shuffling, minus due to different order
                me0 = get_multi_mi(
                    ts_bunch2, ts1, ds=ds, shift=-optimal_delays[i, 0] // ds
                )
                me_table[i, 0] = (
                    me0 + np.random.random() * noise_const
                )  # add small noise for better fitting

                np.random.seed(seed)
                random_noise = (
                    np.random.random(size=len(random_shifts[i, 0, :])) * noise_const
                )  # add small noise for better fitting
                for k, shift in enumerate(random_shifts[i, 0, :]):
                    mi = get_multi_mi(ts_bunch2, ts1, ds=ds, shift=shift)
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
                        check_for_coincidence=True,
                    )  # default metric without shuffling

                    np.random.seed(seed)
                    me_table[i, j] = (
                        me0 + np.random.random() * noise_const
                    )  # add small noise for better fitting

                    np.random.seed(seed)
                    random_noise = (
                        np.random.random(size=len(random_shifts[i, j, :])) * noise_const
                    )  # add small noise for better fitting

                    for k, shift in enumerate(random_shifts[i, j, :]):
                        np.random.seed(seed)
                        # mi = get_1d_mi(ts1, ts2, shift=shift, ds=ds)
                        me = get_sim(ts1, ts2, metric, ds=ds, shift=shift)

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
    joint_distr : bool, optional
        If True, treats all ts_bunch2 as components of a single multifeature.
        Default: False.
    ds : int, optional
        Downsampling factor. Default: 1.
    mask : np.ndarray, optional
        Binary mask of shape (len(ts_bunch1), len(ts_bunch2)).
        0 = skip computation, 1 = compute. Default: all ones.
    noise_const : float, optional
        Small noise added to improve numerical stability. Default: 1e-3.
    seed : int, optional
        Random seed for reproducibility. Default: None.
    n_jobs : int, optional
        Number of parallel jobs. Default: -1 (use all cores).

    Returns
    -------
    random_shifts : np.ndarray of shape (len(ts_bunch1), len(ts_bunch2), nsh)
        Random shifts used for shuffling.
    me_total : np.ndarray of shape (len(ts_bunch1), len(ts_bunch2), nsh+1)
        Metric values. [:,:,0] contains true values, [:,:,1:] contains shuffles.

    See Also
    --------
    scan_pairs : Sequential version of this function
    scan_pairs_router : Wrapper that chooses between parallel and sequential
    """

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
            split_optimal_delays[_],
            joint_distr=joint_distr,
            allow_mixed_dimensions=allow_mixed_dimensions,
            ds=ds,
            mask=split_mask[_],
            noise_const=noise_const,
            seed=seed,
            enable_progressbar=False,
        )
        for _, small_ts_bunch in enumerate(split_ts_bunch1)
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
    joint_distr : bool, optional
        If True, treats all ts_bunch2 as components of a single multifeature.
        Default: False.
    ds : int, optional
        Downsampling factor. Default: 1.
    mask : np.ndarray, optional
        Binary mask of shape (len(ts_bunch1), len(ts_bunch2)).
        0 = skip computation, 1 = compute. Default: all ones.
    noise_const : float, optional
        Small noise added to improve numerical stability. Default: 1e-3.
    seed : int, optional
        Random seed for reproducibility. Default: None.
    enable_parallelization : bool, optional
        Whether to use parallel processing. Default: True.
    n_jobs : int, optional
        Number of parallel jobs if parallelization enabled. Default: -1 (use all cores).

    Returns
    -------
    random_shifts : np.ndarray of shape (len(ts_bunch1), len(ts_bunch2), nsh)
        Random shifts used for shuffling.
    me_total : np.ndarray of shape (len(ts_bunch1), len(ts_bunch2), nsh+1)
        Metric values. [:,:,0] contains true values, [:,:,1:] contains shuffles.

    See Also
    --------
    scan_pairs : Sequential implementation
    scan_pairs_parallel : Parallel implementation
    """

    if enable_parallelization:
        random_shifts, me_total = scan_pairs_parallel(
            ts_bunch1,
            ts_bunch2,
            metric,
            nsh,
            optimal_delays,
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
    >>> results = IntenseResults()
    >>> results.update('stats', computed_stats)
    >>> results.update('info', {'optimal_delays': delays})
    >>> results.save_to_hdf5('intense_results.h5')
    """

    def __init__(self):
        pass

    def update(self, property_name, data):
        """Add or update a property with data."""
        setattr(self, property_name, data)

    def update_multiple(self, datadict):
        """Update multiple properties from a dictionary."""
        for dname, data in datadict.items():
            setattr(self, dname, data)

    def save_to_hdf5(self, fname):
        """Save all results to an HDF5 file."""
        dict_repr = self.__dict__
        write_dict_to_hdf5(dict_repr, fname)


def compute_me_stats(
    ts_bunch1,
    ts_bunch2,
    names1=None,
    names2=None,
    mode="two_stage",
    metric="mi",
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
    ts_bunch1: list of TimeSeries objects

    ts_bunch2: list of TimeSeries objects

    names1: list of str
        names than will be given to time series from tsbunch1 in final results

    names2: list of str
        names than will be given to time series from tsbunch2 in final results

    mode: str
        Computation mode. 3 modes are available:
        'stage1': perform preliminary scanning with "n_shuffles_stage1" shuffles only.
                  Rejects strictly non-significant neuron-feature pairs, does not give definite results
                  about significance of the others.
        'stage2': skip stage 1 and perform full-scale scanning ("n_shuffles_stage2" shuffles) of all neuron-feature pairs.
                  Gives definite results, but can be very time-consuming. Also reduces statistical power
                  of multiple comparison tests, since the number of hypotheses is very high.
        'two_stage': prune non-significant pairs during stage 1 and perform thorough testing for the rest during stage 2.
                     Recommended mode.
        default: 'two-stage'

    metric: similarity metric between TimeSeries
        default: 'mi'

    precomputed_mask_stage1: np.array of shape (len(ts_bunch1), len(ts_bunch2)) or (len(ts_bunch), 1) if joint_distr=True
          precomputed mask for skipping some of possible pairs in stage 1.
          0 in mask values means calculation will be skipped.
          1 in mask values means calculation will proceed.

    precomputed_mask_stage2: np.array of shape (len(ts_bunch1), len(ts_bunch2)) or (len(ts_bunch), 1) if joint_distr=True
          precomputed mask for skipping some of possible pairs in stage 2.
          0 in mask values means calculation will be skipped.
          1 in mask values means calculation will proceed.

    n_shuffles_stage1: int
        number of shuffles for first stage
        default: 100

    n_shuffles_stage2: int
        number of shuffles for second stage
        default: 10000

    joint_distr: bool
        if joint_distr=True, ALL features in feat_bunch will be treated as components of a single multifeature
        For example, 'x' and 'y' features will be put together into ('x','y') multifeature.
        default: False

    allow_mixed_dimensions: bool
        if True, both TimeSeries and MultiTimeSeries can be provided as signals.
        This parameter overrides "joint_distr"
        default: False

    metric_distr_type: str
        Distribution type for shuffled metric distribution fit. Supported options are distributions from scipy.stats
        Note: While 'gamma' is theoretically appropriate for MI distributions, empirical testing shows
        that 'norm' (normal distribution) often performs better due to its conservative p-values when
        fitting poorly to the skewed MI data. This conservatism reduces false positives.
        default: "gamma"

    noise_ampl: float
        Small noise amplitude, which is added to metrics to improve numerical fit
        default: 1e-3

    ds: int
        Downsampling constant. Every "ds" point will be taken from the data time series.
        default: 1

    topk1: int
        true MI for stage 1 should be among topk1 MI shuffles
        default: 1

    topk2: int
        true MI for stage 2 should be among topk2 MI shuffles
        default: 5

    multicomp_correction: str or None
        type of multiple comparisons correction. Supported types are None (no correction),
        "bonferroni", "holm", and "fdr_bh".
        default: 'holm'

    pval_thr: float
        pvalue threshold. if multicomp_correction=None, this is a p-value for a single pair.
        For FWER methods (bonferroni, holm), this is the family-wise error rate.
        For FDR methods (fdr_bh), this is the false discovery rate.

    find_optimal_delays: bool
        Allows slight shifting (not more than +- shift_window) of time series,
        selects a shift with the highest MI as default.
        default: True

    skip_delays: list
        List of indices from ts_bunch2 for which delays are not applied (set to 0).
        Has no effect if find_optimal_delays = False

    shift_window: int
        Window for optimal shift search (frames). Optimal shift will lie in the range
        -shift_window <= opt_shift <= shift_window

    verbose: bool
        whether to print intermediate information

    seed: int
        random seed for reproducibility

    duplicate_behavior: str
        How to handle duplicate TimeSeries in ts_bunch1 or ts_bunch2.
        - 'ignore': Process duplicates normally (default)
        - 'raise': Raise an error if duplicates are found
        - 'warn': Print a warning but continue processing

    Returns
    -------
    stats: dict of dict of dicts
        Outer dict keys: indices of tsbunch1 or names1, if given
        Inner dict keys: indices or tsbunch2 or names2, if given
        Last dict: dictionary of stats variables.
        Can be easily converted to pandas DataFrame by pd.DataFrame(stats)

    significance: dict of dict of dicts
        Outer dict keys: indices of tsbunch1 or names1, if given
        Inner dict keys: indices or tsbunch2 or names2, if given
        Last dict: dictionary of significance-related variables.
        Can be easily converted to pandas DataFrame by pd.DataFrame(significance)

    accumulated_info: dict
        Data collected during computation.
    """

    # TODO: add automatic min_shifts from autocorrelation time

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
    if precomputed_mask_stage2 is None:
        precomputed_mask_stage2 = np.ones((n1, n2))

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
    mode : str or None, optional
        Multiple comparison correction method. Default: 'holm'.
        - None: No correction, threshold = fwer
        - 'bonferroni': Bonferroni correction (FWER control)
        - 'holm': Holm-Bonferroni correction (FWER control, more powerful)
        - 'fdr_bh': Benjamini-Hochberg FDR correction
    **multicomp_kwargs : dict
        Additional arguments for correction method:
        - For 'bonferroni': nhyp (int) - number of hypotheses
        - For 'holm': all_pvals (list) - all p-values to be tested
        - For 'fdr_bh': all_pvals (list) - all p-values to be tested

    Returns
    -------
    threshold : float
        Adjusted p-value threshold for individual hypothesis testing.

    Raises
    ------
    ValueError
        If required arguments are missing or unknown method specified.

    Notes
    -----
    - FWER methods (bonferroni, holm) control probability of ANY false positive
    - FDR methods control expected proportion of false positives among rejections
    - Holm is uniformly more powerful than Bonferroni
    - FDR typically allows more discoveries but with controlled false positive rate

    Examples
    --------
    >>> # Holm correction (default)
    >>> pvals = [0.001, 0.01, 0.02, 0.03, 0.04]
    >>> thr = get_multicomp_correction_thr(0.05, mode='holm', all_pvals=pvals)
    >>>
    >>> # FDR correction
    >>> thr = get_multicomp_correction_thr(0.05, mode='fdr_bh', all_pvals=pvals)
    """
    if mode is None:
        threshold = fwer

    elif mode == "bonferroni":
        if "nhyp" in multicomp_kwargs:
            threshold = fwer / multicomp_kwargs["nhyp"]
        else:
            raise ValueError(
                "Number of hypotheses for Bonferroni correction not provided"
            )

    elif mode == "holm":
        if "all_pvals" in multicomp_kwargs:
            all_pvals = sorted(multicomp_kwargs["all_pvals"])
            nhyp = len(all_pvals)
            threshold = 0  # Default if no discoveries
            for i, pval in enumerate(all_pvals):
                cthr = fwer / (nhyp - i)
                if pval > cthr:
                    break
                threshold = cthr
        else:
            raise ValueError("List of p-values for Holm correction not provided")

    elif mode == "fdr_bh":
        if "all_pvals" in multicomp_kwargs:
            all_pvals = sorted(multicomp_kwargs["all_pvals"])
            nhyp = len(all_pvals)
            threshold = 0.0

            # Benjamini-Hochberg procedure
            for i in range(nhyp - 1, -1, -1):
                if all_pvals[i] <= fwer * (i + 1) / nhyp:
                    threshold = all_pvals[i]
                    break
        else:
            raise ValueError("List of p-values for FDR correction not provided")

    else:
        raise ValueError("Unknown multiple comparisons correction method")

    return threshold
