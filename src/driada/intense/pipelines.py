import numpy as np

from .stats import stats_not_empty
from .intense_base import compute_me_stats, IntenseResults
from ..information.info_base import TimeSeries, MultiTimeSeries
from .disentanglement import disentangle_all_selectivities, DEFAULT_MULTIFEATURE_MAP
from ..experiment.exp_base import DEFAULT_STATS


def compute_cell_feat_significance(
    exp,
    cell_bunch=None,
    feat_bunch=None,
    data_type="calcium",
    metric="mi",
    mi_estimator="gcmi",
    mode="two_stage",
    n_shuffles_stage1=100,
    n_shuffles_stage2=10000,
    joint_distr=False,
    allow_mixed_dimensions=False,
    metric_distr_type="gamma",
    noise_ampl=1e-3,
    ds=1,
    use_precomputed_stats=True,
    save_computed_stats=True,
    force_update=False,
    topk1=1,
    topk2=5,
    multicomp_correction="holm",
    pval_thr=0.01,
    find_optimal_delays=True,
    skip_delays=[],
    shift_window=5,
    verbose=True,
    enable_parallelization=True,
    n_jobs=-1,
    seed=42,
    with_disentanglement=False,
    multifeature_map=None,
    duplicate_behavior="ignore",
):
    """
    Calculates significant neuron-feature pairs

    Parameters
    ----------
    exp : Experiment
        Experiment object to read and write data from
    cell_bunch : int, iterable or None, optional
        Neuron indices. By default, (cell_bunch=None), all neurons will be taken
    feat_bunch : str, iterable or None, optional
        Feature names. By default, (feat_bunch=None), all single features will be taken
    data_type : str, optional
        Data type used for INTENSE computations. Can be 'calcium' or 'spikes'. Default is 'calcium'
    metric : str, optional
        Similarity metric between TimeSeries. Default is 'mi'

    mi_estimator : str, optional
        Mutual information estimator to use when metric='mi'. Options: 'gcmi' or 'ksg'.
        Default is 'gcmi'
    mode : str, optional
        Computation mode. 3 modes are available:
        'stage1': perform preliminary scanning with "n_shuffles_stage1" shuffles only.
                  Rejects strictly non-significant neuron-feature pairs, does not give definite results
                  about significance of the others.
        'stage2': skip stage 1 and perform full-scale scanning ("n_shuffles_stage2" shuffles) of all neuron-feature pairs.
                  Gives definite results, but can be very time-consuming. Also reduces statistical power
                  of multiple comparison tests, since the number of hypotheses is very high.
        'two_stage': prune non-significant pairs during stage 1 and perform thorough testing for the rest during stage 2.
                     Recommended mode.
        Default is 'two_stage'

    n_shuffles_stage1 : int, optional
        Number of shuffles for first stage. Default is 100
    n_shuffles_stage2 : int, optional
        Number of shuffles for second stage. Default is 10000
    joint_distr : bool, optional
        If True, ALL features in feat_bunch will be treated as components of a single multifeature.
        For example, 'x' and 'y' features will be put together into ('x','y') multifeature.
        Note: This parameter is marked for deprecation. Use allow_mixed_dimensions instead.
        Default is False
    allow_mixed_dimensions : bool, optional
        If True, both TimeSeries and MultiTimeSeries can be provided as signals.
        This parameter overrides "joint_distr". Default is False

    metric_distr_type : str, optional
        Distribution type for shuffled metric distribution fit. Supported options are distributions from scipy.stats.
        Note: While 'gamma' is theoretically appropriate for MI distributions, empirical testing shows
        that 'norm' (normal distribution) often performs better due to its conservative p-values when
        fitting poorly to the skewed MI data. This conservatism reduces false positives.
        Default is "gamma"
    noise_ampl : float, optional
        Small noise amplitude, which is added to MI and shuffled MI to improve numerical fit.
        Default is 1e-3
    ds : int, optional
        Downsampling constant. Every "ds" point will be taken from the data time series.
        Reduces the computational load, but needs caution since with large "ds" some important information may be lost.
        Experiment class performs an internal check for this effect.
        Default is 1

    use_precomputed_stats : bool, optional
        Whether to use stats saved in Experiment instance. Stats are accumulated separately for stage1 and stage2.
        Notes on stats data rewriting (if save_computed_stats=True):
        If you want to recalculate stage1 results only, use "use_precomputed_stats=False" and "mode='stage1'".
        Stage 2 stats will be erased since they will become irrelevant.
        If you want to recalculate stage2 results only, use "use_precomputed_stats=True" and "mode='stage2'" or "mode='two-stage'"
        If you want to recalculate everything, use "use_precomputed_stats=False" and "mode='two-stage'".
        Default is True
    save_computed_stats : bool, optional
        Whether to save computed stats to Experiment instance. Default is True
    force_update : bool, optional
        Whether to force saved statistics data update in case the collision between actual data hashes and
        saved stats data hashes is found (for example, if neuronal or behavior data has been changed externally).
        Default is False

    topk1 : int, optional
        True MI for stage 1 should be among topk1 MI shuffles. Default is 1
    topk2 : int, optional
        True MI for stage 2 should be among topk2 MI shuffles. Default is 5
    multicomp_correction : str or None, optional
        Type of multiple comparison correction. Supported types are None (no correction),
        "bonferroni" and "holm". Default is 'holm'
    pval_thr : float, optional
        P-value threshold. If multicomp_correction=None, this is a p-value for a single pair.
        Otherwise it is a FWER significance level. Default is 0.01

    find_optimal_delays : bool, optional
        Allows slight shifting (not more than +- shift_window) of time series,
        selects a shift with the highest MI as default. Default is True
    skip_delays : list, optional
        List of features for which delays are not applied (set to 0).
        Only features that exist in feat_bunch will be processed.
        Has no effect if find_optimal_delays = False. Default is []
    shift_window : int, optional
        Window for optimal shift search (seconds). Optimal shift (in frames) will lie in the range
        -shift_window*fps <= opt_shift <= shift_window*fps.
        Has no effect if find_optimal_delays = False. Default is 5

    verbose : bool, optional
        Whether to print progress messages. Default is True
    enable_parallelization : bool, optional
        Whether to enable parallel processing. Default is True
    n_jobs : int, optional
        Number of parallel jobs. -1 means use all processors. Default is -1
    seed : int, optional
        Random seed for reproducibility. Default is 42
    with_disentanglement : bool, optional
        If True, performs a full INTENSE pipeline with mixed selectivity analysis:
        1. Computes behavioral feature-feature significance
        2. Computes neuron-feature significance
        3. Disentangles mixed selectivities using behavioral correlations.
        Default is False
    multifeature_map : dict or None, optional
        Mapping from multifeature tuples to aggregated names for disentanglement.
        If None, uses DEFAULT_MULTIFEATURE_MAP from disentanglement module.
        Only used when with_disentanglement=True. Default is None
    duplicate_behavior : str, optional
        How to handle duplicate TimeSeries in neuron or feature bunches.
        - 'ignore': Process duplicates normally (default)
        - 'raise': Raise an error if duplicates are found
        - 'warn': Print a warning but continue processing.
        Default is 'ignore'

    Returns
    -------
    stats: dict of dict of dicts
        Outer dict: dynamic features, inner dict: cells, last dict: stats.
        Can be easily converted to pandas DataFrame by pd.DataFrame(stats)
    significance: dict of dict of bools
        Significance results for each neuron-feature pair
    info: dict
        Additional information from compute_me_stats
    intense_res: IntenseResults
        Complete results object
    disentanglement_results: dict (only if with_disentanglement=True)
        Contains:
        - 'feat_feat_significance': Feature-feature significance matrix
        - 'disent_matrix': Disentanglement results matrix
        - 'count_matrix': Count matrix from disentanglement
        - 'summary': Summary statistics from disentanglement
    
    Raises
    ------
    ValueError
        If data_type is not 'calcium' or 'spikes'
        If features are not found in experiment
        If allow_mixed_dimensions enabled with unknown feature type
    
    Notes
    -----
    - When joint_distr=True, all features are combined into a single multifeature
    - shift_window is converted from seconds to frames using exp.fps
    - Updates exp.optimal_nf_delays as a side effect
    - Relative MI values are computed using appropriate neural data entropy
    - When with_disentanglement=True, feat-feat uses 1/10 the shuffles of neuron-feat
    
    Examples
    --------
    >>> from driada.experiment.synthetic import generate_synthetic_exp
    >>> import numpy as np
    >>> 
    >>> # Create small test experiment
    >>> exp = generate_synthetic_exp(n_dfeats=2, n_cfeats=1, nneurons=3, 
    ...                              duration=60, fps=10, seed=42, verbose=False)
    >>> 
    >>> # Basic neuron-feature analysis (stage1 for speed)
    >>> stats, sig, info, res = compute_cell_feat_significance(
    ...     exp, 
    ...     cell_bunch=[0, 1],
    ...     feat_bunch=['d_feat_0'],
    ...     mode='stage1',
    ...     n_shuffles_stage1=10,
    ...     verbose=False
    ... )  # doctest: +ELLIPSIS
    ...
    >>> len(stats)  # Number of neurons analyzed
    2
    >>> 'd_feat_0' in stats[0]  # Feature present in results
    True
    >>> 
    >>> # With disentanglement analysis
    >>> result = compute_cell_feat_significance(
    ...     exp,
    ...     cell_bunch=[0, 1],
    ...     mode='stage1',
    ...     n_shuffles_stage1=10,
    ...     with_disentanglement=True,
    ...     verbose=False
    ... )  # doctest: +ELLIPSIS
    ...
    >>> len(result)  # Returns 5 values with disentanglement
    5
    >>> stats, sig, info, res, disent = result
    >>> 'disent_matrix' in disent
    True    """

    exp.check_ds(ds)

    cell_ids = exp._process_cbunch(cell_bunch)
    feat_ids = exp._process_fbunch(feat_bunch, allow_multifeatures=True, mode=data_type)
    cells = [exp.neurons[cell_id] for cell_id in cell_ids]

    if data_type == "calcium":
        signals = [cell.ca for cell in cells]
    elif data_type == "spikes":
        signals = [cell.sp for cell in cells]
    else:
        raise ValueError('"data_type" can be either "calcium" or "spikes"')

    # min_shifts = [int(cell.get_t_off() * MIN_CA_SHIFT) for cell in cells]
    if not allow_mixed_dimensions:
        feats = [
            exp.dynamic_features[feat_id]
            for feat_id in feat_ids
            if feat_id in exp.dynamic_features
        ]
        if joint_distr:
            feat_ids = [tuple(sorted(feat_ids))]
    else:
        feats = []
        for feat_id in feat_ids:
            if isinstance(feat_id, str):
                if feat_id not in exp.dynamic_features:
                    raise ValueError(
                        f"Feature '{feat_id}' not found in experiment. Available features: {list(exp.dynamic_features.keys())}"
                    )
                ts = exp.dynamic_features[feat_id]
                feats.append(ts)
            elif isinstance(feat_id, tuple):
                for f in feat_id:
                    if f not in exp.dynamic_features:
                        raise ValueError(
                            f"Feature '{f}' not found in experiment. Available features: {list(exp.dynamic_features.keys())}"
                        )
                parts = [exp.dynamic_features[f] for f in feat_id]
                mts = MultiTimeSeries(parts)
                feats.append(mts)
            else:
                raise ValueError("Unknown feature id type")

    n, t, f = len(cells), exp.n_frames, len(feats)

    precomputed_mask_stage1 = np.ones((n, f))
    precomputed_mask_stage2 = np.ones((n, f))

    if not exp.selectivity_tables_initialized:
        exp._set_selectivity_tables(data_type, cbunch=cell_ids, fbunch=feat_ids)

    if use_precomputed_stats:
        if verbose:
            print("Retrieving saved stats data...")
        # 0 in mask values means precomputed results are found, calculation will be skipped.
        # 1 in mask values means precomputed results are not found or incomplete, calculation will proceed.

        for i, cell_id in enumerate(cell_ids):
            for j, feat_id in enumerate(feat_ids):
                try:
                    pair_stats = exp.get_neuron_feature_pair_stats(
                        cell_id, feat_id, mode=data_type
                    )
                except (ValueError, KeyError):
                    if isinstance(feat_id, str):
                        raise ValueError(
                            f"Unknown single feature in feat_bunch: {feat_id}. Check initial data"
                        )
                    else:
                        exp._add_multifeature_to_data_hashes(feat_id, mode=data_type)
                        exp._add_multifeature_to_stats(feat_id, mode=data_type)
                        pair_stats = DEFAULT_STATS.copy()

                current_data_hash = exp._data_hashes[data_type][feat_id][cell_id]

                if stats_not_empty(pair_stats, current_data_hash, stage=1):
                    precomputed_mask_stage1[i, j] = 0
                if stats_not_empty(pair_stats, current_data_hash, stage=2):
                    precomputed_mask_stage2[i, j] = 0

    combined_precomputed_mask = np.ones((n, f))
    if mode in ["stage2", "two_stage"]:
        combined_precomputed_mask[
            np.where((precomputed_mask_stage1 == 0) & (precomputed_mask_stage2 == 0))
        ] = 0
    elif mode == "stage1":
        combined_precomputed_mask[np.where(precomputed_mask_stage1 == 0)] = 0
    else:
        raise ValueError("Wrong mode!")

    computed_stats, computed_significance, info = compute_me_stats(
        signals,
        feats,
        mode=mode,
        names1=cell_ids,
        names2=feat_ids,
        metric=metric,
        mi_estimator=mi_estimator,
        precomputed_mask_stage1=precomputed_mask_stage1,
        precomputed_mask_stage2=precomputed_mask_stage2,
        n_shuffles_stage1=n_shuffles_stage1,
        n_shuffles_stage2=n_shuffles_stage2,
        joint_distr=joint_distr,
        allow_mixed_dimensions=allow_mixed_dimensions,
        metric_distr_type=metric_distr_type,
        noise_ampl=noise_ampl,
        ds=ds,
        topk1=topk1,
        topk2=topk2,
        multicomp_correction=multicomp_correction,
        pval_thr=pval_thr,
        find_optimal_delays=find_optimal_delays,
        skip_delays=[feat_ids.index(f) for f in skip_delays if f in feat_ids],
        shift_window=int(shift_window * exp.fps),
        verbose=verbose,
        enable_parallelization=enable_parallelization,
        n_jobs=n_jobs,
        seed=seed,
        duplicate_behavior=duplicate_behavior,
    )

    exp.optimal_nf_delays = info["optimal_delays"]
    # add hash data and update Experiment saved statistics and significance if needed
    for i, cell_id in enumerate(cell_ids):
        for j, feat_id in enumerate(feat_ids):
            # Check for non-existing feature if use_precomputed_stats==False
            if not use_precomputed_stats:
                if feat_id not in exp._data_hashes[data_type]:
                    raise ValueError(
                        f"Feature '{feat_id}' not found in data hashes. This may indicate the feature was not properly initialized."
                    )
            computed_stats[cell_id][feat_id]["data_hash"] = exp._data_hashes[data_type][
                feat_id
            ][cell_id]

            me_val = computed_stats[cell_id][feat_id].get("me")
            if me_val is not None and metric == "mi":
                feat_entropy = exp.get_feature_entropy(feat_id, ds=ds)
                # Get entropy from appropriate data type
                if data_type == "calcium":
                    neural_entropy = exp.neurons[int(cell_id)].ca.get_entropy(ds=ds)
                elif data_type == "spikes":
                    neural_entropy = exp.neurons[int(cell_id)].sp.get_entropy(ds=ds)
                computed_stats[cell_id][feat_id]["rel_me_beh"] = me_val / feat_entropy
                computed_stats[cell_id][feat_id]["rel_me_ca"] = me_val / neural_entropy

            if save_computed_stats:
                stage2_only = True if mode == "stage2" else False
                if combined_precomputed_mask[i, j]:
                    exp.update_neuron_feature_pair_stats(
                        computed_stats[cell_id][feat_id],
                        cell_id,
                        feat_id,
                        mode=data_type,
                        force_update=force_update,
                        stage2_only=stage2_only,
                    )

                    sig = computed_significance[cell_id][feat_id]
                    exp.update_neuron_feature_pair_significance(
                        sig, cell_id, feat_id, mode=data_type
                    )

    # save all results to a single object
    intense_params = {
        "neurons": {i: cell_ids[i] for i in range(len(cell_ids))},
        "feat_bunch": {i: feat_ids[i] for i in range(len(feat_ids))},
        "data_type": data_type,
        "mode": mode,
        "metric": metric,
        "n_shuffles_stage1": n_shuffles_stage1,
        "n_shuffles_stage2": n_shuffles_stage2,
        "joint_distr": joint_distr,
        "metric_distr_type": metric_distr_type,
        "noise_ampl": noise_ampl,
        "ds": ds,
        "topk1": topk1,
        "topk2": topk2,
        "multicomp_correction": multicomp_correction,
        "pval_thr": pval_thr,
        "find_optimal_delays": find_optimal_delays,
        "shift_window": shift_window,
    }

    intense_res = IntenseResults()
    intense_res.update('stats', computed_stats)
    intense_res.update('significance', computed_significance)
    intense_res.update("info", info)
    intense_res.update("intense_params", intense_params)

    # Perform disentanglement analysis if requested
    if with_disentanglement:
        if verbose:
            print("\nPerforming mixed selectivity disentanglement analysis...")

        # Step 1: Compute feature-feature significance
        _, feat_feat_significance, _, feat_names, _ = compute_feat_feat_significance(
            exp,
            feat_bunch=feat_bunch if feat_bunch is not None else "all",
            metric=metric,
            mode=mode,
            n_shuffles_stage1=n_shuffles_stage1,
            n_shuffles_stage2=n_shuffles_stage2 // 10,  # Reduce shuffles for feat-feat
            metric_distr_type=metric_distr_type,
            noise_ampl=noise_ampl,
            ds=ds,
            topk1=topk1,
            topk2=topk2,
            multicomp_correction=multicomp_correction,
            pval_thr=pval_thr,
            verbose=verbose,
            enable_parallelization=enable_parallelization,
            n_jobs=n_jobs,
            seed=seed,
        )

        # Step 2: Use default multifeature map if not provided
        if multifeature_map is None:
            multifeature_map = DEFAULT_MULTIFEATURE_MAP

        # Step 3: Run disentanglement analysis
        disent_matrix, count_matrix = disentangle_all_selectivities(
            exp,
            feat_names,
            ds=ds,
            multifeature_map=multifeature_map,
            feat_feat_significance=feat_feat_significance,
            cell_bunch=cell_ids,
        )

        # Step 4: Get summary statistics
        from .disentanglement import get_disentanglement_summary

        summary = get_disentanglement_summary(
            disent_matrix, count_matrix, feat_names, feat_feat_significance
        )

        # Package disentanglement results
        disentanglement_results = {
            "feat_feat_significance": feat_feat_significance,
            "disent_matrix": disent_matrix,
            "count_matrix": count_matrix,
            "feature_names": feat_names,
            "summary": summary,
        }

        # Add to IntenseResults
        intense_res.update("disentanglement", disentanglement_results)

        if verbose:
            print("\nDisentanglement analysis complete!")
            if summary.get("overall_stats"):
                print(
                    f"Total mixed selectivity pairs analyzed: {summary['overall_stats']['total_neuron_pairs']}"
                )
                if "redundancy_rate" in summary["overall_stats"]:
                    print(
                        f"Redundancy rate: {summary['overall_stats']['redundancy_rate']:.1f}%"
                    )
                if "independence_rate" in summary["overall_stats"]:
                    print(
                        f"Independence rate: {summary['overall_stats']['independence_rate']:.1f}%"
                    )
                if "true_mixed_selectivity_rate" in summary["overall_stats"]:
                    print(
                        f"True mixed selectivity rate: {summary['overall_stats']['true_mixed_selectivity_rate']:.1f}%"
                    )
            else:
                print("No mixed selectivity pairs found in the selected neurons.")

        # Return with disentanglement results
        return (
            computed_stats,
            computed_significance,
            info,
            intense_res,
            disentanglement_results,
        )

    # Return multiple values for backward compatibility
    return computed_stats, computed_significance, info, intense_res


def compute_feat_feat_significance(
    exp,
    feat_bunch="all",
    metric="mi",
    mi_estimator="gcmi",
    mode="two_stage",
    n_shuffles_stage1=100,
    n_shuffles_stage2=1000,
    metric_distr_type="gamma",
    noise_ampl=1e-3,
    ds=1,
    topk1=1,
    topk2=5,
    multicomp_correction="holm",
    pval_thr=0.01,
    verbose=True,
    enable_parallelization=True,
    n_jobs=-1,
    seed=42,
    duplicate_behavior="ignore",
    # FUTURE: Add save_computed_stats=True, use_precomputed_stats=True parameters
    # to enable caching of feat-feat results in experiment object similar to cell-feat
):
    """
    Compute pairwise significance between all behavioral features.

    This function calculates pairwise similarity (e.g., mutual information) between
    all behavioral features using the two-stage INTENSE approach. The diagonal
    elements are set to zero as self-similarity is prevented by the check_for_coincidence
    mechanism in get_mi.

    Parameters
    ----------
    exp : Experiment
        Experiment object containing behavioral data.
    feat_bunch : str, list or None
        Feature names to analyze. Default: 'all' (all features including multifeatures).
        Can be a list of specific feature names.
    metric : str, optional
        Similarity metric to use. Default: 'mi' (mutual information).
    mi_estimator : str, optional
        Mutual information estimator to use when metric='mi'. Default: 'gcmi'.
        Options: 'gcmi' or 'ksg'
    mode : str, optional
        Computation mode: 'two_stage', 'stage1', or 'stage2'. Default: 'two_stage'.
    n_shuffles_stage1 : int, optional
        Number of shuffles for stage 1. Default: 100.
    n_shuffles_stage2 : int, optional
        Number of shuffles for stage 2. Default: 1000.
    metric_distr_type : str, optional
        Distribution type for metric null distribution. Default: 'gamma'.
    noise_ampl : float, optional
        Small noise amplitude for numerical stability. Default: 1e-3.
    ds : int, optional
        Downsampling factor. Default: 1.
    topk1 : int, optional
        Top-k criterion for stage 1. Default: 1.
    topk2 : int, optional
        Top-k criterion for stage 2. Default: 5.
    multicomp_correction : str or None, optional
        Multiple comparison correction method. Default: 'holm'.
    pval_thr : float, optional
        P-value threshold for significance. Default: 0.01.
    verbose : bool, optional
        Whether to print progress information. Default: True.
    enable_parallelization : bool, optional
        Whether to use parallel processing. Default: True.
    n_jobs : int, optional
        Number of parallel jobs. -1 means use all processors. Default: -1.
    seed : int, optional
        Random seed for reproducibility. Default: 42.
    duplicate_behavior : str, optional
        How to handle duplicate TimeSeries in ts_bunch1 or ts_bunch2.
        - 'ignore': Process duplicates normally (default)
        - 'raise': Raise an error if duplicates are found
        - 'warn': Print a warning but continue processing
        Default: 'ignore'.

    Returns
    -------
    similarity_matrix : ndarray
        Matrix of similarity values between features. Element [i,j] contains
        the similarity between feature i and feature j. Diagonal is zero.
    significance_matrix : ndarray
        Matrix of binary significance values. 1 indicates significant similarity.
    p_value_matrix : ndarray
        Matrix of p-values for each comparison.
    feature_names : list
        List of feature names corresponding to matrix indices.
        May include tuples for multifeatures (e.g., ('x', 'y')).
    info : dict
        Dictionary containing additional information from compute_me_stats.

    Notes
    -----
    - Uses the two-stage INTENSE approach for efficient significance testing
    - Diagonal elements are zero (self-similarity check prevents computation)
    - The function handles both discrete and continuous variables
    - Supports MultiTimeSeries (e.g., place fields from x,y coordinates)
    - For mutual information, values are in bits
    - No optimal delay search is performed (delays are set to 0)

    Examples
    --------
    >>> from driada.experiment.synthetic import generate_synthetic_exp
    >>> 
    >>> # Create test experiment
    >>> exp = generate_synthetic_exp(n_dfeats=2, n_cfeats=2, nneurons=3,
    ...                              duration=60, fps=10, seed=42, verbose=False)
    >>> 
    >>> # Compute feature-feature correlations
    >>> sim_mat, sig_mat, pval_mat, features, info = compute_feat_feat_significance(
    ...     exp,
    ...     mode='stage1',
    ...     n_shuffles_stage1=10,
    ...     verbose=False
    ... )
    >>> sim_mat.shape == (4, 4)  # 2 discrete + 2 continuous features
    True
    >>> np.allclose(np.diag(sim_mat), 0)  # Diagonal is zero
    True
    >>> 
    >>> # Analyze specific features only  
    >>> sim_mat2, sig_mat2, pval_mat2, features2, info2 = compute_feat_feat_significance(
    ...     exp,
    ...     feat_bunch=['d_feat_0', 'd_feat_1'],
    ...     mode='stage1',
    ...     n_shuffles_stage1=10,
    ...     verbose=False
    ... )
    >>> sim_mat2.shape == (2, 2)
    True
    
    Raises
    ------
    ValueError
        If features are not found in experiment
    
    Notes
    -----
    - Only upper triangle is computed for efficiency (matrix is symmetric)
    - Diagonal elements are always zero (self-similarity prevented)
    - No delay optimization is performed between features
    - Supports both discrete and continuous features
    - Multifeatures are created using aggregate_multiple_ts    """
    import numpy as np

    # Process feature bunch - default is all features
    if feat_bunch == "all":
        feat_bunch = None  # None means all features in _process_fbunch
    feat_ids = exp._process_fbunch(feat_bunch, allow_multifeatures=True, mode="calcium")
    n_features = len(feat_ids)

    # Handle empty feature list case
    if n_features == 0:
        if verbose:
            print("No features to analyze - returning empty results")
        return (
            np.array([]).reshape(0, 0),  # similarity_matrix
            np.array([]).reshape(0, 0),  # significance_matrix
            np.array([]).reshape(0, 0),  # p_value_matrix
            [],  # feature_names
            {},  # info
        )

    if verbose:
        print(f"Computing behavioral similarity matrix for {n_features} features...")
        print(f"Features: {feat_ids}")
        # Note: computing only unique pairs (upper triangle), not all n²
        n_unique_pairs = n_features * (n_features - 1) // 2
        print(f"Unique pairs to compute: {n_unique_pairs} (avoiding redundancy)")

    # Get TimeSeries/MultiTimeSeries objects for all features
    from ..information.info_base import aggregate_multiple_ts

    feature_ts = []
    for feat_id in feat_ids:
        if isinstance(feat_id, tuple):
            # Create MultiTimeSeries for tuples using aggregate_multiple_ts
            ts_list = [exp.dynamic_features[f] for f in feat_id]
            ts = aggregate_multiple_ts(*ts_list)
        else:
            ts = exp.dynamic_features[feat_id]
        feature_ts.append(ts)

    # Create masks that exclude diagonal (self-comparisons) AND lower triangle
    # This ensures we only compute the upper triangle for symmetric results
    precomputed_mask_stage1 = np.triu(np.ones((n_features, n_features)), k=1)
    precomputed_mask_stage2 = np.triu(np.ones((n_features, n_features)), k=1)

    # Call compute_me_stats with features against themselves
    # Note: optimal delays are disabled (set to False)
    stats, significance, info = compute_me_stats(
        feature_ts,
        feature_ts,
        names1=feat_ids,
        names2=feat_ids,
        metric=metric,
        mi_estimator=mi_estimator,
        mode=mode,
        precomputed_mask_stage1=precomputed_mask_stage1,
        precomputed_mask_stage2=precomputed_mask_stage2,
        n_shuffles_stage1=n_shuffles_stage1,
        n_shuffles_stage2=n_shuffles_stage2,
        joint_distr=False,
        allow_mixed_dimensions=True,  # Allow MultiTimeSeries
        metric_distr_type=metric_distr_type,
        noise_ampl=noise_ampl,
        ds=ds,
        topk1=topk1,
        topk2=topk2,
        multicomp_correction=multicomp_correction,
        pval_thr=pval_thr,
        find_optimal_delays=False,  # No delay optimization
        shift_window=0,  # No shift window needed
        verbose=verbose,
        enable_parallelization=enable_parallelization,
        n_jobs=n_jobs,
        seed=seed,
        duplicate_behavior="ignore",  # Default behavior for feature-feature comparison
    )

    # Extract matrices from results
    similarity_matrix = np.zeros((n_features, n_features))
    significance_matrix = np.zeros((n_features, n_features))
    p_value_matrix = np.ones((n_features, n_features))

    # Fill matrices from stats and significance dictionaries
    # Since we only computed upper triangle, we need to fill both upper and lower
    for i, feat1 in enumerate(feat_ids):
        for j, feat2 in enumerate(feat_ids):
            if i == j:
                # Diagonal is already 0
                continue

            # Convert tuples to strings for dictionary keys if needed
            key1 = str(feat1) if isinstance(feat1, tuple) else feat1
            key2 = str(feat2) if isinstance(feat2, tuple) else feat2

            # We computed only upper triangle, so check if this pair was computed
            if i < j:
                # Upper triangle - get from stats
                if key1 in stats and key2 in stats[key1]:
                    stats_dict = stats[key1][key2]
                    if stats_dict:  # Check if dict is not empty
                        similarity_matrix[i, j] = stats_dict.get("me", 0)
                        p_value_matrix[i, j] = stats_dict.get("p", 1)

                    sig_dict = significance.get(key1, {}).get(key2, {})
                    if sig_dict.get("stage2") is not None:
                        significance_matrix[i, j] = float(sig_dict["stage2"])
                    elif sig_dict.get("stage1") is not None:
                        significance_matrix[i, j] = float(sig_dict["stage1"])
            else:
                # Lower triangle - copy from upper triangle for symmetry
                similarity_matrix[i, j] = similarity_matrix[j, i]
                p_value_matrix[i, j] = p_value_matrix[j, i]
                significance_matrix[i, j] = significance_matrix[j, i]

    # Ensure diagonal is zero (should already be due to coincidence check)
    np.fill_diagonal(similarity_matrix, 0)
    np.fill_diagonal(significance_matrix, 0)
    np.fill_diagonal(p_value_matrix, 1)

    if verbose:
        print("\nBehavioral similarity matrix computation complete!")
        print(f"Feature pairs analyzed: {n_features * n_features}")
        print(f"Significant pairs (stage 1): {info.get('n_significant_stage1', 0)}")
        print(f"Significant pairs (final): {np.sum(significance_matrix)}")
        # Count unique significant pairs (upper triangle only)
        unique_sig = np.sum(np.triu(significance_matrix, k=1))
        print(f"Unique significant pairs: {unique_sig}")

    return similarity_matrix, significance_matrix, p_value_matrix, feat_ids, info


def compute_cell_cell_significance(
    exp,
    cell_bunch=None,
    data_type="calcium",
    metric="mi",
    mi_estimator="gcmi",
    mode="two_stage",
    n_shuffles_stage1=100,
    n_shuffles_stage2=1000,
    metric_distr_type="gamma",
    noise_ampl=1e-3,
    ds=1,
    topk1=1,
    topk2=5,
    multicomp_correction="holm",
    pval_thr=0.01,
    verbose=True,
    enable_parallelization=True,
    n_jobs=-1,
    seed=42,
    duplicate_behavior="ignore",
    # FUTURE: Add save_computed_stats=True, use_precomputed_stats=True parameters
    # to enable caching of cell-cell results in experiment object similar to cell-feat
):
    """
    Compute pairwise functional correlations between neurons using INTENSE.

    This function calculates pairwise similarity (e.g., mutual information) between
    all neurons using the two-stage INTENSE approach. This can reveal functionally
    correlated neurons that may form assemblies or functional modules.

    Parameters
    ----------
    exp : Experiment
        Experiment object containing neural data.
    cell_bunch : int, list or None, optional
        Neuron indices to analyze. Default: None (all neurons).
    data_type : str, optional
        Type of neural data: 'calcium' or 'spikes'. Default: 'calcium'.
    metric : str, optional
        Similarity metric to use. Default: 'mi' (mutual information).
    mi_estimator : str, optional
        Mutual information estimator to use when metric='mi'. Default: 'gcmi'.
        Options: 'gcmi' or 'ksg'
    mode : str, optional
        Computation mode: 'two_stage', 'stage1', or 'stage2'. Default: 'two_stage'.
    n_shuffles_stage1 : int, optional
        Number of shuffles for stage 1. Default: 100.
    n_shuffles_stage2 : int, optional
        Number of shuffles for stage 2. Default: 1000.
    metric_distr_type : str, optional
        Distribution type for metric null distribution. Default: 'gamma'.
    noise_ampl : float, optional
        Small noise amplitude for numerical stability. Default: 1e-3.
    ds : int, optional
        Downsampling factor. Default: 1.
    topk1 : int, optional
        Top-k criterion for stage 1. Default: 1.
    topk2 : int, optional
        Top-k criterion for stage 2. Default: 5.
    multicomp_correction : str or None, optional
        Multiple comparison correction method. Default: 'holm'.
    pval_thr : float, optional
        P-value threshold for significance. Default: 0.01.
    verbose : bool, optional
        Whether to print progress information. Default: True.
    enable_parallelization : bool, optional
        Whether to use parallel processing. Default: True.
    n_jobs : int, optional
        Number of parallel jobs. -1 means use all processors. Default: -1.
    seed : int, optional
        Random seed for reproducibility. Default: 42.
    duplicate_behavior : str, optional
        How to handle duplicate TimeSeries in ts_bunch1 or ts_bunch2.
        - 'ignore': Process duplicates normally (default)
        - 'raise': Raise an error if duplicates are found
        - 'warn': Print a warning but continue processing
        Default: 'ignore'.

    Returns
    -------
    similarity_matrix : ndarray
        Matrix of similarity values between neurons. Element [i,j] contains
        the similarity between neuron i and neuron j. Diagonal is zero.
    significance_matrix : ndarray
        Matrix of binary significance values. 1 indicates significant similarity.
    p_value_matrix : ndarray
        Matrix of p-values for each comparison.
    cell_ids : list
        List of cell IDs corresponding to matrix indices.
    info : dict
        Dictionary containing additional information from compute_me_stats.

    Notes
    -----
    - Uses the two-stage INTENSE approach for efficient significance testing
    - Diagonal elements are zero (self-similarity check prevents computation)
    - For calcium imaging data, considers temporal dynamics
    - For spike data, uses discrete MI formulation
    - Can identify functional assemblies through graph analysis of significant pairs
    - No optimal delay search is performed (synchronous activity assumed)

    Examples
    --------
    >>> from driada.experiment.synthetic import generate_synthetic_exp
    >>> from driada.information.info_base import TimeSeries
    >>> import numpy as np
    >>> 
    >>> # Create experiment with correlated neurons
    >>> exp = generate_synthetic_exp(n_dfeats=1, n_cfeats=1, nneurons=3,
    ...                              duration=60, fps=10, seed=42, verbose=False)
    >>> 
    >>> # Make neurons 0 and 1 correlated
    >>> noise = np.random.RandomState(42).randn(len(exp.neurons[0].ca.data)) * 0.1
    >>> exp.neurons[1].ca = TimeSeries(
    ...     exp.neurons[0].ca.data + noise, discrete=False
    ... )
    >>> 
    >>> # Compute neuron-neuron correlations
    >>> sim_mat, sig_mat, pval_mat, cells, info = compute_cell_cell_significance(
    ...     exp,
    ...     cell_bunch=[0, 1, 2],
    ...     mode='stage1',
    ...     n_shuffles_stage1=10,
    ...     verbose=False
    ... )
    >>> sim_mat.shape == (3, 3)
    True
    >>> np.allclose(np.diag(sim_mat), 0)  # Self-correlation is zero
    True
    >>> sim_mat[0, 1] > sim_mat[0, 2]  # Neurons 0,1 more correlated than 0,2
    True
    
    Raises
    ------
    ValueError
        If data_type is not 'calcium' or 'spikes'
        If spike data is missing for requested neurons
    
    Notes
    -----
    - Only upper triangle is computed for efficiency (matrix is symmetric)
    - Warns if all neurons have identical spike data
    - Computes network statistics when verbose=True
    - Synchronous activity assumed (no delay optimization)    """
    import numpy as np

    # Check downsampling
    exp.check_ds(ds)

    # Process cell bunch
    cell_ids = exp._process_cbunch(cell_bunch)
    n_cells = len(cell_ids)
    cells = [exp.neurons[cell_id] for cell_id in cell_ids]

    if verbose:
        print(f"Computing neuronal similarity matrix for {n_cells} neurons...")
        print(f"Data type: {data_type}")
        # Note: computing only unique pairs (upper triangle), not all n²
        n_unique_pairs = n_cells * (n_cells - 1) // 2
        print(f"Unique pairs to compute: {n_unique_pairs} (avoiding redundancy)")

    # Get neural signals based on data type
    if data_type == "calcium":
        signals = [cell.ca for cell in cells]
    elif data_type == "spikes":
        signals = [cell.sp for cell in cells]
        # Check if spike data exists and is non-degenerate
        if any(sig is None for sig in signals):
            raise ValueError(
                "Some neurons have no spike data. Use reconstruct_spikes or provide spike data."
            )
        # Check if all spike data is identical (e.g., all zeros)
        if len(signals) > 1:
            first_data = signals[0].data
            if all(np.array_equal(sig.data, first_data) for sig in signals[1:]):
                import warnings

                warnings.warn(
                    "All neurons have identical spike data. This may lead to degenerate results."
                )
    else:
        raise ValueError('"data_type" can be either "calcium" or "spikes"')

    # Create masks that exclude diagonal (self-comparisons) AND lower triangle
    # This ensures we only compute the upper triangle for symmetric results
    precomputed_mask_stage1 = np.triu(np.ones((n_cells, n_cells)), k=1)
    precomputed_mask_stage2 = np.triu(np.ones((n_cells, n_cells)), k=1)

    # Call compute_me_stats with neurons against themselves
    # Note: optimal delays are disabled (set to False) for synchronous analysis
    stats, significance, info = compute_me_stats(
        signals,
        signals,
        names1=cell_ids,
        names2=cell_ids,
        metric=metric,
        mi_estimator=mi_estimator,
        mode=mode,
        precomputed_mask_stage1=precomputed_mask_stage1,
        precomputed_mask_stage2=precomputed_mask_stage2,
        n_shuffles_stage1=n_shuffles_stage1,
        n_shuffles_stage2=n_shuffles_stage2,
        joint_distr=False,
        allow_mixed_dimensions=False,  # Neurons are single time series
        metric_distr_type=metric_distr_type,
        noise_ampl=noise_ampl,
        ds=ds,
        topk1=topk1,
        topk2=topk2,
        multicomp_correction=multicomp_correction,
        pval_thr=pval_thr,
        find_optimal_delays=False,  # Assume synchronous activity
        shift_window=0,  # No shift window needed
        verbose=verbose,
        enable_parallelization=enable_parallelization,
        n_jobs=n_jobs,
        seed=seed,
        duplicate_behavior="ignore",  # Default behavior for cell-cell comparison
    )

    # Extract matrices from results
    similarity_matrix = np.zeros((n_cells, n_cells))
    significance_matrix = np.zeros((n_cells, n_cells))
    p_value_matrix = np.ones((n_cells, n_cells))

    # Fill matrices from stats and significance dictionaries
    # Since we only computed upper triangle, we need to fill both upper and lower
    for i, cell1 in enumerate(cell_ids):
        for j, cell2 in enumerate(cell_ids):
            if i == j:
                # Diagonal is already 0
                continue

            # We computed only upper triangle, so check if this pair was computed
            if i < j:
                # Upper triangle - get from stats
                if cell1 in stats and cell2 in stats[cell1]:
                    stats_dict = stats[cell1][cell2]
                    if stats_dict:  # Check if dict is not empty
                        similarity_matrix[i, j] = stats_dict.get("me", 0)
                        p_value_matrix[i, j] = stats_dict.get("p", 1)

                    sig_dict = significance.get(cell1, {}).get(cell2, {})
                    if sig_dict.get("stage2") is not None:
                        significance_matrix[i, j] = float(sig_dict["stage2"])
                    elif sig_dict.get("stage1") is not None:
                        significance_matrix[i, j] = float(sig_dict["stage1"])
            else:
                # Lower triangle - copy from upper triangle for symmetry
                similarity_matrix[i, j] = similarity_matrix[j, i]
                p_value_matrix[i, j] = p_value_matrix[j, i]
                significance_matrix[i, j] = significance_matrix[j, i]

    # Ensure diagonal is zero (should already be due to coincidence check)
    np.fill_diagonal(similarity_matrix, 0)
    np.fill_diagonal(significance_matrix, 0)
    np.fill_diagonal(p_value_matrix, 1)

    if verbose:
        print("\nNeuronal similarity matrix computation complete!")
        print(f"Neuron pairs analyzed: {n_cells * n_cells}")
        print(f"Significant pairs (stage 1): {info.get('n_significant_stage1', 0)}")
        print(f"Significant pairs (final): {np.sum(significance_matrix)}")
        # Count unique significant pairs (upper triangle only)
        unique_sig = np.sum(np.triu(significance_matrix, k=1))
        print(f"Unique significant pairs: {unique_sig}")

        # Basic network statistics
        if unique_sig > 0:
            avg_connections = np.sum(significance_matrix) / n_cells
            print(f"Average connections per neuron: {avg_connections:.2f}")
            max_connections = np.max(np.sum(significance_matrix, axis=1))
            print(f"Maximum connections for a single neuron: {int(max_connections)}")

    return similarity_matrix, significance_matrix, p_value_matrix, cell_ids, info


def compute_embedding_selectivity(
    exp,
    embedding_methods=None,
    cell_bunch=None,
    data_type="calcium",
    metric="mi",
    mi_estimator="gcmi",
    mode="two_stage",
    n_shuffles_stage1=100,
    n_shuffles_stage2=10000,
    metric_distr_type="gamma",
    noise_ampl=1e-3,
    ds=1,
    use_precomputed_stats=True,
    save_computed_stats=True,
    force_update=False,
    topk1=1,
    topk2=5,
    multicomp_correction="holm",
    pval_thr=0.01,
    find_optimal_delays=True,
    shift_window=5,
    verbose=True,
    enable_parallelization=True,
    n_jobs=-1,
    seed=42,
):
    """
    Compute INTENSE selectivity between neurons and dimensionality reduction embeddings.

    This function treats each embedding component as a dynamic feature and computes
    the mutual information between neural activity and embedding dimensions. This reveals
    how individual neurons contribute to the population-level manifold structure.

    Parameters
    ----------
    exp : Experiment
        Experiment object with stored embeddings
    embedding_methods : str, list or None
        Names of embedding methods to analyze. If None, analyzes all stored embeddings.
    cell_bunch : int, iterable or None
        Neuron indices. By default (None), all neurons will be taken
    data_type : str
        Data type used for embeddings and INTENSE ('calcium' or 'spikes')
    metric : str
        Similarity metric between TimeSeries (default: 'mi')
    mi_estimator : str
        Mutual information estimator to use when metric='mi'. Default: 'gcmi'.
        Options: 'gcmi' or 'ksg'
    mode : str
        Computation mode: 'stage1', 'stage2', or 'two_stage' (default)
    n_shuffles_stage1 : int
        Number of shuffles for first stage (default: 100)
    n_shuffles_stage2 : int
        Number of shuffles for second stage (default: 10000)
    metric_distr_type : str
        Distribution type for shuffled metric distribution fit (default: 'norm')
    noise_ampl : float
        Small noise amplitude added to improve numerical fit (default: 1e-3)
    ds : int
        Downsampling constant (default: 1)
    use_precomputed_stats : bool
        Whether to use stats saved in Experiment instance (default: True)
    save_computed_stats : bool
        Whether to save computed stats to Experiment instance (default: True)
    force_update : bool
        Force update saved statistics if data hash collision found (default: False)
    topk1 : int
        True MI for stage 1 should be among topk1 MI shuffles (default: 1)
    topk2 : int
        True MI for stage 2 should be among topk2 MI shuffles (default: 5)
    multicomp_correction : str or None
        Multiple comparison correction type: None, 'bonferroni', or 'holm' (default)
    pval_thr : float
        P-value threshold (default: 0.01)
    find_optimal_delays : bool
        Find optimal temporal delays between neural activity and embeddings (default: True)
    shift_window : int
        Window for optimal shift search in seconds (default: 5)
    verbose : bool
        Print progress information (default: True)
    enable_parallelization : bool
        Enable parallel computation (default: True)
    n_jobs : int
        Number of parallel jobs, -1 for all cores (default: -1)
    seed : int
        Random seed (default: 42)

    Returns
    -------
    results : dict
        Dictionary with keys as embedding method names, each containing:
        - 'stats': Statistics for each neuron-component pair
        - 'significance': Significance results
        - 'info': Additional information from compute_me_stats
        - 'intense_results': Full IntenseResults object from INTENSE computation
        - 'significant_neurons': Dict of neurons significantly selective to embedding components
        - 'n_components': Number of embedding components
        - 'component_selectivity': For each component, list of selective neurons
    
    Raises
    ------
    ValueError
        If no embeddings found for specified data_type
        If embedding method not found
    
    Notes
    -----
    - Temporarily adds embedding components as dynamic features
    - Forces use_precomputed_stats=False for temporary features
    - Component names follow pattern "{method}_comp{index}"
    - Cleanup in finally block ensures experiment state restored
    - Only stage2 significance is considered for results
    
    Examples
    --------
    >>> from driada.experiment.synthetic import generate_synthetic_exp
    >>> from sklearn.decomposition import PCA
    >>> import numpy as np
    >>> 
    >>> # Create experiment
    >>> exp = generate_synthetic_exp(n_dfeats=1, n_cfeats=1, nneurons=5,
    ...                              duration=60, fps=10, seed=42, verbose=False)
    >>> 
    >>> # Create and store PCA embedding
    >>> neural_data = np.array([exp.neurons[i].ca.data for i in range(5)]).T
    >>> pca = PCA(n_components=2, random_state=42)
    >>> embedding = pca.fit_transform(neural_data)
    >>> exp.store_embedding(embedding, method_name='pca', data_type='calcium')
    >>> 
    >>> # Compute embedding selectivity
    >>> results = compute_embedding_selectivity(
    ...     exp,
    ...     embedding_methods=['pca'],
    ...     cell_bunch=[0, 1, 2],
    ...     mode='stage1',
    ...     n_shuffles_stage1=10,
    ...     verbose=False
    ... )  # doctest: +ELLIPSIS
    ...
    >>> 
    >>> 'pca' in results
    True
    >>> results['pca']['n_components']
    2
    >>> 'component_selectivity' in results['pca']
    True
    
    See Also
    --------
    ~driada.intense.pipelines.compute_cell_feat_significance : Compute selectivity for behavioral features
    ~driada.integration.manifold_analysis.get_functional_organization : Analyze organization in embeddings
    ~driada.integration.manifold_analysis.compare_embeddings : Compare multiple embedding methods    """

    # Get list of embedding methods to analyze
    if embedding_methods is None:
        embedding_methods = list(exp.embeddings[data_type].keys())
    elif isinstance(embedding_methods, str):
        embedding_methods = [embedding_methods]

    if not embedding_methods:
        raise ValueError(
            f"No embeddings found for data_type '{data_type}'. "
            "Use exp.store_embedding() to add embeddings first."
        )

    results = {}

    # Process each embedding method
    for method_name in embedding_methods:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Computing selectivity for embedding: {method_name}")
            print(f"{'='*60}")

        # Get embedding data
        embedding_dict = exp.get_embedding(method_name, data_type)
        embedding_data = embedding_dict["data"]
        n_components = embedding_data.shape[1]

        # Create TimeSeries for each embedding component
        embedding_features = {}
        for comp_idx in range(n_components):
            feat_name = f"{method_name}_comp{comp_idx}"
            embedding_features[feat_name] = TimeSeries(
                embedding_data[:, comp_idx], discrete=False
            )

        # Temporarily add embedding components to dynamic features
        original_features = exp.dynamic_features.copy()
        exp.dynamic_features.update(embedding_features)

        # Also update internal experiment attributes for the new features
        for feat_name, feat_ts in embedding_features.items():
            setattr(exp, feat_name, feat_ts)

        # Rebuild data hashes to include new features
        exp._build_data_hashes(mode=data_type)

        # Initialize stats tables if not already done
        if save_computed_stats and data_type not in exp.stats_tables:
            exp._set_selectivity_tables(data_type)

        try:
            # Run INTENSE analysis
            stats, significance, info, intense_res = compute_cell_feat_significance(
                exp,
                cell_bunch=cell_bunch,
                feat_bunch=list(embedding_features.keys()),
                data_type=data_type,
                metric=metric,
                mi_estimator=mi_estimator,
                mode=mode,
                n_shuffles_stage1=n_shuffles_stage1,
                n_shuffles_stage2=n_shuffles_stage2,
                metric_distr_type=metric_distr_type,
                noise_ampl=noise_ampl,
                ds=ds,
                use_precomputed_stats=False,  # Must be False for new dynamic features
                save_computed_stats=False,  # Don't save stats for temporary embedding features
                force_update=force_update,
                topk1=topk1,
                topk2=topk2,
                multicomp_correction=multicomp_correction,
                pval_thr=pval_thr,
                find_optimal_delays=find_optimal_delays,
                shift_window=shift_window,
                verbose=verbose,
                enable_parallelization=enable_parallelization,
                n_jobs=n_jobs,
                seed=seed,
            )

            # Extract significant neurons from the significance results
            # Note: significance structure is significance[neuron_id][feat_name]
            significant_neurons = {}
            for neuron_id in significance.keys():
                for feat_name in embedding_features.keys():
                    if feat_name in significance[neuron_id]:
                        sig_info = significance[neuron_id][feat_name]
                        if sig_info.get(
                            "stage2", False
                        ):  # Check if significant in stage 2
                            if neuron_id not in significant_neurons:
                                significant_neurons[neuron_id] = []
                            significant_neurons[neuron_id].append(feat_name)

            # Organize component selectivity
            component_selectivity = {comp_idx: [] for comp_idx in range(n_components)}
            for neuron_id, features in significant_neurons.items():
                for feat in features:
                    comp_idx = int(feat.split("_comp")[-1])
                    component_selectivity[comp_idx].append(neuron_id)

            # Store results
            results[method_name] = {
                "stats": stats,
                "significance": significance,
                "info": info,
                "intense_results": intense_res,  # Include the full IntenseResults object
                "significant_neurons": significant_neurons,
                "n_components": n_components,
                "component_selectivity": component_selectivity,
                "embedding_metadata": embedding_dict.get("metadata", {}),
            }

            if verbose:
                n_sig_neurons = len(significant_neurons)
                n_total_neurons = len(exp._process_cbunch(cell_bunch))
                print(f"\nResults for {method_name}:")
                print(f"  Embedding dimensions: {n_components}")
                print(
                    f"  Significant neurons: {n_sig_neurons}/{n_total_neurons} ({100*n_sig_neurons/n_total_neurons:.1f}%)"
                )

                # Component-wise summary
                for comp_idx in range(n_components):
                    n_selective = len(component_selectivity[comp_idx])
                    if n_selective > 0:
                        print(
                            f"  Component {comp_idx}: {n_selective} selective neurons"
                        )

        finally:
            # Restore original features
            exp.dynamic_features = original_features

            # Remove temporary attributes
            for feat_name in embedding_features.keys():
                if hasattr(exp, feat_name):
                    delattr(exp, feat_name)

    return results
