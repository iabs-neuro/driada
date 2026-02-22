"""
Mixed selectivity disentanglement analysis for INTENSE.

This module provides functions to analyze and disentangle mixed selectivity
in neural responses when neurons respond to multiple, potentially correlated
behavioral variables.
"""

import numpy as np
from itertools import combinations
from joblib import Parallel, delayed
from ..information.info_base import get_mi, conditional_mi, MultiTimeSeries
from ..information.gcmi import cmi_ggg
import driada  # For PARALLEL_BACKEND
from .intense_base import _parallel_executor


DEFAULT_MULTIFEATURE_MAP = {
    ("x", "y"): "place",  # 2D spatial location multifeature
    ("x", "y", "z"): "3d-place",  # 3D spatial location multifeature
}
"""Default multifeature mapping for common behavioral variable combinations.

Maps component tuples to their semantic names:

- ``("x", "y")``: mapped to ``"place"`` (2D spatial location)
- ``("x", "y", "z")``: mapped to ``"3d-place"`` (3D spatial location)
"""

# Epsilon tolerance for floating-point comparisons
# MI values are computed via numerical estimation and rarely equal exactly 0.0
MI_EPSILON = 1e-6

# Ratio threshold for "dominant" feature detection in synergy cases
# 2.0 means one feature has >2x the MI of the other, indicating strong dominance
DOMINANCE_RATIO_THRESHOLD = 2.0

# Valid disentanglement result values
VALID_DISRES_VALUES = (0, 0.5, 1)


def _flip_decision(decision):
    """Flip disentanglement decision: 0↔1, keep 0.5 unchanged.

    Parameters
    ----------
    decision : float
        Original decision (0, 0.5, or 1).

    Returns
    -------
    float
        Flipped decision.
    """
    return {0: 1, 1: 0, 0.5: 0.5}[decision]


def _downsample_copnorm(data, ds):
    """Downsample copula-normalized data along time axis.

    Parameters
    ----------
    data : ndarray
        Copula-normalized data (1D for TimeSeries, 2D for MultiTimeSeries).
    ds : int
        Downsampling factor.

    Returns
    -------
    ndarray
        Downsampled data.
    """
    if data.ndim == 1:
        return data[::ds]
    else:
        return data[:, ::ds]


def _lookup_cell_feat_mi(cell_feat_stats, neuron_id, feat_name):
    """Look up MI(neuron, feature) from pre-computed stats dict.

    Parameters
    ----------
    cell_feat_stats : dict or None
        Nested dictionary: stats[cell_id][feat_name]["me"] = MI value.
    neuron_id : any
        Neuron identifier (cell ID).
    feat_name : str or tuple
        Feature name/identifier.

    Returns
    -------
    float or None
        The pre-computed MI value, or None if not found.
    """
    if cell_feat_stats is None:
        return None
    try:
        return cell_feat_stats[neuron_id][feat_name].get("me")
    except (KeyError, TypeError):
        return None


def _lookup_feat_feat_mi(feat_feat_similarity, feat_names, feat1_name, feat2_name):
    """Look up MI(feature1, feature2) from pre-computed similarity matrix.

    Parameters
    ----------
    feat_feat_similarity : ndarray or None
        Symmetric matrix where [i, j] = MI(feat_i, feat_j).
    feat_names : list
        List of feature names corresponding to matrix indices.
    feat1_name : str or tuple
        First feature name.
    feat2_name : str or tuple
        Second feature name.

    Returns
    -------
    float or None
        The pre-computed MI value, or None if not found.
    """
    if feat_feat_similarity is None or feat_names is None:
        return None
    try:
        ind1 = feat_names.index(feat1_name)
        ind2 = feat_names.index(feat2_name)
        return feat_feat_similarity[ind1, ind2]
    except (ValueError, IndexError):
        return None


def _disentangle_pair_with_precomputed(
    ts1, ts2, ts3,
    mi12=None,
    mi13=None,
    mi23=None,
    ts1_copnorm=None,
    ts2_copnorm=None,
    ts3_copnorm=None,
    verbose=False,
    ds=1,
):
    """Disentangle with optional pre-computed MI values and copula data.

    Internal function that performs disentanglement analysis, optionally
    using pre-computed pairwise MI values and copula-normalized data to
    avoid redundant computation.

    Parameters
    ----------
    ts1 : TimeSeries
        Neural activity time series.
    ts2 : TimeSeries
        First behavioral variable.
    ts3 : TimeSeries
        Second behavioral variable.
    mi12 : float or None, optional
        Pre-computed MI(ts1, ts2). Computed if None.
    mi13 : float or None, optional
        Pre-computed MI(ts1, ts3). Computed if None.
    mi23 : float or None, optional
        Pre-computed MI(ts2, ts3). Computed if None.
    ts1_copnorm : ndarray or None, optional
        Pre-computed copula-normalized data for ts1 (downsampled).
    ts2_copnorm : ndarray or None, optional
        Pre-computed copula-normalized data for ts2 (downsampled).
    ts3_copnorm : ndarray or None, optional
        Pre-computed copula-normalized data for ts3 (downsampled).
    verbose : bool, optional
        If True, print detailed analysis results. Default: False.
    ds : int, optional
        Downsampling factor. Default: 1.

    Returns
    -------
    float
        Disentanglement result (0, 0.5, or 1).
    """
    # Compute only missing pairwise MI values
    if mi12 is None:
        mi12 = get_mi(ts1, ts2, ds=ds)
    if mi13 is None:
        mi13 = get_mi(ts1, ts3, ds=ds)
    if mi23 is None:
        mi23 = get_mi(ts2, ts3, ds=ds)

    # Compute conditional MI - use pre-computed copula data if all available (CCC case)
    if ts1_copnorm is not None and ts2_copnorm is not None and ts3_copnorm is not None:
        # Direct cmi_ggg call with pre-cached copula data (faster)
        cmi123 = cmi_ggg(ts1_copnorm, ts2_copnorm, ts3_copnorm, biascorrect=True, demeaned=True)
        cmi132 = cmi_ggg(ts1_copnorm, ts3_copnorm, ts2_copnorm, biascorrect=True, demeaned=True)
    else:
        # Fallback for mixed discrete/continuous cases
        cmi123 = conditional_mi(ts1, ts2, ts3, ds=ds)  # MI(neuron, behavior1 | behavior2)
        cmi132 = conditional_mi(ts1, ts3, ts2, ds=ds)  # MI(neuron, behavior2 | behavior1)

    # Compute interaction information (average of two equivalent formulas)
    I_av = np.mean([cmi123 - mi12, cmi132 - mi13])

    if verbose:
        print()
        print("MI(A,X):", mi12)
        print("MI(A,Y):", mi13)
        print("MI(X,Y):", mi23)

        print()
        print("MI(A,X|Y):", cmi123)
        print("MI(A,Y|X):", cmi132)

        print()
        print("MI(A,X|Y) / MI(A,X):", np.round(cmi123 / mi12, 3) if mi12 > 0 else "N/A")
        print("MI(A,Y|X) / MI(A,Y):", np.round(cmi132 / mi13, 3) if mi13 > 0 else "N/A")

        print()
        print("I(A,X,Y) 1:", cmi123 - mi12)
        print("I(A,X,Y) 2:", cmi132 - mi13)
        print("I(A,X,Y) av:", I_av)

        print()
        print("Analysis (X=behavior1, Y=behavior2):")
        print(f"  Redundancy detected: {I_av < 0}")
        print(f"  MI(A,X) < |II|: {mi12 < np.abs(I_av)}")
        print(f"  MI(A,Y) < |II|: {mi13 < np.abs(I_av)}")

    if I_av < 0:  # Negative interaction information (redundancy)
        criterion1 = mi12 < np.abs(I_av) and not cmi132 < np.abs(I_av)
        criterion2 = mi13 < np.abs(I_av) and not cmi123 < np.abs(I_av)

        if criterion1 and not criterion2:
            return 1  # ts2 is redundant, ts3 is primary
        elif criterion2 and not criterion1:
            return 0  # ts3 is redundant, ts2 is primary
        else:
            return 0.5  # Both contribute - undistinguishable

    else:  # Positive interaction information (synergy)
        # Use epsilon tolerance for near-zero MI comparisons
        if mi13 < MI_EPSILON and cmi123 > cmi132:
            return 0  # ts2 is primary (ts3 has negligible MI)

        if mi12 < MI_EPSILON and cmi132 > cmi123:
            return 1  # ts3 is primary (ts2 has negligible MI)

        if mi13 >= MI_EPSILON and mi12 / mi13 > DOMINANCE_RATIO_THRESHOLD and cmi123 > cmi132:
            return 0  # ts2 is strongly dominant

        if mi12 >= MI_EPSILON and mi13 / mi12 > DOMINANCE_RATIO_THRESHOLD and cmi132 > cmi123:
            return 1  # ts3 is strongly dominant

        return 0.5  # Both contribute - undistinguishable


def disentangle_pair(ts1, ts2, ts3, verbose=False, ds=1):
    """Disentangle mixed selectivity between two behavioral variables for a neuron.

    Determines which of two correlated behavioral variables (ts2, ts3) provides
    the primary information about neural activity (ts1) using interaction information
    and conditional mutual information analysis.

    Parameters
    ----------
    ts1 : TimeSeries
        Neural activity time series (e.g., calcium signal or spike train).
    ts2 : TimeSeries
        First behavioral variable.
    ts3 : TimeSeries
        Second behavioral variable.
    verbose : bool, optional
        If True, print detailed analysis results. Default: False.
    ds : int, optional
        Downsampling factor. Default: 1.

    Returns
    -------
    float
        Disentanglement result:

        - 0: ts2 is the primary variable (ts3 is redundant)
        - 1: ts3 is the primary variable (ts2 is redundant)
        - 0.5: Both variables contribute - undistinguishable

    Notes
    -----
    The method uses interaction information to detect redundancy/synergy:

    - If II < 0 (redundancy), identifies the "weakest link" using criteria
      based on pairwise MI and conditional MI values
    - If II > 0 (synergy), uses different criteria for special cases

    See docs/intense_mathematical_framework.md for theoretical background."""
    # Delegate to internal function (no pre-computed values)
    return _disentangle_pair_with_precomputed(
        ts1, ts2, ts3, mi12=None, mi13=None, mi23=None, verbose=verbose, ds=ds
    )


def _process_neuron_disentanglement(
    neuron_id,
    sels,
    neur_ts,
    feat_names,
    multifeature_map,
    multifeature_ts,
    feature_ts_dict,
    ds,
    feat_feat_significance,
    cell_feat_stats,
    feat_feat_similarity,
    pre_decisions=None,
    pre_renames=None,
    feat_copnorm_cache=None,
):
    """Process disentanglement for a single neuron.

    Worker function for parallel disentanglement processing. Analyzes all
    feature pairs for a single neuron and returns partial result matrices.

    Parameters
    ----------
    neuron_id : any
        Neuron identifier.
    sels : list
        List of feature selectivities for this neuron (already filtered by pre-filter).
    neur_ts : TimeSeries
        Neural activity time series.
    feat_names : list
        List of all feature names.
    multifeature_map : dict
        Mapping from multifeature tuples to aggregated names.
    multifeature_ts : dict
        Pre-built MultiTimeSeries objects for multifeatures.
    feature_ts_dict : dict
        Pre-extracted feature TimeSeries, keyed by feature name.
    ds : int
        Downsampling factor.
    feat_feat_significance : ndarray or None
        Binary significance matrix for feature pairs.
    cell_feat_stats : dict or None
        Pre-computed neuron-feature MI values.
    feat_feat_similarity : ndarray or None
        Pre-computed feature-feature MI values.
    pre_decisions : dict or None, optional
        Pre-computed pair decisions from filter chain: {(feat_i, feat_j): 0/0.5/1}.
        0 = feat_i is primary, 1 = feat_j is primary, 0.5 = keep both.
        Default: None.
    pre_renames : dict or None, optional
        Pre-computed feature renames from filter chain: {new_name: (old1, old2)}.
        Default: None.
    feat_copnorm_cache : dict or None, optional
        Pre-computed downsampled copula-normalized data for features.
        Maps feature name to downsampled copula data. Pre-computed once
        and shared across all neurons to avoid redundant computation.
        Default: None.

    Returns
    -------
    neuron_id : any
        The neuron identifier (passed through for result aggregation).
    partial_disent : ndarray
        Partial disentanglement matrix for this neuron.
    partial_count : ndarray
        Partial count matrix for this neuron.
    neuron_info : dict
        Per-neuron details containing:
        - 'pairs': {(feat_i, feat_j): {'result': 0/0.5/1, 'source': str}}
        - 'renames': {new_name: (old1, old2)} from pre_renames
        - 'final_sels': list of final selectivities after filtering
        - 'errors': list of (neuron_id, sel_comb, error_msg) tuples for failed pairs
    """
    pre_decisions = pre_decisions or {}
    pre_renames = pre_renames or {}

    n_features = len(feat_names)
    partial_disent = np.zeros((n_features, n_features))
    partial_count = np.zeros((n_features, n_features))

    # Track per-neuron pair results and errors
    neuron_pairs_dict = {}
    errors = []

    # Build set of renamed feature names for skip logic
    renamed_features = set(pre_renames.keys())

    # Pre-cache downsampled copula-normalized data for faster CMI computation
    # Only for continuous time series (discrete use different code paths)
    neur_copnorm = _downsample_copnorm(neur_ts.copula_normal_data, ds) if not neur_ts.discrete else None

    # Use pre-computed feature copula cache (passed from caller)
    feat_copnorm_cache = feat_copnorm_cache or {}

    # Test all pairs of features this neuron responds to
    for sel_comb in combinations(sels, 2):
        try:
            sel_comb = list(sel_comb)

            # Skip pairs involving renamed combined features
            if sel_comb[0] in renamed_features or sel_comb[1] in renamed_features:
                continue

            feat_ts = []
            finds = []

            # Get time series for each feature
            for fname in sel_comb:
                # Check if this is a multifeature tuple
                if isinstance(fname, tuple) and fname in multifeature_map:
                    agg_name = multifeature_map[fname]
                    if agg_name in feat_names:
                        feat_ts.append(multifeature_ts[agg_name])
                        finds.append(feat_names.index(agg_name))
                    else:
                        raise ValueError(f"Aggregated name '{agg_name}' not in feat_names")
                else:
                    # Regular single feature - use pre-extracted dict
                    if fname in feature_ts_dict:
                        feat_ts.append(feature_ts_dict[fname])
                        finds.append(feat_names.index(fname))
                    else:
                        raise ValueError(f"Feature '{fname}' not found in experiment")

            # Get feature indices
            ind1 = finds[0]
            ind2 = finds[1]

            # Check if this pair has a pre-computed decision from filter chain
            pair_key = (sel_comb[0], sel_comb[1])
            reverse_key = (sel_comb[1], sel_comb[0])

            if pair_key in pre_decisions:
                disres = pre_decisions[pair_key]
                source = 'pre_filter'
            elif reverse_key in pre_decisions:
                # Flip the result for reversed pair (0↔1, 0.5 unchanged)
                disres = _flip_decision(pre_decisions[reverse_key])
                source = 'pre_filter'
            else:
                # Check if this feature pair has significant behavioral correlation
                if feat_feat_significance is not None:
                    if feat_feat_significance[ind1, ind2] == 0:
                        # Features are not significantly correlated
                        # Skip disentanglement - this is true mixed selectivity
                        disres = 0.5
                        source = 'not_significant'

                        partial_count[ind1, ind2] += 1
                        partial_count[ind2, ind1] += 1
                        partial_disent[ind1, ind2] += 0.5
                        partial_disent[ind2, ind1] += 0.5

                        # Record the pair result
                        neuron_pairs_dict[(feat_names[ind1], feat_names[ind2])] = {
                            'result': disres,
                            'source': source,
                        }
                        continue

                # Look up pre-computed MI values (if available)
                mi12 = _lookup_cell_feat_mi(cell_feat_stats, neuron_id, sel_comb[0])
                mi13 = _lookup_cell_feat_mi(cell_feat_stats, neuron_id, sel_comb[1])
                mi23 = _lookup_feat_feat_mi(
                    feat_feat_similarity, feat_names,
                    feat_names[ind1], feat_names[ind2]
                )

                # Perform disentanglement analysis only for significant pairs
                # Pass pre-computed copula data for faster CMI computation
                disres = _disentangle_pair_with_precomputed(
                    neur_ts, feat_ts[0], feat_ts[1],
                    mi12=mi12, mi13=mi13, mi23=mi23,
                    ts1_copnorm=neur_copnorm,
                    ts2_copnorm=feat_copnorm_cache.get(sel_comb[0]),
                    ts3_copnorm=feat_copnorm_cache.get(sel_comb[1]),
                    ds=ds, verbose=False
                )
                source = 'standard'

            # Validate disres value
            if disres not in VALID_DISRES_VALUES:
                raise ValueError(f"Unexpected disres value: {disres} (expected 0, 0.5, or 1)")

            # Update matrices
            partial_count[ind1, ind2] += 1
            partial_count[ind2, ind1] += 1

            if disres == 0:
                partial_disent[ind1, ind2] += 1  # Feature 1 is primary
            elif disres == 1:
                partial_disent[ind2, ind1] += 1  # Feature 2 is primary
            else:  # disres == 0.5 (validated above)
                partial_disent[ind1, ind2] += 0.5  # Both contribute
                partial_disent[ind2, ind1] += 0.5

            # Record the pair result
            neuron_pairs_dict[(feat_names[ind1], feat_names[ind2])] = {
                'result': disres,
                'source': source,
            }

        except (ValueError, AttributeError, KeyError) as e:
            # Accumulate errors for reporting (don't just print and lose them)
            errors.append((neuron_id, sel_comb, str(e)))
            continue

    # Compute final_sels by applying pair decisions
    # result=0: feat_i is primary (remove feat_j)
    # result=1: feat_j is primary (remove feat_i)
    # result=0.5: keep both
    features_to_remove = set()
    for (feat_i, feat_j), info in neuron_pairs_dict.items():
        result = info.get('result', 0.5)
        if result == 0:
            features_to_remove.add(feat_j)
        elif result == 1:
            features_to_remove.add(feat_i)

    final_sels = [f for f in sels if f not in features_to_remove]

    # Build neuron info
    neuron_info = {
        'pairs': neuron_pairs_dict,
        'renames': pre_renames,
        'final_sels': final_sels,
        'errors': errors,  # List of (neuron_id, sel_comb, error_msg) tuples
    }

    return neuron_id, partial_disent, partial_count, neuron_info


def disentangle_all_selectivities(
    exp,
    feat_names,
    ds=1,
    multifeature_map=None,
    feat_feat_significance=None,
    cell_bunch=None,
    cell_feat_stats=None,
    feat_feat_similarity=None,
    n_jobs=-1,
    pre_filter_func=None,
    post_filter_func=None,
    filter_kwargs=None,
):
    """Analyze mixed selectivity across all significant neuron-feature pairs.

    For each neuron that responds to multiple features, determines which
    features provide primary vs redundant information using disentanglement
    analysis. Only analyzes feature pairs that show significant correlation
    in the behavioral data.

    Parameters
    ----------
    exp : Experiment
        Experiment object containing neural and behavioral data.
    feat_names : list of str
        List of feature names to analyze. Should match features in experiment
        and any aggregated names from multifeature_map.
    ds : int, optional
        Downsampling factor. Default: 1.
    multifeature_map : dict, optional
        Mapping from multifeature tuples to aggregated names and their
        corresponding MultiTimeSeries. If None, uses DEFAULT_MULTIFEATURE_MAP.
        Example: ``{('x', 'y'): 'place', ('speed', 'head_direction'): 'locomotion'}``.
    feat_feat_significance : ndarray, optional
        Binary significance matrix from compute_feat_feat_significance.
        If provided, only feature pairs marked as significant (value=1)
        will be analyzed for disentanglement. Non-significant pairs are
        assumed to represent true mixed selectivity.
    cell_bunch : list or None, optional
        List of cell IDs to analyze. If None, analyzes all cells.
        Default: None.
    cell_feat_stats : dict or None, optional
        Pre-computed neuron-feature statistics from INTENSE analysis.
        Structure: stats[cell_id][feat_name]["me"] = MI value.
        If provided, MI(neuron, feature) values will be looked up instead
        of recomputed, significantly speeding up disentanglement.
        Default: None.
    feat_feat_similarity : ndarray or None, optional
        Pre-computed feature-feature similarity matrix from
        compute_feat_feat_significance. Matrix where [i, j] = MI(feat_i, feat_j).
        If provided, MI(feature1, feature2) values will be looked up.
        Default: None.
    n_jobs : int, optional
        Number of parallel jobs for processing neurons. -1 means use all
        available processors. Default: -1.
    pre_filter_func : callable or None, optional
        Population-level filter function (or composed filter) to run BEFORE
        the parallel processing loop. The filter mutates neuron selectivities
        and pre-computes pair decisions for all neurons at once.

        Signature::

            def pre_filter_func(
                neuron_selectivities,    # dict: {neuron_id: [feat1, feat2, ...]} - MUTATE
                pair_decisions,          # dict: {neuron_id: {(f1, f2): 0/0.5/1}} - MUTATE
                renames,                 # dict: {neuron_id: {new_name: (old1, old2)}} - MUTATE
                cell_feat_stats,         # Pre-computed MI values (READ ONLY)
                feat_feat_significance,  # Binary matrix (READ ONLY)
                feat_names,              # List of feature names (READ ONLY)
                **kwargs,                # User-provided extra arguments
            ):
                ...

        Default: None (no filtering).
    post_filter_func : callable or None, optional
        Population-level filter function to run AFTER parallel disentanglement.
        Can modify pair results (e.g., tie-breaking). Mutates per_neuron_disent
        and recalculates final_sels.

        Signature::

            def post_filter_func(
                per_neuron_disent,       # dict: {nid: {'pairs': {...}, ...}} - MUTATE
                cell_feat_stats,         # Pre-computed MI values (READ ONLY)
                feat_names,              # List of feature names (READ ONLY)
                **kwargs,                # User-provided extra arguments
            ):
                ...

        Default: None (no post-filtering).
    filter_kwargs : dict or None, optional
        Dictionary of keyword arguments to pass to pre_filter_func and post_filter_func.
        Can include pre-extracted data like calcium_data, feature_data,
        thresholds, etc. Default: None.

    Returns
    -------
    dict
        Dictionary containing:

        - 'disent_matrix': ndarray where element [i,j] indicates how many times
          feature i was primary when paired with feature j across all neurons.
        - 'count_matrix': ndarray where element [i,j] indicates how many
          neuron-feature pairs were tested for features i and j.
        - 'per_neuron_disent': dict mapping neuron_id to detailed results
          with keys 'pairs', 'renames', 'final_sels', and 'errors'.

    Notes
    -----
    The analysis is performed only on neurons with significant selectivity
    to at least 2 features. If feat_feat_significance is provided, only
    behaviorally correlated feature pairs are analyzed for redundancy.
    Non-significant pairs indicate true mixed selectivity.

    When cell_feat_stats and feat_feat_similarity are provided, 3 out of 5
    pairwise MI computations per disentangle_pair call are skipped by using
    lookups, providing ~60% reduction in MI computation overhead.

    The neuron loop is parallelized using joblib, providing significant speedup
    when analyzing many neurons. Each neuron is processed independently and
    results are merged at the end.

    **Filter chain execution:**

    1. Filters run at population level BEFORE the parallel loop
    2. Filters mutate neuron_selectivities, pair_decisions, and renames in place
    3. Workers receive pre-computed decisions (lightweight serialization)
    4. Each filter in the chain can override decisions from earlier filters

    Raises
    ------
    ValueError
        If a feature name is not found in the experiment or feat_names.
    AttributeError
        If required attributes are missing from the experiment.
    KeyError
        If expected keys are missing from data structures."""
    # Use default multifeature mapping if none provided
    if multifeature_map is None:
        multifeature_map = DEFAULT_MULTIFEATURE_MAP.copy()

    # Initialize result matrices
    n_features = len(feat_names)
    disent_matrix = np.zeros((n_features, n_features))
    count_matrix = np.zeros((n_features, n_features))

    # Create MultiTimeSeries for each multifeature
    multifeature_ts = {}
    for mf_tuple, agg_name in multifeature_map.items():
        if agg_name in feat_names:
            # Get individual TimeSeries for each component
            component_ts = []
            for component in mf_tuple:
                if hasattr(exp, component):
                    component_ts.append(getattr(exp, component))
                else:
                    raise ValueError(f"Component '{component}' not found in experiment")

            # Create MultiTimeSeries
            # Allow zero columns since behavioral features might be constant
            multifeature_ts[agg_name] = MultiTimeSeries(component_ts, allow_zero_columns=True)

    # Get neurons with significant selectivity to multiple features
    sneur = exp.get_significant_neurons(min_nspec=2, cbunch=cell_bunch)

    # Pre-extract feature TimeSeries to avoid serializing entire exp object
    # This is critical for parallel performance with joblib
    feature_ts_dict = {}
    for fname in feat_names:
        if hasattr(exp, fname):
            feature_ts_dict[fname] = getattr(exp, fname)

    # Pre-extract neuron TimeSeries
    neuron_ts_dict = {neuron: exp.neurons[neuron].ca for neuron in sneur.keys()}

    # ============================================================
    # PHASE 1: Build filter state and run filter chain (BEFORE parallel loop)
    # ============================================================
    # Build mutable state for filter chain
    neuron_selectivities = {nid: list(sels) for nid, sels in sneur.items()}
    pair_decisions = {nid: {} for nid in sneur}
    renames = {nid: {} for nid in sneur}

    # Run filter chain (population-level, BEFORE parallel loop)
    if pre_filter_func is not None:
        pre_filter_func(
            neuron_selectivities=neuron_selectivities,
            pair_decisions=pair_decisions,
            renames=renames,
            cell_feat_stats=cell_feat_stats,
            feat_feat_significance=feat_feat_significance,
            feat_names=feat_names,
            **(filter_kwargs or {}),
        )

    # ============================================================
    # PHASE 2: Parallel processing (WORKERS)
    # ============================================================
    per_neuron_disent = {}

    # Pre-compute feature copula cache ONCE (shared across all neurons)
    # This avoids redundant copula normalization in each worker
    feat_copnorm_cache = {}
    for fname in feat_names:
        ts = feature_ts_dict.get(fname) or multifeature_ts.get(fname)
        if ts is not None and not getattr(ts, 'discrete', True):
            feat_copnorm_cache[fname] = _downsample_copnorm(ts.copula_normal_data, ds)

    # Process neurons in parallel with backend-specific config
    if len(neuron_selectivities) > 0:
        with _parallel_executor(n_jobs) as parallel:
            results = parallel(
                delayed(_process_neuron_disentanglement)(
                    neuron_id=neuron,
                    sels=neuron_selectivities[neuron],  # Already filtered
                    neur_ts=neuron_ts_dict[neuron],
                    feat_names=feat_names,
                    multifeature_map=multifeature_map,
                    multifeature_ts=multifeature_ts,
                    feature_ts_dict=feature_ts_dict,
                    ds=ds,
                    feat_feat_significance=feat_feat_significance,
                    cell_feat_stats=cell_feat_stats,
                    feat_feat_similarity=feat_feat_similarity,
                    pre_decisions=pair_decisions[neuron],  # Pre-computed
                    pre_renames=renames[neuron],           # Pre-computed
                    feat_copnorm_cache=feat_copnorm_cache, # Pre-computed
                )
                for neuron in neuron_selectivities.keys()
            )

        # Merge partial results from all workers
        for neuron_id, partial_disent, partial_count, neuron_info in results:
            disent_matrix += partial_disent
            count_matrix += partial_count
            # Store per-neuron info if it has any content (including errors)
            if neuron_info['pairs'] or neuron_info['renames'] or neuron_info['errors']:
                per_neuron_disent[neuron_id] = neuron_info

    # ============================================================
    # PHASE 3: Post-filter (AFTER parallel loop)
    # ============================================================
    if post_filter_func is not None:
        post_filter_func(
            per_neuron_disent=per_neuron_disent,
            cell_feat_stats=cell_feat_stats,
            feat_names=feat_names,
            **(filter_kwargs or {}),
        )

        # Recalculate final_sels for neurons modified by post-filter
        for nid, neuron_info in per_neuron_disent.items():
            features_to_remove = set()
            for (feat_i, feat_j), info in neuron_info['pairs'].items():
                result = info.get('result', 0.5)
                if result == 0:
                    features_to_remove.add(feat_j)
                elif result == 1:
                    features_to_remove.add(feat_i)

            original_sels = neuron_selectivities[nid]
            neuron_info['final_sels'] = [f for f in original_sels if f not in features_to_remove]

    return {
        'disent_matrix': disent_matrix,
        'count_matrix': count_matrix,
        'per_neuron_disent': per_neuron_disent,
    }


def create_multifeature_map(exp, mapping_dict):
    """Create a multifeature mapping with validation.

    Parameters
    ----------
    exp : Experiment
        Experiment object to validate feature existence.
    mapping_dict : dict
        Dictionary mapping tuples of features to aggregated names.
        Example: {('x', 'y'): 'place', ('speed', 'head_direction'): 'locomotion'}

    Returns
    -------
    dict
        Validated multifeature mapping.

    Raises
    ------
    ValueError
        If any component features don't exist in the experiment."""
    validated_map = {}

    for mf_tuple, agg_name in mapping_dict.items():
        # Validate that all components exist
        for component in mf_tuple:
            if not hasattr(exp, component):
                raise ValueError(
                    f"Component '{component}' in multifeature {mf_tuple} "
                    f"not found in experiment"
                )

        # Ensure tuple is sorted for consistency
        sorted_tuple = tuple(sorted(mf_tuple))
        validated_map[sorted_tuple] = agg_name

    return validated_map


def get_disentanglement_summary(
    disent_matrix, count_matrix, feat_names, feat_feat_significance=None
):
    """Generate a summary of disentanglement results.

    Parameters
    ----------
    disent_matrix : ndarray
        Disentanglement result matrix from disentangle_all_selectivities.
    count_matrix : ndarray
        Count matrix from disentangle_all_selectivities.
    feat_names : list of str
        Feature names corresponding to matrix indices.
    feat_feat_significance : ndarray, optional
        Binary significance matrix indicating which feature pairs
        were analyzed for disentanglement.

    Returns
    -------
    dict
        Summary statistics including:
        - Primary feature percentages for each pair
        - Total counts for each pair
        - Overall redundancy vs independence rates
        - Breakdown by significant vs non-significant feature pairs

    Notes
    -----
    The calculation distinguishes between:
    - Redundant cases: One feature is primary (disentangle result 0 or 1)
    - Undistinguishable cases: Both features contribute (disentangle result 0.5)

    Undistinguishable cases are identified by fractional values in the
    disentanglement matrix, as each such case contributes 0.5 to both features."""
    summary = {"feature_pairs": {}, "overall_stats": {}}

    n_features = len(feat_names)
    total_redundant = 0
    total_undistinguishable = 0
    total_pairs = 0

    for i in range(n_features):
        for j in range(i + 1, n_features):
            if count_matrix[i, j] > 0:
                n_total = count_matrix[i, j]
                n_i_primary = disent_matrix[i, j]
                n_j_primary = disent_matrix[j, i]

                # Calculate undistinguishable cases
                # When feat_feat_significance indicates features are not correlated (sig=0),
                # disentanglement is skipped and all neurons get 0.5 (undistinguishable)
                if feat_feat_significance is not None and feat_feat_significance[i, j] == 0:
                    # Non-significant feature pair: all neurons are undistinguishable
                    # (true mixed selectivity - features are independent)
                    n_undistinguishable = int(n_total)
                else:
                    # Significant feature pair: calculate from matrix values
                    # Each undistinguishable case contributes 0.5 to both matrices
                    # Use: n_undist = n_total - (i_wins + j_wins)
                    # where i_wins = int(n_i_primary), j_wins = int(n_j_primary)
                    # when n_undist is even (no fractional part)
                    frac_i = n_i_primary - int(n_i_primary)
                    if frac_i > 0.25:  # Has fractional part (odd n_undist)
                        n_undistinguishable = round(frac_i * 2)
                    else:  # No fractional part (even n_undist, including 0)
                        # i_wins + j_wins + n_undist = n_total
                        # i_wins = n_i_primary (when frac=0), j_wins = n_j_primary
                        n_undistinguishable = int(n_total - int(n_i_primary) - int(n_j_primary))
                n_redundant = int(n_total) - n_undistinguishable

                pair_key = f"{feat_names[i]}_vs_{feat_names[j]}"
                summary["feature_pairs"][pair_key] = {
                    "total_neurons": int(n_total),
                    f"{feat_names[i]}_primary": n_i_primary / n_total * 100,
                    f"{feat_names[j]}_primary": n_j_primary / n_total * 100,
                    "undistinguishable_pct": n_undistinguishable / n_total * 100,
                    "redundant_pct": n_redundant / n_total * 100,
                }

                total_redundant += n_redundant
                total_undistinguishable += n_undistinguishable
                total_pairs += n_total

    if total_pairs > 0:
        summary["overall_stats"] = {
            "total_neuron_pairs": int(total_pairs),
            "redundancy_rate": total_redundant / total_pairs * 100,
            "undistinguishable_rate": total_undistinguishable / total_pairs * 100,
        }

        # Add breakdown by behavioral significance if provided
        if feat_feat_significance is not None:
            sig_pairs = 0
            nonsig_pairs = 0
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    if count_matrix[i, j] > 0:
                        if feat_feat_significance[i, j] == 1:
                            sig_pairs += count_matrix[i, j]
                        else:
                            nonsig_pairs += count_matrix[i, j]

            summary["overall_stats"]["significant_behavior_pairs"] = int(sig_pairs)
            summary["overall_stats"]["nonsignificant_behavior_pairs"] = int(nonsig_pairs)
            summary["overall_stats"]["true_mixed_selectivity_rate"] = (
                nonsig_pairs / total_pairs * 100 if total_pairs > 0 else 0
            )

    return summary
