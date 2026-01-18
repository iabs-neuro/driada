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


# Default multifeature mapping for common behavioral variable combinations
# Maps component tuples to their semantic names
DEFAULT_MULTIFEATURE_MAP = {
    ("x", "y"): "place",  # spatial location multifeature
}


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
    verbose=False,
    ds=1,
):
    """Disentangle with optional pre-computed MI values.

    Internal function that performs disentanglement analysis, optionally
    using pre-computed pairwise MI values to avoid redundant computation.

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

    # Conditional MI must always be computed (not cached)
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
        if mi13 == 0 and cmi123 > cmi132:
            return 0  # ts2 is primary

        if mi12 == 0 and cmi132 > cmi123:
            return 1  # ts3 is primary

        if mi13 > 0 and mi12 / mi13 > 2.0 and cmi123 > cmi132:
            return 0  # ts2 is strongly dominant

        if mi12 > 0 and mi13 / mi12 > 2.0 and cmi132 > cmi123:
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

    See README_INTENSE.md for theoretical background."""
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
):
    """Process disentanglement for a single neuron.

    Worker function for parallel disentanglement processing. Analyzes all
    feature pairs for a single neuron and returns partial result matrices.

    Parameters
    ----------
    neuron_id : any
        Neuron identifier.
    sels : list
        List of feature selectivities for this neuron.
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

    Returns
    -------
    partial_disent : ndarray
        Partial disentanglement matrix for this neuron.
    partial_count : ndarray
        Partial count matrix for this neuron.
    """
    n_features = len(feat_names)
    partial_disent = np.zeros((n_features, n_features))
    partial_count = np.zeros((n_features, n_features))

    # Test all pairs of features this neuron responds to
    for sel_comb in combinations(sels, 2):
        try:
            sel_comb = list(sel_comb)
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

            # Check if this feature pair has significant behavioral correlation
            if feat_feat_significance is not None:
                if feat_feat_significance[ind1, ind2] == 0:
                    # Features are not significantly correlated
                    # Skip disentanglement - this is true mixed selectivity
                    partial_count[ind1, ind2] += 1
                    partial_count[ind2, ind1] += 1
                    # Add 0.5 to each to indicate undistinguishable contributions
                    partial_disent[ind1, ind2] += 0.5
                    partial_disent[ind2, ind1] += 0.5
                    continue

            # Look up pre-computed MI values (if available)
            mi12 = _lookup_cell_feat_mi(cell_feat_stats, neuron_id, sel_comb[0])
            mi13 = _lookup_cell_feat_mi(cell_feat_stats, neuron_id, sel_comb[1])
            mi23 = _lookup_feat_feat_mi(
                feat_feat_similarity, feat_names,
                feat_names[ind1], feat_names[ind2]
            )

            # Perform disentanglement analysis only for significant pairs
            disres = _disentangle_pair_with_precomputed(
                neur_ts, feat_ts[0], feat_ts[1],
                mi12=mi12, mi13=mi13, mi23=mi23,
                ds=ds, verbose=False
            )

            # Update matrices
            partial_count[ind1, ind2] += 1
            partial_count[ind2, ind1] += 1

            if disres == 0:
                partial_disent[ind1, ind2] += 1  # Feature 1 is primary
            elif disres == 1:
                partial_disent[ind2, ind1] += 1  # Feature 2 is primary
            elif disres == 0.5:
                partial_disent[ind1, ind2] += 0.5  # Both contribute
                partial_disent[ind2, ind1] += 0.5

        except (ValueError, AttributeError, KeyError) as e:
            # Log specific errors but continue processing other pairs
            print(f"WARNING: Skipping neuron {neuron_id}, features {sel_comb}: {str(e)}")
            continue

    return partial_disent, partial_count


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
        Example: {
            ('x', 'y'): 'place',
            ('speed', 'head_direction'): 'locomotion',
            ('lick', 'reward'): 'consummatory'
        }
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

    Returns
    -------
    disent_matrix : ndarray
        Matrix where element [i,j] indicates how many times feature i
        was primary when paired with feature j across all neurons.
    count_matrix : ndarray
        Matrix where element [i,j] indicates how many neuron-feature
        pairs were tested for features i and j.

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

    # Process neurons in parallel
    if len(sneur) > 0:
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_process_neuron_disentanglement)(
                neuron_id=neuron,
                sels=sels,
                neur_ts=neuron_ts_dict[neuron],
                feat_names=feat_names,
                multifeature_map=multifeature_map,
                multifeature_ts=multifeature_ts,
                feature_ts_dict=feature_ts_dict,
                ds=ds,
                feat_feat_significance=feat_feat_significance,
                cell_feat_stats=cell_feat_stats,
                feat_feat_similarity=feat_feat_similarity,
            )
            for neuron, sels in sneur.items()
        )

        # Merge partial results from all workers
        for partial_disent, partial_count in results:
            disent_matrix += partial_disent
            count_matrix += partial_count

    return disent_matrix, count_matrix


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

                # Calculate undistinguishable cases from fractional parts
                # Each undistinguishable case contributes 0.5 to both matrices
                # So fractional part * 2 gives the number of such cases
                frac_i = n_i_primary - int(n_i_primary)
                frac_j = n_j_primary - int(n_j_primary)

                # Fractional parts should match (both get 0.5 from each undistinguishable)
                # Use minimum to handle floating point precision
                n_undistinguishable = round(min(frac_i, frac_j) * 2)
                n_redundant = n_total - n_undistinguishable

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
