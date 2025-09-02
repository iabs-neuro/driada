"""
Mixed selectivity disentanglement analysis for INTENSE.

This module provides functions to analyze and disentangle mixed selectivity
in neural responses when neurons respond to multiple, potentially correlated
behavioral variables.
"""

import numpy as np
from itertools import combinations
from ..information.info_base import get_mi, conditional_mi, MultiTimeSeries


# Default multifeature mapping for common behavioral variable combinations
# Maps component tuples to their semantic names
DEFAULT_MULTIFEATURE_MAP = {
    ("x", "y"): "place",  # spatial location multifeature
}


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

    See README_INTENSE.md for theoretical background.    """
    # Compute pairwise mutual information
    mi12 = get_mi(ts1, ts2, ds=ds)  # MI(neuron, behavior1)
    mi13 = get_mi(ts1, ts3, ds=ds)  # MI(neuron, behavior2)
    mi23 = get_mi(ts2, ts3, ds=ds)  # MI(behavior1, behavior2)

    # Compute conditional mutual information
    cmi123 = conditional_mi(ts1, ts2, ts3, ds=ds)  # MI(neuron, behavior1 | behavior2)
    cmi132 = conditional_mi(ts1, ts3, ts2, ds=ds)  # MI(neuron, behavior2 | behavior1)

    # Compute interaction information (average of two equivalent formulas)
    # Using Williams & Beer convention: II = I(X;Y|Z) - I(X;Y)
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
        # Check if either variable is a "weak link"
        criterion1 = mi12 < np.abs(I_av) and not cmi132 < np.abs(I_av)
        criterion2 = mi13 < np.abs(I_av) and not cmi123 < np.abs(I_av)

        if criterion1 and not criterion2:
            return 1  # ts2 is redundant, ts3 is primary
        elif criterion2 and not criterion1:
            return 0  # ts3 is redundant, ts2 is primary
        else:
            return 0.5  # Both contribute - undistinguishable

    else:  # Positive interaction information (synergy)
        # Special cases for synergistic relationships
        if mi13 == 0 and cmi123 > cmi132:
            return 0  # ts2 is primary

        if mi12 == 0 and cmi132 > cmi123:
            return 1  # ts3 is primary

        if mi13 > 0 and mi12 / mi13 > 2.0 and cmi123 > cmi132:
            return 0  # ts2 is strongly dominant

        if mi12 > 0 and mi13 / mi12 > 2.0 and cmi132 > cmi123:
            return 1  # ts3 is strongly dominant

        return 0.5  # Both contribute - undistinguishable


def disentangle_all_selectivities(
    exp,
    feat_names,
    ds=1,
    multifeature_map=None,
    feat_feat_significance=None,
    cell_bunch=None,
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
    
    Raises
    ------
    ValueError
        If a feature name is not found in the experiment or feat_names.
    AttributeError
        If required attributes are missing from the experiment.
    KeyError
        If expected keys are missing from data structures.    """
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

    for neuron, sels in sneur.items():
        neur_ts = exp.neurons[neuron].ca

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
                            raise ValueError(
                                f"Aggregated name '{agg_name}' not in feat_names"
                            )
                    else:
                        # Regular single feature
                        if hasattr(exp, fname):
                            feat_ts.append(getattr(exp, fname))
                            finds.append(feat_names.index(fname))
                        else:
                            raise ValueError(
                                f"Feature '{fname}' not found in experiment"
                            )

                # Get feature indices
                ind1 = finds[0]
                ind2 = finds[1]

                # Check if this feature pair has significant behavioral correlation
                if feat_feat_significance is not None:
                    if feat_feat_significance[ind1, ind2] == 0:
                        # Features are not significantly correlated
                        # Skip disentanglement - this is true mixed selectivity
                        count_matrix[ind1, ind2] += 1
                        count_matrix[ind2, ind1] += 1
                        # Add 0.5 to each to indicate undistinguishable contributions
                        disent_matrix[ind1, ind2] += 0.5
                        disent_matrix[ind2, ind1] += 0.5
                        continue

                # Perform disentanglement analysis only for significant pairs
                disres = disentangle_pair(
                    neur_ts, feat_ts[0], feat_ts[1], ds=ds, verbose=False
                )

                # Update matrices
                count_matrix[ind1, ind2] += 1
                count_matrix[ind2, ind1] += 1

                if disres == 0:
                    disent_matrix[ind1, ind2] += 1  # Feature 1 is primary
                elif disres == 1:
                    disent_matrix[ind2, ind1] += 1  # Feature 2 is primary
                elif disres == 0.5:
                    disent_matrix[ind1, ind2] += 0.5  # Both contribute
                    disent_matrix[ind2, ind1] += 0.5

            except (ValueError, AttributeError, KeyError) as e:
                # Log specific errors but continue processing other pairs
                print(
                    f"WARNING: Skipping neuron {neuron}, features {sel_comb}: {str(e)}"
                )
                continue

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
        If any component features don't exist in the experiment.    """
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
    disentanglement matrix, as each such case contributes 0.5 to both features.    """
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
            summary["overall_stats"]["nonsignificant_behavior_pairs"] = int(
                nonsig_pairs
            )
            summary["overall_stats"]["true_mixed_selectivity_rate"] = (
                nonsig_pairs / total_pairs * 100 if total_pairs > 0 else 0
            )

    return summary
