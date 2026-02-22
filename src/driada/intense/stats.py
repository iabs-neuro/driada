import warnings

import numpy as np
import scipy
from scipy.stats import lognorm, gamma, norm
from ..utils.data import populate_nested_dict

# Default distribution type for p-value calculation from shuffled MI values
DEFAULT_METRIC_DISTR_TYPE = "gamma_zi"


def chebyshev_ineq(data, val):
    """
    Calculate upper bound on tail probability using Chebyshev's inequality.

    Parameters
    ----------
    data : array-like
        Sample data to estimate mean and std from. Must have non-zero variance.
    val : float
        Value to compute tail probability for.

    Returns
    -------
    p_bound : float
        Upper bound on P(X >= val) based on Chebyshev's inequality.

    Raises
    ------
    ValueError
        If data has zero variance (all values are identical).

    Notes
    -----
    Chebyshev's inequality states that P(\|X - μ\| >= k*σ) <= 1/k²
    This gives P(X >= val) <= 1/z² where z = (val - μ)/σ"""
    mean = np.mean(data)
    std = np.std(data)

    if std == 0:
        raise ValueError("Cannot apply Chebyshev's inequality to data with zero variance")

    z = (val - mean) / std
    return 1.0 / z**2


def get_lognormal_p(data, val):
    """
    Calculate p-value assuming log-normal distribution.

    Parameters
    ----------
    data : array-like
        Sample data to fit log-normal distribution.
        Must contain positive values.
    val : float
        Observed value to compute p-value for.

    Returns
    -------
    p_value : float
        P(X >= val) under fitted log-normal distribution.

    Raises
    ------
    ValueError
        If data contains non-positive values.

    Notes
    -----
    Fits log-normal distribution with floc=0 (zero lower bound).
    Log-normal distribution is suitable for positive-valued data."""
    data = np.asarray(data)
    if np.any(data <= 0):
        raise ValueError("Data must contain only positive values for log-normal distribution")

    params = lognorm.fit(data, floc=0)
    rv = lognorm(*params)
    return rv.sf(val)


def get_gamma_p(data, val):
    """
    Calculate p-value assuming gamma distribution.

    Parameters
    ----------
    data : array-like
        Sample data to fit gamma distribution.
        Must contain positive values.
    val : float
        Observed value to compute p-value for.

    Returns
    -------
    p_value : float
        P(X >= val) under fitted gamma distribution.

    Raises
    ------
    ValueError
        If data contains non-positive values.

    Notes
    -----
    Fits gamma distribution with floc=0 (zero lower bound).
    Gamma distribution is suitable for positive-valued data."""
    data = np.asarray(data)
    if np.any(data <= 0):
        raise ValueError("Data must contain only positive values for gamma distribution")

    params = gamma.fit(data, floc=0)
    rv = gamma(*params)
    return rv.sf(val)


def get_gamma_zi_p(data, val, zero_threshold=1e-10):
    """
    Calculate p-value using zero-inflated gamma (ZIG) distribution.

    Parameters
    ----------
    data : array-like
        Sample data to fit ZIG distribution (typically shuffled MI values).
        Can contain zeros and positive values.
    val : float
        Observed value to compute p-value for.
    zero_threshold : float, optional
        Values below this threshold are considered zeros. Default: 1e-10.

    Returns
    -------
    p_value : float
        P(X >= val) under fitted ZIG distribution.

    Notes
    -----
    Zero-Inflated Gamma model:

    - P(X = 0) = pi (zero inflation parameter)
    - P(X > 0) = (1-pi) * Gamma(shape, scale)

    P-value calculation:

    - If val <= zero_threshold: p_value = 1.0 (zeros never significant)
    - If val > zero_threshold: p_value = (1-pi) * Gamma.sf(val)
      (zero mass doesn't contribute to tail probability for positive values)

    Edge cases:

    - All zeros (pi=1): Returns 1.0 for any val
    - No zeros (pi=0): Equivalent to pure gamma distribution
    - Gamma fit fails: Returns conservative 1.0
    """
    data = np.asarray(data)

    # Handle observed value at or near zero
    if val <= zero_threshold:
        return 1.0

    # Estimate zero-inflation parameter pi
    n_zeros = np.sum(data <= zero_threshold)
    n_total = len(data)
    pi = n_zeros / n_total

    # Edge case: All zeros
    if pi == 1.0:
        return 1.0

    # Fit gamma to non-zero values
    non_zero_data = data[data > zero_threshold]

    # Edge case: No non-zero values
    if len(non_zero_data) == 0:
        return 1.0

    try:
        # Fit gamma with fixed location at 0
        params = gamma.fit(non_zero_data, floc=0)
        rv = gamma(*params)

        # ZIG p-value: P(X >= val) = (1-pi) * P_gamma(X >= val)
        # The zero mass doesn't contribute since 0 < val
        gamma_sf = rv.sf(val)
        p_value = (1 - pi) * gamma_sf

        # Ensure p-value is in valid range [0, 1]
        return np.clip(p_value, 0.0, 1.0)

    except Exception as e:
        warnings.warn(
            f"Gamma-ZI fit failed ({e}), returning conservative p=1.0",
            RuntimeWarning,
            stacklevel=2,
        )
        return 1.0


def get_distribution_function(dist_name):
    """
    Get distribution function from scipy.stats by name.

    Parameters
    ----------
    dist_name : str
        Name of distribution (e.g., 'gamma', 'lognorm', 'norm').

    Returns
    -------
    dist : scipy.stats distribution
        Distribution function object.

    Raises
    ------
    ValueError
        If distribution name not found in scipy.stats."""
    try:
        return getattr(scipy.stats, dist_name)
    except AttributeError:
        raise ValueError(f"Distribution '{dist_name}' not found in scipy.stats")


def get_mi_distr_pvalue(data, val, distr_type="gamma"):
    """
    Calculate p-value by fitting a distribution to data.

    Parameters
    ----------
    data : array-like
        Sample data (typically shuffled metric values).
    val : float
        Observed value to compute p-value for.
    distr_type : str, optional
        Distribution type to fit. Options:
        - 'gamma': Gamma distribution (requires positive values)
        - 'gamma_zi': Zero-inflated gamma (handles zeros explicitly)
        - 'lognorm': Log-normal distribution
        - Any scipy.stats distribution name
        Default: 'gamma'.

    Returns
    -------
    p_value : float
        P(X >= val) under fitted distribution.
        Returns 1.0 if distribution fitting fails.

    Notes
    -----
    - For 'gamma_zi', uses zero-inflated gamma model (handles zeros)
    - For 'gamma' and 'lognorm', fits with floc=0 (zero lower bound)
    - For other distributions, uses default fitting
    - Returns conservative p-value (1.0) on fitting errors"""
    # Special handling for zero-inflated gamma
    if distr_type == "gamma_zi":
        return get_gamma_zi_p(data, val)

    # Original logic for other distributions
    distr = get_distribution_function(distr_type)
    try:
        if distr_type in ["gamma", "lognorm"]:
            params = distr.fit(data, floc=0)
        else:
            params = distr.fit(data)

        rv = distr(*params)
        return rv.sf(val)
    except Exception:  # Distribution fitting can fail for various reasons
        # Return conservative p-value on error
        return 1.0


def reconstruct_stage1_pvals(me_total1, metric_distr_type="gamma"):
    """Reconstruct Stage 1 p-values from saved shuffle distributions.

    Stage 1 skips p-value computation for performance (pre_pval=None).
    This function reconstructs them post-hoc from the saved me_total1 array
    using the same distribution fitting as Stage 2.

    Parameters
    ----------
    me_total1 : np.ndarray, shape (n1, n2, nsh1+1)
        Stage 1 MI array. [:,:,0] = true MI, [:,:,1:] = shuffles.
    metric_distr_type : str, optional
        Distribution type for p-value fitting. Default: 'gamma'.

    Returns
    -------
    pre_pvals : np.ndarray, shape (n1, n2)
        Reconstructed p-values. NaN where MI data is missing/zero.
    mi_values : np.ndarray, shape (n1, n2)
        True MI values (me_total1[:,:,0]).
    """
    n1, n2 = me_total1.shape[0], me_total1.shape[1]
    pre_pvals = np.full((n1, n2), np.nan)
    mi_values = me_total1[:, :, 0].copy()

    for i in range(n1):
        for j in range(n2):
            row = me_total1[i, j, :]
            if np.all(row == 0):
                continue
            me = row[0]
            random_mi_samples = row[1:]
            pre_pvals[i, j] = get_mi_distr_pvalue(
                random_mi_samples, me, distr_type=metric_distr_type
            )

    return pre_pvals, mi_values


def get_mask(ptable, rtable, pval_thr, rank_thr):
    """
    Create binary mask based on p-value and rank thresholds.

    Parameters
    ----------
    ptable : np.ndarray
        Array of p-values.
    rtable : np.ndarray
        Array of ranks (0 to 1).
    pval_thr : float
        P-value threshold. Values <= pval_thr pass.
    rank_thr : float
        Rank threshold. Values >= rank_thr pass.

    Returns
    -------
    mask : np.ndarray
        Binary mask: 1 where both conditions satisfied
        (p <= pval_thr AND rank >= rank_thr), 0 otherwise."""
    mask = np.ones(ptable.shape)
    mask[np.where(ptable > pval_thr)] = 0
    mask[np.where(rtable < rank_thr)] = 0
    return mask


def stats_not_empty(pair_stats, current_data_hash, stage=1):
    """
    Check if statistics are valid and complete for given stage.

    Parameters
    ----------
    pair_stats : dict
        Dictionary of computed statistics.
    current_data_hash : str
        Hash of current data to validate against.
    stage : int, optional
        Stage to check (1 or 2). Default: 1.

    Returns
    -------
    is_valid : bool
        True if stats are valid and complete, False otherwise.

    Raises
    ------
    ValueError
        If stage is not 1 or 2."""
    if stage == 1:
        stats_to_check = ["pre_rval", "pre_pval"]
    elif stage == 2:
        stats_to_check = ["rval", "pval", "me"]
    else:
        raise ValueError(f"Stage should be 1 or 2, but {stage} was passed")

    data_hash_from_stats = pair_stats["data_hash"]
    is_valid = current_data_hash == data_hash_from_stats
    is_not_empty = np.all(np.array([pair_stats[st] is not None for st in stats_to_check]))
    return is_valid and is_not_empty


def criterion1(pair_stats, nsh1, topk=1):
    """
    Check if pair passes stage 1 significance criterion.

    Parameters
    ----------
    pair_stats : dict
        Dictionary containing 'pre_rval' from stage 1 analysis.
    nsh1 : int
        Number of shuffles for first stage.
    topk : int, optional
        True MI should rank in top k among shuffles. Default: 1.

    Returns
    -------
    crit_passed : bool
        True if pair's rank exceeds threshold (1 - topk/(nsh1+1)).

    Notes
    -----
    The criterion checks if: pre_rval > (1 - topk/(nsh1+1))
    For topk=1 and nsh1=100, this requires pre_rval > 0.99"""

    if pair_stats.get("pre_rval") is not None:
        return pair_stats["pre_rval"] > (1 - 1.0 * topk / (nsh1 + 1))
        # return pair_stats['pre_rval'] == 1 # true MI should be top-1 among all shuffles
    else:
        return False


def criterion2(pair_stats, nsh2, pval_thr, topk=5):
    """
    Check if pair passes stage 2 significance criterion.

    Parameters
    ----------
    pair_stats : dict
        Dictionary containing 'rval' and 'pval' from stage 2 analysis.
    nsh2 : int
        Number of shuffles for second stage.
    pval_thr : float
        P-value threshold after multiple hypothesis correction.
    topk : int, optional
        True MI should rank in top k among shuffles. Default: 5.

    Returns
    -------
    crit_passed : bool
        True if both conditions met:
        1) rval > (1 - topk/(nsh2+1))
        2) pval < pval_thr

    Notes
    -----
    Both rank and p-value criteria must be satisfied.
    Missing 'rval' or 'pval' results in False."""
    # whether pair passed stage 1 and has statistics from stage 2
    if pair_stats.get("rval") is not None and pair_stats.get("pval") is not None:
        # whether true MI is among topk shuffles (in practice it is top-1 almost always)
        if pair_stats["rval"] > (1 - 1.0 * topk / (nsh2 + 1)):
            criterion = pair_stats["pval"] < pval_thr
            return criterion
        else:
            return False
    else:
        return False


def apply_stage_criterion(
    stage_stats: dict,
    stage_num: int,
    n1: int,
    n2: int,
    n_shuffles: int,
    topk: int,
    multicorr_thr: float = None,
) -> tuple:
    """Apply stage-appropriate significance criterion to all pairs.

    Thin wrapper that delegates to criterion1 (Stage 1) or criterion2 (Stage 2)
    based on stage_num. Enables unified scan_stage() function.

    Parameters
    ----------
    stage_stats : dict
        Nested dictionary with statistics from get_table_of_stats.
    stage_num : int
        Stage number (1 or 2).
    n1 : int
        Number of items in first dimension (neurons).
    n2 : int
        Number of items in second dimension (features).
    n_shuffles : int
        Number of shuffles used in this stage.
    topk : int
        True MI should rank in top k among shuffles.
    multicorr_thr : float, optional
        Multiple comparison corrected p-value threshold.
        Required for stage_num=2, ignored for stage_num=1.

    Returns
    -------
    significance : dict
        Nested dict with boolean significance for each pair.
        Keys are stage-specific: 'stage1' or 'stage2'.
    pass_mask : np.ndarray
        Binary mask of shape (n1, n2) with 1 for pairs that passed.
    """
    pass_mask = np.zeros((n1, n2))
    significance = populate_nested_dict(dict(), range(n1), range(n2))

    if stage_num == 1:
        sig_key = "stage1"
        for i in range(n1):
            for j in range(n2):
                passed = criterion1(stage_stats[i][j], n_shuffles, topk=topk)
                if passed:
                    pass_mask[i, j] = 1
                significance[i][j] = {sig_key: passed}
    else:
        # Stage 2 - requires multicorr_thr
        if multicorr_thr is None:
            raise ValueError("multicorr_thr required for stage 2")
        sig_key = "stage2"
        for i in range(n1):
            for j in range(n2):
                passed = criterion2(
                    stage_stats[i][j], n_shuffles, multicorr_thr, topk=topk
                )
                if passed:
                    pass_mask[i, j] = 1
                significance[i][j] = {sig_key: passed}

    return significance, pass_mask


def get_all_nonempty_pvals(all_stats, ids1, ids2) -> list:
    """
    Extract all non-empty p-values from nested statistics dictionary.

    Parameters
    ----------
    all_stats : dict of dict
        Nested dictionary with statistics.
    ids1 : list
        First dimension indices.
    ids2 : list
        Second dimension indices.

    Returns
    -------
    all_pvals : list
        List of all non-None p-values found."""
    all_pvals = []
    for i, id1 in enumerate(ids1):
        for j, id2 in enumerate(ids2):
            pval = all_stats[id1][id2].get("pval")
            if pval is not None:
                all_pvals.append(pval)

    return all_pvals


def get_table_of_stats(
    metable,
    optimal_delays,
    precomputed_mask=None,
    metric_distr_type=DEFAULT_METRIC_DISTR_TYPE,
    nsh=0,
    stage=1,
):
    """
    Convert metric table to statistics dictionary.

    Parameters
    ----------
    metable : np.ndarray of shape (n1, n2, nsh+1)
        Metric values where [:,:,0] is true values, [:,:,1:] are shuffles.
    optimal_delays : np.ndarray of shape (n1, n2)
        Optimal delays for each pair.
    precomputed_mask : np.ndarray, optional
        Binary mask: 1 = compute stats, 0 = skip. Default: all ones.
    metric_distr_type : str, optional
        Distribution for p-value calculation. Default: 'gamma'.
    nsh : int, optional
        Number of shuffles. Default: 0.
    stage : int, optional
        Stage (1 or 2) determines which stats to compute. Default: 1.

    Returns
    -------
    stage_stats : dict of dict
        Nested dictionary with computed statistics for each pair."""
    # 0 in mask values means that stats for this pair will not be calculated
    # 1 in mask values means that stats for this pair will be calculated from new results.
    if precomputed_mask is None:
        precomputed_mask = np.ones(metable.shape[:2])

    a, b, sh = metable.shape
    stage_stats = populate_nested_dict(dict(), range(a), range(b))

    # Count shuffles strictly below true MI — O(n1*n2*nsh) vectorized comparison.
    # For no-tie data: identical to rankdata. For ties: more conservative (safe direction).
    count_below = (metable[:, :, 1:] < metable[:, :, 0:1]).sum(axis=2)
    ranks = (count_below + 1.0) / (nsh + 1)

    # Vectorized p-value computation for 'norm' distribution (14x faster than per-pair)
    # norm.fit() just computes mean+std, so vectorized mean/std + norm.sf is identical
    pvals_matrix = None
    if stage == 2 and metric_distr_type == 'norm':
        shuffle_data = metable[:, :, 1:]
        means = shuffle_data.mean(axis=2)
        stds = shuffle_data.std(axis=2)
        z_scores = (metable[:, :, 0] - means) / (stds + 1e-30)
        pvals_matrix = norm.sf(z_scores)

    for i in range(a):
        for j in range(b):
            if precomputed_mask[i, j]:
                new_stats = {}
                me = metable[i, j, 0]
                opt_delay = optimal_delays[i, j]

                if stage == 1:
                    # Stage 1 only needs rank - skip expensive p-value fitting
                    # criterion1() only checks pre_rval, not pre_pval
                    new_stats["pre_rval"] = ranks[i, j]
                    new_stats["pre_pval"] = None  # Not computed for performance
                    new_stats["opt_delay"] = opt_delay
                    new_stats["me"] = me

                elif stage == 2:
                    # Stage 2 needs p-value for multiple comparison correction
                    if pvals_matrix is not None:
                        pval = float(pvals_matrix[i, j])
                    else:
                        random_mi_samples = metable[i, j, 1:]
                        pval = get_mi_distr_pvalue(
                            random_mi_samples, me, distr_type=metric_distr_type
                        )
                    new_stats["rval"] = ranks[i, j]
                    new_stats["pval"] = pval
                    new_stats["me"] = me
                    new_stats["opt_delay"] = opt_delay

                stage_stats[i][j].update(new_stats)

    return stage_stats


def merge_stage_stats(stage1_stats, stage2_stats) -> dict:
    """
    Merge statistics from stage 1 and stage 2.

    Parameters
    ----------
    stage1_stats : dict of dict
        Statistics from stage 1 (preliminary).
    stage2_stats : dict of dict
        Statistics from stage 2 (full).

    Returns
    -------
    merged_stats : dict of dict
        Combined statistics with both stage 1 and 2 results."""
    merged_stats = stage2_stats.copy()
    for i in stage2_stats:
        for j in stage2_stats[i]:
            # Only merge if the entry exists in stage1_stats
            if i in stage1_stats and j in stage1_stats[i] and stage1_stats[i][j]:
                if "pre_rval" in stage1_stats[i][j]:
                    merged_stats[i][j]["pre_rval"] = stage1_stats[i][j]["pre_rval"]
                if "pre_pval" in stage1_stats[i][j]:
                    merged_stats[i][j]["pre_pval"] = stage1_stats[i][j]["pre_pval"]

    return merged_stats


def merge_stage_significance(stage_1_significance, stage_2_significance) -> dict:
    """
    Merge significance results from stage 1 and stage 2.

    Parameters
    ----------
    stage_1_significance : dict of dict
        Significance results from stage 1.
    stage_2_significance : dict of dict
        Significance results from stage 2.

    Returns
    -------
    merged_significance : dict of dict
        Combined significance results."""
    merged_significance = stage_2_significance.copy()
    for i in stage_2_significance:
        for j in stage_2_significance[i]:
            merged_significance[i][j].update(stage_1_significance[i][j])

    return merged_significance
