import numpy as np
import scipy
from scipy.stats import lognorm, gamma, rankdata
from ..utils.data import populate_nested_dict


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
    Chebyshev's inequality states that P(|X - μ| >= k*σ) <= 1/k²
    This gives P(X >= val) <= 1/z² where z = (val - μ)/σ    """
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
    Log-normal distribution is suitable for positive-valued data.    """
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
    Gamma distribution is suitable for positive-valued data.    """
    data = np.asarray(data)
    if np.any(data <= 0):
        raise ValueError("Data must contain only positive values for gamma distribution")
    
    params = gamma.fit(data, floc=0)
    rv = gamma(*params)
    return rv.sf(val)


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
        If distribution name not found in scipy.stats.    """
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
        Distribution type to fit. Default: 'gamma'.

    Returns
    -------
    p_value : float
        P(X >= val) under fitted distribution.
        Returns 1.0 if distribution fitting fails.

    Notes
    -----
    - For 'gamma' and 'lognorm', fits with floc=0 (zero lower bound)
    - For other distributions, uses default fitting
    - Returns conservative p-value (1.0) on fitting errors    """
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
        (p <= pval_thr AND rank >= rank_thr), 0 otherwise.    """
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
        If stage is not 1 or 2.    """
    if stage == 1:
        stats_to_check = ["pre_rval", "pre_pval"]
    elif stage == 2:
        stats_to_check = ["rval", "pval", "me"]
    else:
        raise ValueError(f"Stage should be 1 or 2, but {stage} was passed")

    data_hash_from_stats = pair_stats["data_hash"]
    is_valid = current_data_hash == data_hash_from_stats
    is_not_empty = np.all(
        np.array([pair_stats[st] is not None for st in stats_to_check])
    )
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
    For topk=1 and nsh1=100, this requires pre_rval > 0.99    """

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
    Missing 'rval' or 'pval' results in False.    """
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


def get_all_nonempty_pvals(all_stats, ids1, ids2):
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
        List of all non-None p-values found.    """
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
    metric_distr_type="gamma",
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
        Nested dictionary with computed statistics for each pair.    """
    # 0 in mask values means that stats for this pair will not be calculated
    # 1 in mask values means that stats for this pair will be calculated from new results.
    if precomputed_mask is None:
        precomputed_mask = np.ones(metable.shape[:2])

    a, b, sh = metable.shape
    stage_stats = populate_nested_dict(dict(), range(a), range(b))

    ranked_total_mi = rankdata(metable, axis=2, nan_policy="omit")
    ranks = ranked_total_mi[:, :, 0] / (
        nsh + 1
    )  # how many shuffles have MI lower than true mi

    for i in range(a):
        for j in range(b):
            if precomputed_mask[i, j]:
                new_stats = {}  # DEFAULT_STATS.copy()
                me = metable[i, j, 0]
                random_mi_samples = metable[i, j, 1:]
                pval = get_mi_distr_pvalue(
                    random_mi_samples, me, distr_type=metric_distr_type
                )
                opt_delay = optimal_delays[i, j]

                if stage == 1:
                    new_stats["pre_rval"] = ranks[i, j]
                    new_stats["pre_pval"] = pval
                    new_stats["opt_delay"] = opt_delay
                    new_stats["me"] = metable[i, j, 0]  # Add MI value for stage 1 too

                elif stage == 2:
                    new_stats["rval"] = ranks[i, j]
                    new_stats["pval"] = pval
                    new_stats["me"] = metable[i, j, 0]
                    new_stats["opt_delay"] = opt_delay

                stage_stats[i][j].update(new_stats)

    return stage_stats


def merge_stage_stats(stage1_stats, stage2_stats):
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
        Combined statistics with both stage 1 and 2 results.    """
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


def merge_stage_significance(stage_1_significance, stage_2_significance):
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
        Combined significance results.    """
    merged_significance = stage_2_significance.copy()
    for i in stage_2_significance:
        for j in stage_2_significance[i]:
            merged_significance[i][j].update(stage_1_significance[i][j])

    return merged_significance
