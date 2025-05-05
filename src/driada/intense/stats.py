import numpy as np
import scipy
from scipy.stats import *
from ..utils.data import populate_nested_dict, add_names_to_nested_dict
from ..experiment.exp_base import DEFAULT_STATS


def chebyshev_ineq(data, val):
    z = (val - np.mean(data))/np.std(data)
    return 1./z**2


def get_lognormal_p(data, val):
    params = lognorm.fit(data, floc=0)
    rv = lognorm(*params)
    return rv.sf(val)


def get_gamma_p(data, val):
    params = gamma.fit(data, floc=0)
    rv = gamma(*params)
    return rv.sf(val)


def get_distribution_function(dist_name):
    try:
        return getattr(scipy.stats, dist_name)
    except AttributeError:
        raise ValueError(f"Distribution '{dist_name}' not found in scipy.stats")


def get_mi_distr_pvalue(data, val, distr_type='gamma'):
    distr = get_distribution_function(distr_type)
    #try:
    if distr_type in ['gamma', 'lognorm']:
        params = distr.fit(data, floc=0)
    else:
        params = distr.fit(data)

    rv = distr(*params)
    return rv.sf(val)

    #except: # some rare error in function fitting
    #return 1.0


def get_mask(ptable, rtable, pval_thr, rank_thr):
    mask = np.ones(ptable.shape)
    mask[np.where(ptable > pval_thr)] = 0
    mask[np.where(rtable < rank_thr)] = 0
    return mask


def stats_not_empty(pair_stats, current_data_hash, stage=1):
    if stage == 1:
        stats_to_check = ['pre_rval', 'pre_pval']
    elif stage == 2:
        stats_to_check = ['rval', 'pval', 'me']
    else:
        raise ValueError(f'Stage should be 1 or 2, but {stage} was passed')

    data_hash_from_stats = pair_stats['data_hash']
    is_valid = (current_data_hash == data_hash_from_stats)
    is_not_empty = np.all(np.array([pair_stats[st] is not None for st in stats_to_check]))
    return is_valid and is_not_empty


def criterion1(pair_stats, nsh1, topk=1):
    """
    Calculates whether the given neuron-feature pair is potentially significant after preliminary shuffling

    Parameters
    ----------
    pair_stats: dict
        dictionary of computed stats

    nsh1: int
        number of shuffles for first stage

    topk: int
        true MI should be among topk MI shuffles
        default: 1

    Returns
    -------
    crit_passed: bool
        True if significance confirmed, False if not.
    """

    if pair_stats.get('pre_rval') is not None:
        return pair_stats['pre_rval'] > (1 - 1.*topk/(nsh1+1))
        #return pair_stats['pre_rval'] == 1 # true MI should be top-1 among all shuffles
    else:
        return False


def criterion2(pair_stats, nsh2, pval_thr, topk=5):
    """
    Calculates whether the given neuron-feature pair is significant after full-scale shuffling

    Parameters
    ----------
    pair_stats: dict
        dictionary of computed stats

    nsh2: int
        number of shuffles for second stage

    pval_thr: float
        pvalue threshold for a single pair. It depends on a FWER significance level and multiple
        hypothesis correction algorithm.

    topk: int
        true MI should be among topk MI shuffles
        default: 5

    Returns
    -------
    crit_passed: bool
        True if significance is confirmed, False if not.
    """
    # whether pair passed stage 1 and has statistics from stage 2
    if pair_stats.get('rval') is not None and pair_stats.get('pval') is not None:
        # whether true MI is among topk shuffles (in practice it is top-1 almost always)
        if pair_stats['rval'] > (1 - 1.*topk/(nsh2+1)):
            criterion = pair_stats['pval'] < pval_thr
            return criterion
        else:
            return False
    else:
        return False


def get_all_nonempty_pvals(all_stats, ids1, ids2):
    all_pvals = []
    for i, id1 in enumerate(ids1):
        for j, id2 in enumerate(ids2):
            pval = all_stats[id1][id2].get('pval')
            if pval is not None:
                all_pvals.append(pval)

    return all_pvals


def get_table_of_stats(metable,
                       optimal_delays,
                       precomputed_mask=None,
                       metric_distr_type='gamma',
                       nsh=0,
                       stage=1):

    # 0 in mask values means that stats for this pair will not be calculated
    # 1 in mask values means that stats for this pair will be calculated from new results.
    if precomputed_mask is None:
        precomputed_mask = np.ones(metable.shape)

    a, b, sh = metable.shape
    stage_stats = populate_nested_dict(dict(), range(a), range(b))

    ranked_total_mi = rankdata(metable, axis=2, nan_policy='omit')
    ranks = (ranked_total_mi[:, :, 0] / (nsh + 1))  # how many shuffles have MI lower than true mi

    for i in range(a):
        for j in range(b):
            if precomputed_mask[i, j]:
                new_stats = {}#DEFAULT_STATS.copy()
                me = metable[i, j, 0]
                random_mi_samples = metable[i, j, 1:]
                pval = get_mi_distr_pvalue(random_mi_samples, me, distr_type=metric_distr_type)
                opt_delay = optimal_delays[i, j]

                if stage == 1:
                    new_stats['pre_rval'] = ranks[i, j]
                    new_stats['pre_pval'] = pval
                    new_stats['opt_delay'] = opt_delay

                elif stage == 2:
                    new_stats['rval'] = ranks[i,j]
                    new_stats['pval'] = pval
                    new_stats['me'] = metable[i,j,0]
                    new_stats['opt_delay'] = opt_delay

                stage_stats[i][j].update(new_stats)

    return stage_stats


def merge_stage_stats(stage1_stats, stage2_stats):
    merged_stats = stage2_stats.copy()
    for i in stage2_stats:
        for j in stage2_stats[i]:
            merged_stats[i][j]['pre_rval'] = stage1_stats[i][j]['pre_rval']
            merged_stats[i][j]['pre_pval'] = stage1_stats[i][j]['pre_pval']

    return merged_stats


def merge_stage_significance(stage_1_significance, stage_2_significance):
    merged_significance = stage_2_significance.copy()
    for i in stage_2_significance:
        for j in stage_2_significance[i]:
            merged_significance[i][j].update(stage_1_significance[i][j])

    return merged_significance
