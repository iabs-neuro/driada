import numpy as np
from scipy.stats import gamma, lognorm


def chebyshev_ineq(data, val):
    z = (val - np.mean(data))/np.std(data)
    #print(z)
    return 1./z**2


def get_lognormal_p(data, val):
    params = lognorm.fit(data, floc=0)
    rv = lognorm(*params)
    return rv.sf(val)


def get_gamma_p(data, val):
    params = gamma.fit(data, floc=0)
    rv = lognorm(*params)
    return rv.sf(val)


def get_mi_distr_pvalue(data, val, distr_type = 'gamma'):
    if distr_type == 'lognormal':
        distr = lognorm
    elif distr_type == 'gamma':
        distr = gamma
    else:
        raise ValueError(f'wrong MI distribution: {distr_type}')

    params = distr.fit(data, floc=0)
    rv = distr(*params)
    return rv.sf(val)


def get_mask(ptable, rtable, pval_thr, rank_thr):
    mask = np.ones(ptable.shape)
    mask[np.where(ptable>pval_thr)] = 0
    mask[np.where(rtable<rank_thr)] = 0
    return mask


def stats_not_empty(pair_stats, data_hash, stage=1):
    if stage==1:
        stats_to_check = ['pre_rval', 'pre_pval']
    elif stage==2:
        stats_to_check = ['rval', 'pval', 'mi']
    else:
        raise ValueError(f'Stage should be 1 or 2, but {stage} was passed')

    is_valid = (data_hash == pair_stats['data_hash'])
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

    if pair_stats['pre_rval'] is not None:
        return pair_stats['pre_rval'] > (1 - topk/nsh1)
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
        True if significance confirmed, False if not.
    """

    if pair_stats['rval'] is not None and pair_stats['pval'] is not None: # whether pair passed stage 1 and has statistics from stage 2
        if pair_stats['rval'] > (1 - topk/nsh2): #  whether true MI is among topk shuffles (in practice it is top-1 almost always)
            criterion = pair_stats['pval'] < pval_thr
            return criterion
        else:
            return False
    else:
        return False


def get_all_nonempty_pvals(all_stats, cell_ids, feat_ids):

    all_pvals = []
    for i, cell_id in enumerate(cell_ids):
        for j, feat_id in enumerate(feat_ids):
            pval = all_stats[feat_id][cell_id]['pval']
            if not(pval is None):
                all_pvals.append(pval)

    return all_pvals


def get_table_of_stats(exp,
                       cell_ids,
                       feat_ids,
                       rtable,
                       ptable,
                       mitable,
                       precomputed_mask=None,
                       ds=1,
                       stage=1):

    # 0 in mask values means that stats for this pair will be taken from Experiment instance.
    # 1 in mask values means that stats for this pair will be calculated from new results.
    if precomputed_mask is None:
        precomputed_mask = np.ones((len(cell_ids), len(feat_ids)))

    all_stats = exp._populate_dict(dict(), fbunch=feat_ids, cbunch=cell_ids)

    for j, feat_id in enumerate(feat_ids):
        if not isinstance(feat_id, str):
            print(f'Multifeature {feat_id} is new, it will be added to stats table')
            exp._add_multifeature_to_data_hashes(feat_id)
            exp._add_multifeature_to_stats(feat_id)

        for i, cell_id in enumerate(cell_ids):
            if precomputed_mask[i,j]:
                new_stats = exp.null_stats_dict.copy()
                data_hash = exp._data_hashes[feat_id][cell_id]
                new_stats['data_hash'] = data_hash

                if stage==1:
                    new_stats['pre_rval'] = rtable[i,j]
                    new_stats['pre_pval'] = ptable[i,j]

                elif stage==2:
                    new_stats['pre_rval'] = exp.stats_table[feat_id][cell_id]['pre_rval']
                    new_stats['pre_pval'] = exp.stats_table[feat_id][cell_id]['pre_pval']

                    new_stats['rval'] = rtable[i,j]
                    new_stats['pval'] = ptable[i,j]
                    new_stats['mi'] = mitable[i,j,0]

                    feat_entropy = exp.get_feature_entropy(feat_id, ds=ds)
                    ca_entropy = exp.neurons[int(cell_id)].ca.get_entropy(ds=ds)
                    new_stats['rel_mi_beh'] = mitable[i,j,0]/feat_entropy
                    new_stats['rel_mi_ca'] = mitable[i,j,0]/ca_entropy

            else:
                new_stats = exp.get_neuron_feature_pair_stats(cell_id, feat_id)

            all_stats[feat_id][cell_id].update(new_stats)

    return all_stats