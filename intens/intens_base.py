import numpy as np
import tqdm

from .stats import *
from ..information.info_base import get_1d_mi, get_multi_mi


# TODO: add cbunch and fbunch logic
def get_calcium_feature_mi_profile(exp, cell_id, feat_id, window=1000, ds=1):

    cell = exp.neurons[cell_id]
    ts1 = cell.ca
    shifted_mi = []

    if isinstance(feat_id, str):
        ts2 = exp.dynamic_features[feat_id]
        mi0 = get_1d_mi(ts1, ts2, ds=ds)

        for shift in tqdm.tqdm(np.arange(-window, window, ds)):
            lag_mi = get_1d_mi(ts1, ts2, ds=ds, shift=shift)
            shifted_mi.append(lag_mi)

    else:
        feats = [exp.dynamic_features[fid] for fid in feat_id]
        mi0 = get_multi_mi(feats, ts1, ds=ds)

        for shift in tqdm.tqdm(np.arange(-window, window, ds)):
            lag_mi = get_multi_mi(feats, ts1, ds=ds, shift=shift)
            shifted_mi.append(lag_mi)

    return mi0, shifted_mi


def scan_pairs(ts_bunch1,
               ts_bunch2,
               nsh,
               joint_distr=False,
               ds=1,
               mask=None,
               noise_const=1e-3,
               seed=None):

    """
    Calculates MI shuffles for 2 given sets of TimeSeries
    This function is generally assumed to be used internally,
    but can be also called manually to "look inside" high-level computation routines

    Parameters
    ----------
    ts_bunch1: list of TimeSeries objects

    ts_bunch2: list of TimeSeries objects

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


    Returns
    -------
    random_shifts: np.array of shape (nsh, len(cells))
        signals shifts used for MI distribution computation

    mi_total: np.array of shape (len(ts_bunch1), len(ts_bunch2)), nsh+1) or (len(ts_bunch1), 1, nsh+1) if joint_distr=True
        Aggregated array of true and shuffled MI values.
        True MI matrix can be obtained by mi_total[:,:,0]
        Shuffled MI tensor of shape (len(ts_bunch1), len(ts_bunch2)), nsh) or (len(ts_bunch1), 1, nsh) if joint_distr=True
        can be obtained by mi_total[:,:,1:]
    """

    if seed is not None:
        np.random.seed = seed

    n1 = len(ts_bunch1)
    n2 = 1 if joint_distr else len(ts_bunch2)
    t = len(ts_bunch1[0].data)  # full length is the same for all time series

    if mask is None:
        mask = np.ones((n1, n2))

    mi_table = np.zeros((n1, n2))
    mi_table_shuffles = np.zeros((n1, n2, nsh))
    random_shifts = np.zeros((n1, n2, nsh), dtype=int)

    # fill random shifts according to the allowed shuffles masks of both time series
    for i, ts1 in enumerate(ts_bunch1):
        if joint_distr:
            # TODO: add combination of ts shuffle masks for all ts from tsbunch2
            combined_shuffle_mask = ts1.shuffle_mask
            indices_to_select = np.arange(t)[combined_shuffle_mask]
            random_shifts[i, 0, :] = np.random.choice(indices_to_select, size=nsh) // ds
        else:
            for j, ts2 in enumerate(ts_bunch2):
                combined_shuffle_mask = ts1.shuffle_mask & ts2.shuffle_mask
                indices_to_select = np.arange(t)[combined_shuffle_mask]
                random_shifts[i, j, :] = np.random.choice(indices_to_select, size=nsh)//ds

    for i, ts1 in tqdm.tqdm(enumerate(ts_bunch1), total=len(ts_bunch1), position=0, leave=True):
        #min_shift = min_shifts[i]
        #ca_random_shifts = np.random.randint(low=min_shift//ds, high=(t-min_shift)//ds, size=nsh)
        #random_shifts[:,i] = ca_random_shifts[:]

        if joint_distr:
            if mask[i,0] == 1:
                mi0 = get_multi_mi(ts_bunch2, ts1, ds=ds)
                mi_table[i,0] = mi0 + np.random.random()*noise_const  # add small noise for better fitting

                for k, shift in enumerate(random_shifts[i,0,:]):
                    mi = get_multi_mi(ts_bunch2, ts1, ds=ds, shift=shift)
                    mi_table_shuffles[i,0,k] = mi + np.random.random()*noise_const  # add small noise for better fitting

            else:
                mi_table[i,0] = None
                mi_table_shuffles[i,0,:] = np.full(shape=nsh, fill_value=None)

        else:
            for j, ts2 in enumerate(ts_bunch2):
                if mask[i,j] == 1:
                    mi0 = get_1d_mi(ts1, ts2, ds=ds)
                    mi_table[i,j] = mi0 + np.random.random()*noise_const  # add small noise for better fitting

                    for k, shift in enumerate(random_shifts[i,j,:]):
                        mi = get_1d_mi(ts1, ts2, shift=shift, ds=ds)
                        mi_table_shuffles[i,j,k] = mi + np.random.random()*noise_const  # add small noise for better fitting

                else:
                    mi_table[i,j] = None
                    mi_table_shuffles[i,j,:] = np.array([None for _ in range(nsh)])

    mi_total = np.dstack((mi_table, mi_table_shuffles))

    return random_shifts, mi_total


def compute_mi_stats(ts_bunch1,
                     ts_bunch2,
                     names1=None,
                     names2=None,
                     mode='two_stage',
                     precomputed_mask_stage1=None,
                     precomputed_mask_stage2=None,
                     n_shuffles_stage1=100,
                     n_shuffles_stage2=10000,
                     joint_distr=False,
                     mi_distr_type='gamma',
                     noise_ampl=1e-3,
                     ds=1,
                     topk1=1,
                     topk2=5,
                     multicomp_correction='holm',
                     pval_thr=0.01,
                     verbose=True,
                     seed=None):

    """
    Calculates MI statistics for TimeSeries pairs

    Parameters
    ----------
    ts_bunch1: list of TimeSeries objects

    ts_bunch2: list of TimeSeries objects

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

    mi_distr_type: str
        Distribution type for shuffles MI distribution fit. Supported options are "gamma" and "lognormal"
        default: "gamma"

    noise_ampl: float
        Small noise amplitude, which is added to MI and shuffled MI to improve numerical fit
        default: 1e-3

    ds: int
        Downsampling constant. Every "ds" point will be taken from the data time series.
        Reduces the computational load, but needs caution since with large "ds" some important information may be lost.
        Experiment instance has an internal check for this effect.
        default: 1

    topk1: int
        true MI for stage 1 should be among topk1 MI shuffles
        default: 1

    topk2: int
        true MI for stage 2 should be among topk2 MI shuffles
        default: 5

    multicomp_correction: str or None
        type of multiple comparisons correction. Supported types are None (no correction),
        "bonferroni" and "holm".
        default: 'holm'

    pval_thr: float
        pvalue threshold. if multicomp_correction=None, this is a p-value for a single pair.
        Otherwise it is a FWER significance level.

    verbose: bool
        whether to print intermediate information

    seed: int
        random seed for reproducibility

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
    """

    # TODO: add automatic min_shifts from autocorrelation time

    accumulated_info = dict()

    n1 = len(ts_bunch1)
    n2 = 1 if joint_distr else len(ts_bunch2)

    if precomputed_mask_stage1 is None:
        precomputed_mask_stage1 = np.ones((n1, n2))
    if precomputed_mask_stage2 is None:
        precomputed_mask_stage2 = np.ones((n1, n2))

    mask_from_stage1 = np.zeros((n1, n2))
    mask_from_stage2 = np.zeros((n1, n2))
    nhyp = n1*n2

    if mode in ['two_stage', 'stage1']:
        npairs_to_check1 = int(np.sum(precomputed_mask_stage1))
        if verbose:
            print(f'Starting stage 1 scanning for {npairs_to_check1}/{nhyp} possible pairs')

        # STAGE 1 - primary scanning
        random_shifts1, mi_total1 = scan_pairs(ts_bunch1,
                                               ts_bunch2,
                                               n_shuffles_stage1,
                                               joint_distr=joint_distr,
                                               ds=ds,
                                               mask=precomputed_mask_stage1,
                                               noise_const=noise_ampl,
                                               seed=seed)

        accumulated_info.update({'random_shifts1': random_shifts1,
                                'mi_total1': mi_total1})
        # turn computed data tables from stage 1 and precomputed data into dict of stats dicts
        stage_1_stats = get_table_of_stats(mi_total1,
                                           mi_distr_type=mi_distr_type,
                                           nsh=n_shuffles_stage1,
                                           precomputed_mask=precomputed_mask_stage1,
                                           stage=1)

        # select potentially significant pairs for stage 2
        # 0 in mask values means the pair MI is definitely insignificant, stage 2 calculation will be skipped.
        # 1 in mask values means the pair MI is potentially significant, stage 2 calculation will proceed.

        if verbose:
            print('Computing significance for all pairs in stage 1...')

        stage_1_significance = populate_nested_dict(dict(), range(n1), range(n2))
        for i in range(n1):
            for j in range(n2):
                pair_passes_stage1 = criterion1(stage_1_stats[i][j],
                                                n_shuffles_stage1,
                                                topk=topk1)
                if pair_passes_stage1:
                    mask_from_stage1[i, j] = 1

                sig1 = {'shuffles1': n_shuffles_stage1,
                        'stage1': pair_passes_stage1}
                stage_1_significance[i][j].update(sig1)

        nhyp = int(np.sum(mask_from_stage1))  # number of hypotheses for further statistical testing
        if verbose:
            print('Stage 1 results:')
            print(f'{nhyp/n1/n2*100:.2f}% ({nhyp}/{n1*n2}) of possible pairs identified as candidates')

    if mode == 'stage1':
        final_stats = add_names_to_nested_dict(stage_1_stats, names1, names2)
        final_significance = add_names_to_nested_dict(stage_1_significance, names1, names2)

        return final_stats, final_significance, accumulated_info

    else:
        # STAGE 2 - full-scale scanning
        combined_mask_for_stage_2 = np.ones((n1, n2))
        combined_mask_for_stage_2[np.where(mask_from_stage1 == 0)] = 0  # exclude non-significant pairs from stage1
        combined_mask_for_stage_2[np.where(precomputed_mask_stage2 == 0)] = 0  # exclude precomputed stage 2 pairs

        npairs_to_check2 = int(np.sum(combined_mask_for_stage_2))
        if verbose:
            print(f'Starting stage 2 scanning for {npairs_to_check2}/{nhyp} possible pairs')

        random_shifts2, mi_total2 = scan_pairs(ts_bunch1,
                                               ts_bunch2,
                                               n_shuffles_stage2,
                                               joint_distr=joint_distr,
                                               ds=ds,
                                               mask=combined_mask_for_stage_2,
                                               noise_const=noise_ampl,
                                               seed=seed)

        accumulated_info.update({'random_shifts2': random_shifts2,
                                 'mi_total2': mi_total2})
        # turn data tables from stage 2 to array of stats dicts
        stage_2_stats = get_table_of_stats(mi_total2,
                                           mi_distr_type=mi_distr_type,
                                           nsh=n_shuffles_stage2,
                                           precomputed_mask=combined_mask_for_stage_2,
                                           stage=2)

        # select significant pairs after stage 2
        if verbose:
            print('Computing significance for all pairs in stage 2...')
        all_pvals = None
        if multicomp_correction == 'holm':  # holm procedure requires all p-values
            all_pvals = get_all_nonempty_pvals(stage_2_stats, range(n1), range(n2))

        multicorr_thr = get_multicomp_correction_thr(pval_thr,
                                                     mode=multicomp_correction,
                                                     all_pvals=all_pvals,
                                                     nhyp=nhyp)

        stage_2_significance = populate_nested_dict(dict(), range(n1), range(n2))
        for i in range(n1):
            for j in range(n2):
                pair_passes_stage2 = criterion2(stage_2_stats[i][j],
                                                n_shuffles_stage2,
                                                multicorr_thr,
                                                topk=topk2)
                if pair_passes_stage2:
                    mask_from_stage2[i,j] = 1

                sig2 = {'shuffles2': n_shuffles_stage2,
                        'stage2': pair_passes_stage2,
                        'final_p_thr': pval_thr,
                        'multicomp_corr': multicomp_correction,
                        'pairwise_pval_thr': multicorr_thr}

                stage_2_significance[i][j] = sig2

        num2 = int(np.sum(mask_from_stage2))
        if verbose:
            print('Stage 2 results:')
            print(f'{num2/n1/n2*100:.2f}% ({num2}/{n1*n2}) of possible pairs identified as significant')

        merged_stats = merge_stage_stats(stage_1_stats, stage_2_stats)
        merged_significance = merge_stage_significance(stage_1_significance, stage_2_significance)
        final_stats = add_names_to_nested_dict(merged_stats, names1, names2)
        final_significance = add_names_to_nested_dict(merged_significance, names1, names2)
        return final_stats, final_significance, accumulated_info


def get_multicomp_correction_thr(fwer, mode='holm', **multicomp_kwargs):

    '''
    Calculates pvalue threshold for a single hypothesis from FWER

    Parameters
    ----------
    fwer: float
        family-wise error rate

    mode: str or None
        type of multiple comparisons correction. Supported types are None (no correction),
        "bonferroni" and "holm".

    multicomp_kwargs: named arguments for multiple comparisons correction procedure
    '''

    if mode is None:
        threshold = fwer

    elif mode == 'bonferroni':
        if 'nhyp' in multicomp_kwargs:
            threshold = fwer / multicomp_kwargs['nhyp']
        else:
            raise ValueError('Number of hypotheses for Bonferroni correction not provided')

    elif mode == 'holm':
        if 'all_pvals' in multicomp_kwargs:
            all_pvals = sorted(multicomp_kwargs['all_pvals'])
            nhyp = len(all_pvals)

            cthr=1
            for i, pval in enumerate(all_pvals):
                cthr = fwer / (nhyp - i)
                if pval > cthr:
                    break

            threshold = cthr

        else:
            raise ValueError('List of p-values for Holm correction not provided')

    else:
        raise ValueError('Unknown multiple comparisons correction method')

    return threshold
