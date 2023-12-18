import tqdm
from scipy.stats import rankdata

from .stats import *
from ..information.info_base import get_1d_mi, get_multi_mi

# TODO: add cbunch and fbunch logic
def get_calcium_feature_mi_profile(exp, cell_id, feat_id, window = 1000, ds=1):

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


# TODO: factor out full MI table processing from the function below into a separate routine
def scan_pairs(cells,
               feats,
               nsh,
               joint_distr=False,
               ds=1,
               mask=None,
               mi_distr_type='gamma',
               noise_const=1e-3):

    """
    Calculates MI shuffles and statistics for given cells and features
    This function is generally assumed to be used internally,
    but can be also called manually to "look inside" high-level computation routines

    Parameters
    ----------
    cells: list of Neuron objects
        neurons to scan

    feats: list of TimeSeries objects
        features to scan

    nsh: int
        number of shuffles

    joint_distr: bool
        if joint_distr=True, ALL (sic!) TimeSeries in feats will be treated as components of a single multifeature
        default: False

    ds: int
        Downsampling constant. Every "ds" point will be taken from the data time series.
        default: 1

    mask: np.array of shape (len(cells), len(feats)) or (len(cells), 1) if joint_distr=True
          precomputed mask for skipping some of possible pairs.
          0 in mask values means calculation will be skipped.
          1 in mask values means calculation will proceed.

    mi_distr_type: str
        Distribution type for shuffled MI distribution fit. Supported options are "gamma" and "lognormal"
        default: "gamma"

    noise_const: float
        Small noise amplitude, which is added to MI and shuffled MI to improve numerical fit
        default: 1e-3


    Returns
    -------
    random_shifts: np.array of shape (nsh, len(cells))
        signal shifts used for MI distribution computation

    ranks: np.array of shape (len(cells), len(feats)) or (len(cells), 1) if joint_distr=True
        normalized rank of true MI with respect to shuffled MI.
        ranks[i,j] = 1.0 means true MI between signal of Neuron i and feature j is higher than MI for all shuffles

    ptable: np.array of shape (len(cells), len(feats)) or (len(cells), 1) if joint_distr=True
        p-values of true MI in respect to shuffled MI distribution
        In other words, it is the probability of getting the value of MI in range [true_MI, +inf) under the null hypothesis
        that true MI comes from the shuffled MI distribution

    MItotal: np.array of shape (len(cells), len(feats), nsh+1) or (len(cells), 1, nsh+1) if joint_distr=True
        Aggregated array of true and shuffled MI values.
        True MI matrix can be obtained by MItotal[:,:,0]
        Shuffled MI tensor of shape (len(cells), len(feats), nsh) or (len(cells), 1, nsh) if joint_distr=True
        can be obtained by MItotal[:,:,1:]
    """

    if joint_distr:
        f = 1
    else:
        f = len(feats)

    t = len(cells[0].ca.data)  # full time series length is the same for all cells and features
    n = len(cells)

    if mask is None:
        mask = np.ones((n, f))

    MItable = np.zeros((n, f))
    MItableshuf = np.zeros((n, f, nsh))
    ptable = np.ones((n, f))
    random_shifts = np.zeros((nsh, n), dtype=int)

    for i in tqdm.tqdm(np.arange(n)):
        cell = cells[i]
        ts1 = cell.ca

        min_shift = int(cell.get_t_off()*5)
        ca_random_shifts = np.random.randint(low=min_shift//ds, high=(t-min_shift)//ds, size=nsh)
        random_shifts[:,i] = ca_random_shifts[:]

        if joint_distr:
            if mask[i,0] == 1:
                mi0 = get_multi_mi(feats, ts1, ds=ds)
                MItable[i,0] = mi0 + np.random.random()*noise_const  # add small noise for better fitting

                for k, shift in enumerate(random_shifts[:,i]):
                    mi = get_multi_mi(feats, ts1, ds=ds, shift=shift)
                    MItableshuf[i,0,k] = mi + np.random.random()*noise_const  # add small noise for better fitting

                pval = get_mi_distr_pvalue(MItableshuf[i,0,:], mi0, distr_type = mi_distr_type)
                ptable[i,0] = pval

            else:
                MItable[i,0] = None
                MItableshuf[i,0,:] = np.array([None for _ in range(nsh)])
                ptable[i,0] = None

        else:
            for j, ts2 in enumerate(feats):
                if mask[i,j] == 1:
                    mi0 = get_1d_mi(ts1, ts2, ds=ds)
                    MItable[i,j] = mi0 + np.random.random()*noise_const  # add small noise for better fitting

                    for k, shift in enumerate(random_shifts[:,i]):
                        mi = get_1d_mi(ts1, ts2, shift=shift, ds=ds)
                        MItableshuf[i,j,k] = mi + np.random.random()*noise_const  # add small noise for better fitting

                    pval = get_mi_distr_pvalue(MItableshuf[i,j,:], mi0, distr_type = mi_distr_type)
                    ptable[i,j] = pval

                else:
                    MItable[i,j] = None
                    MItableshuf[i,j,:] = np.array([None for _ in range(nsh)])
                    ptable[i,j] = None

    MItotal = np.dstack((MItable, MItableshuf))
    ranked_total_mi = rankdata(MItotal, axis=2, nan_policy='omit')
    ranks = (ranked_total_mi[:,:,0]/(nsh+1))  # how many shuffles have MI lower than true mi

    return random_shifts, ranks, ptable, MItotal


def compute_mi_significance(exp,
                            cell_bunch=None,
                            feat_bunch=None,
                            mode='two_stage',
                            n_shuffles_stage1=100,
                            n_shuffles_stage2=10000,
                            joint_distr=False,
                            mi_distr_type='gamma',
                            noise_ampl=1e-3,
                            ds=1,
                            use_precomputed_stats=True,
                            save_computed_stats=True,
                            force_update=False,
                            multicomp_correction='holm',
                            pval_thr=0.01):

    """
    Calculates significant neuron-feature pairs

    Parameters
    ----------
    exp: Experiment instance
        Experiment object to read and write data from

    cell_bunch: int, iterable or None
        Neuron indices. By default (cell_bunch=None), all neurons will be taken

    feat_bunch: str, iterable or None
        Feature names. By default (feat_bunch=None), all single features will be taken

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

    use_precomputed_stats: bool
        Whether to use stats saved in Experiment instance. Stats are accumulated separately for stage1 and stage2.
        Notes on stats data rewriting (if save_computed_stats=True):
        If you want to recalculate stage1 results only, use "use_precomputed_stats=False" and "mode='stage1'".
        Stage 2 stats will be erased since they will become irrelevant.
        If you want to recalculate stage2 results only, use "use_precomputed_stats=True" and "mode='stage2'" or "mode='two-stage'"
        If you want to recalculate everything, use "use_precomputed_stats=False" and "mode='two-stage'"
        default: True

    save_computed_stats: bool
        Whether to save computed stats to Experiment instance
        default: True

    force_update: bool
        Whether to force saved statistics data update in case the collision between actual data hashes and
        saved stats data hashes is found (for example, if neuronal or behavior data has been changed externally).
        default: False

    multicomp_correction: str or None
        type of multiple comparisons correction. Supported types are None (no correction),
        "bonferroni" and "holm".
        default: 'holm'

    pval_thr: float
        pvalue threshold. if multicomp_correction=None, this is a p-value for a single pair.
        Otherwise it is a FWER significance level.

    Returns
    -------
    stats: dict of dict of dicts
        Outer dict: dynamic features, inner dict: cells, inner inner dict: stats.
        Can be easily converted to pandas DataFrame by pd.DataFrame(stats)
    """

    exp.check_ds(ds)

    cell_ids = exp._process_cbunch(cell_bunch)
    feat_ids = exp._process_fbunch(feat_bunch, allow_multifeatures=False)
    cells = [exp.neurons[cell_id] for cell_id in cell_ids]
    feats = [exp.dynamic_features[feat_id] for feat_id in feat_ids]

    if joint_distr:
        feat_ids = [tuple(sorted(feat_ids))]

    n, t, f = len(cells), exp.n_frames, len(feat_ids)

    precomputed_mask_stage1 = np.ones((n,f))
    precomputed_mask_stage2 = np.ones((n,f))
    mask_from_stage1 = np.zeros((n,f))
    mask_from_stage2 = np.zeros((n,f))
    nhyp = n*f

    if use_precomputed_stats:
        print('Retrieving saved stats data...')
        # 0 in mask values means precomputed results are found, calculation will be skipped.
        # 1 in mask values means precomputed results are not found or incomplete, calculation will proceed.

        for i, cell_id in enumerate(cell_ids):
            for j, feat_id in enumerate(feat_ids):
                try:
                    pair_stats = exp.get_neuron_feature_pair_stats(cell_id, feat_id)
                except ValueError:
                    if isinstance(feat_id, str):
                        raise ValueError(f'Unknown single feature in feat_bunch: {feat_id}. Check initial data')
                    else:
                        print(f'Multifeature {feat_id} is new, it will be added to stats table')
                        exp._add_multifeature_to_data_hashes(feat_id)
                        exp._add_multifeature_to_stats(feat_id)
                        pair_stats = exp.get_neuron_feature_pair_stats(cell_id, feat_id)

                data_hash = exp._data_hashes[feat_id][cell_id]

                if stats_not_empty(pair_stats, data_hash, stage=1):
                    precomputed_mask_stage1[i,j]=0
                if stats_not_empty(pair_stats, data_hash, stage=2):
                    precomputed_mask_stage2[i,j]=0

    if mode in ['two_stage', 'stage1']:
        npairs_to_check1 = int(np.sum(precomputed_mask_stage1))
        print(f'Starting stage 1 scanning for {npairs_to_check1}/{nhyp} possible pairs')

        # STAGE 1 - primary scanning
        shifts, rtable, ptable, mitotal = scan_pairs(cells,
                                                     feats,
                                                     n_shuffles_stage1,
                                                     joint_distr=joint_distr,
                                                     ds=ds,
                                                     mask=precomputed_mask_stage1,
                                                     mi_distr_type=mi_distr_type,
                                                     noise_const=noise_ampl)

        # turn computed data tables from stage 1 and precomputed data into dict of stats dicts
        stage_1_stats = get_table_of_stats(exp,
                                           cell_ids,
                                           feat_ids,
                                           rtable,
                                           ptable,
                                           mitotal,
                                           ds=ds,
                                           precomputed_mask=precomputed_mask_stage1,
                                           stage=1)

        # update Experiment saved statistics if needed
        if save_computed_stats:
            for i, cell_id in enumerate(cell_ids):
                for j, feat_id in enumerate(feat_ids):
                    if precomputed_mask_stage1[i,j]:
                        exp.update_neuron_feature_pair_stats(stage_1_stats[feat_id][cell_id],
                                                             cell_id,
                                                             feat_id,
                                                             force_update=force_update,
                                                             stage=1)

        # select potentially significant pairs for stage 2
        # 0 in mask values means the pair MI is definitely insignificant, stage 2 calculation will be skipped.
        # 1 in mask values means the pair MI is potentially significant, stage 2 calculation will proceed.

        print('Computing significance for all pairs in stage 1...')
        for i, cell_id in enumerate(cell_ids):
            for j, feat_id in enumerate(feat_ids):

                pair_passes_stage1 = criterion1(stage_1_stats[feat_id][cell_id], n_shuffles_stage1)
                sig = {'shuffles1': n_shuffles_stage1, 'stage1': pair_passes_stage1}

                # update Experiment saved significance data if needed
                if save_computed_stats:
                    exp.update_neuron_feature_pair_significance(sig, cell_id, feat_id)
                if pair_passes_stage1:
                    mask_from_stage1[i,j] = 1

        print('Stage 1 results:')
        nhyp = int(np.sum(mask_from_stage1)) #number of hypotheses for further statistical testing
        print(f'{nhyp/n/f*100:.2f}% ({nhyp}/{n*f}) of possible pairs identified as candidates')

        if mode == 'stage1':
            return stage_1_stats

    if mode in ['two_stage', 'stage2']:
        # STAGE 2 - full-scale scanning
        combined_mask_for_stage_2 = np.ones((n,f))
        combined_mask_for_stage_2[np.where(mask_from_stage1 == 0)] = 0
        combined_mask_for_stage_2[np.where(precomputed_mask_stage2 == 0)] = 0

        npairs_to_check2 = int(np.sum(combined_mask_for_stage_2))
        print(f'Starting stage 2 scanning for {npairs_to_check2}/{nhyp} possible pairs')

        shifts, rtable, ptable, mitotal = scan_pairs(cells,
                                                     feats,
                                                     n_shuffles_stage2,
                                                     joint_distr=joint_distr,
                                                     ds=ds,
                                                     mask=combined_mask_for_stage_2,
                                                     mi_distr_type=mi_distr_type,
                                                     noise_const=noise_ampl)

        # turn data tables from stage 2 to array of stats dicts
        stage_2_stats = get_table_of_stats(exp,
                                           cell_ids,
                                           feat_ids,
                                           rtable,
                                           ptable,
                                           mitotal,
                                           ds=ds,
                                           precomputed_mask=combined_mask_for_stage_2,
                                           stage=2)

        # update Experiment saved statistics if needed
        if save_computed_stats:
            for i, cell_id in enumerate(cell_ids):
                for j, feat_id in enumerate(feat_ids):
                    if combined_mask_for_stage_2[i,j]:
                        exp.update_neuron_feature_pair_stats(stage_2_stats[feat_id][cell_id],
                                                             cell_id,
                                                             feat_id,
                                                             force_update=force_update,
                                                             stage=2)

        # select significant pairs after stage 2
        print('Computing significance for all pairs in stage 2...')
        all_pvals = None
        if multicomp_correction == 'holm':  # holm procedure requires all p-values
            all_pvals = get_all_nonempty_pvals(stage_2_stats, cell_ids, feat_ids)

        multicorr_thr = get_multicomp_correction_thr(pval_thr,
                                                     mode=multicomp_correction,
                                                     all_pvals=all_pvals,
                                                     nhyp=nhyp)

        for i, cell_id in enumerate(cell_ids):
            for j, feat_id in enumerate(feat_ids):
                pair_passes_stage2 = criterion2(stage_2_stats[feat_id][cell_id],
                                                n_shuffles_stage2,
                                                multicorr_thr)

                sig = {'shuffles2': n_shuffles_stage2,
                       'stage2': pair_passes_stage2,
                       'final_p_thr': pval_thr,
                       'multicomp_corr': multicomp_correction}

                # update Experiment saved significance data if needed
                if save_computed_stats:
                    exp.update_neuron_feature_pair_significance(sig, cell_id, feat_id)
                if pair_passes_stage2:
                    mask_from_stage2[i,j] = 1

        exp._pairwise_pval_thr = multicorr_thr
        print('Stage 2 results:')
        num2 = int(np.sum(mask_from_stage2))
        print(f'{num2/n/f*100:.2f}% ({num2}/{n*f}) of possible pairs identified as significant')

        return stage_2_stats


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
            print(all_pvals)
            nhyp = len(all_pvals)

            for i, pval in enumerate(all_pvals):
                cthr = fwer / (nhyp - i)
                print(cthr)
                if pval > cthr:
                    break

            threshold = cthr

        else:
            raise ValueError('List of p-values for Holm correction not provided')

    else:
        raise ValueError('Unknown multiple comparisons correction method')

    return threshold
