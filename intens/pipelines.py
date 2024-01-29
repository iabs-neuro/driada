from .stats import *
from .intens_base import scan_pairs, compute_mi_significance, get_multicomp_correction_thr

MIN_CA_SHIFT = 5  # MIN_SHIFT*t_off is the minimal random signal shift for a given cell


def compute_cell_feat_mi_significance(exp,
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
                                      topk1=1,
                                      topk2=5,
                                      multicomp_correction='holm',
                                      pval_thr=0.01,
                                      verbose=True):

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

    Returns
    -------
    stats: dict of dict of dicts
        Outer dict: dynamic features, inner dict: cells, last dict: stats.
        Can be easily converted to pandas DataFrame by pd.DataFrame(stats)
    """

    exp.check_ds(ds)

    cell_ids = exp._process_cbunch(cell_bunch)
    feat_ids = exp._process_fbunch(feat_bunch, allow_multifeatures=False)
    cells = [exp.neurons[cell_id] for cell_id in cell_ids]

    ca_signals = [cell.ca for cell in cells]
    min_shifts = [int(cell.get_t_off() * MIN_CA_SHIFT) for cell in cells]
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
                        exp._add_multifeature_to_data_hashes(feat_id)
                        exp._add_multifeature_to_stats(feat_id)
                        pair_stats = DEFAULT_STATS.copy()

                current_data_hash = exp._data_hashes[feat_id][cell_id]

                if stats_not_empty(pair_stats, current_data_hash, stage=1):
                    precomputed_mask_stage1[i,j] = 0
                if stats_not_empty(pair_stats, current_data_hash, stage=2):
                    precomputed_mask_stage2[i,j] = 0

    combined_precomputed_mask = np.ones((n, f))
    if mode in ['stage2', 'two-stage']:
        combined_precomputed_mask[np.where((precomputed_mask_stage1 == 0) & (precomputed_mask_stage2 == 0))] = 0
    elif mode == 'stage1':
        combined_precomputed_mask[np.where(precomputed_mask_stage1 == 0)] = 0
    else:
        raise ValueError('Wrong mode!')

    computed_stats = compute_mi_significance(ca_signals,
                                             feats,
                                             mode=mode,
                                             names1=cell_ids,
                                             names2=feat_ids,
                                             min_shifts=min_shifts,
                                             precomputed_mask_stage1=precomputed_mask_stage1,
                                             precomputed_mask_stage2=precomputed_mask_stage2,
                                             n_shuffles_stage1=n_shuffles_stage1,
                                             n_shuffles_stage2=n_shuffles_stage2,
                                             joint_distr=joint_distr,
                                             mi_distr_type=mi_distr_type,
                                             noise_ampl=noise_ampl,
                                             ds=ds,
                                             topk1=topk1,
                                             topk2=topk2,
                                             multicomp_correction=multicomp_correction,
                                             pval_thr=pval_thr,
                                             verbose=verbose)

    # add hash data and update Experiment saved statistics if needed
    for i, cell_id in enumerate(cell_ids):
        for j, feat_id in enumerate(feat_ids):
            computed_stats[feat_id][cell_id]['data_hash'] = exp._data_hashes[feat_id][cell_id]

            mi_val = computed_stats[feat_id][cell_id]['mi']
            if mi_val is not None:
                feat_entropy = exp.get_feature_entropy(feat_id, ds=ds)
                ca_entropy = exp.neurons[int(cell_id)].ca.get_entropy(ds=ds)
                computed_stats[feat_id][cell_id]['rel_mi_beh'] = mi_val / feat_entropy
                computed_stats[feat_id][cell_id]['rel_mi_ca'] = mi_val / ca_entropy

            if save_computed_stats:
                stage2_only = True if mode == 'stage2' else False

                if combined_precomputed_mask[i,j]:
                    exp.update_neuron_feature_pair_stats(computed_stats[feat_id][cell_id],
                                                         cell_id,
                                                         feat_id,
                                                         force_update=force_update,
                                                         stage2_only=stage2_only)

        # select potentially significant pairs for stage 2
        # 0 in mask values means the pair MI is definitely insignificant, stage 2 calculation will be skipped.
        # 1 in mask values means the pair MI is potentially significant, stage 2 calculation will proceed.

        print('Computing significance for all pairs in stage 1...')
        for i, cell_id in enumerate(cell_ids):
            for j, feat_id in enumerate(feat_ids):

                pair_passes_stage1 = criterion1(stage_1_stats[feat_id][cell_id],
                                                n_shuffles_stage1,
                                                topk=topk1)

                sig = {'shuffles1': n_shuffles_stage1, 'stage1': pair_passes_stage1}

                # update Experiment saved significance data if needed
                if save_computed_stats:
                    exp.update_neuron_feature_pair_significance(sig, cell_id, feat_id)
                if pair_passes_stage1:
                    mask_from_stage1[i, j] = 1

        if mode == 'stage1':
            return stage_1_stats

    if mode in ['two_stage', 'stage2']:
        # STAGE 2 - full-scale scanning
        combined_mask_for_stage_2 = np.ones((n, f))
        combined_mask_for_stage_2[np.where(mask_from_stage1 == 0)] = 0
        combined_mask_for_stage_2[np.where(precomputed_mask_stage2 == 0)] = 0

        npairs_to_check2 = int(np.sum(combined_mask_for_stage_2))
        print(f'Starting stage 2 scanning for {npairs_to_check2}/{nhyp} possible pairs')

        random_shifts2, mi_total2 = scan_pairs(ca_signals,
                                               feats,
                                               n_shuffles_stage2,
                                               joint_distr=joint_distr,
                                               ds=ds,
                                               mask=combined_mask_for_stage_2,
                                               noise_const=noise_ampl,
                                               min_shifts=min_shifts)

        # turn data tables from stage 2 to array of stats dicts
        stage_2_stats = get_updated_table_of_stats(exp,
                                                   cell_ids,
                                                   feat_ids,
                                                   mi_total2,
                                                   ds=ds,
                                                   mi_distr_type=mi_distr_type,
                                                   nsh=n_shuffles_stage2,
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
                                                multicorr_thr,
                                                topk=topk2)

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


compute_mi_significance(ts_bunch1,
                        ts_bunch2,
                        mode='two_stage',
                        min_shifts=None,
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
                        verbose=True)