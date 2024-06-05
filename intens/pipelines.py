from .stats import *
from .intens_base import compute_mi_stats


def compute_cell_feat_mi_significance(exp,
                                      cell_bunch=None,
                                      feat_bunch=None,
                                      data_type='calcium',
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
                                      find_optimal_delays=True,
                                      shift_window=5,
                                      verbose=True):

    """
    Calculates significant neuron-feature pairs

    Parameters
    ----------
    exp: Experiment instance
        Experiment object to read and write data from

    cell_bunch: int, iterable or None
        Neuron indices. By default, (cell_bunch=None), all neurons will be taken

    feat_bunch: str, iterable or None
        Feature names. By default, (feat_bunch=None), all single features will be taken

    data_type: str
        Data used for INTENS computations. Can be 'calcium' or 'spikes'

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
        Experiment class performs an internal check for this effect.
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

    find_optimal_delays: bool
        Allows slight shifting (not more than +- shift_window) of time series,
        selects a shift with the highest MI as default.
        default: True

    shift_window: int
        Window for optimal shift search (seconds). Optimal shift (in frames) will lie in the range
        -shift_window*fps <= opt_shift <= shift_window*fps
        Has no effect if find_optimal_delays = False

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

    if data_type == 'calcium':
        signals = [cell.ca for cell in cells]
    elif data_type == 'spikes':
        signals = [cell.sp for cell in cells]
    else:
        raise ValueError('"data_type" can be either "calcium" or "spikes"')
    #min_shifts = [int(cell.get_t_off() * MIN_CA_SHIFT) for cell in cells]
    feats = [exp.dynamic_features[feat_id] for feat_id in feat_ids]

    if joint_distr:
        feat_ids = [tuple(sorted(feat_ids))]

    n, t, f = len(cells), exp.n_frames, len(feat_ids)

    precomputed_mask_stage1 = np.ones((n,f))
    precomputed_mask_stage2 = np.ones((n,f))

    if use_precomputed_stats:
        print('Retrieving saved stats data...')
        # 0 in mask values means precomputed results are found, calculation will be skipped.
        # 1 in mask values means precomputed results are not found or incomplete, calculation will proceed.

        for i, cell_id in enumerate(cell_ids):
            for j, feat_id in enumerate(feat_ids):
                try:
                    # TODO: add data_type everywhere in stats
                    pair_stats = exp.get_neuron_feature_pair_stats(cell_id, feat_id)
                except ValueError:
                    if isinstance(feat_id, str):
                        raise ValueError(f'Unknown single feature in feat_bunch: {feat_id}. Check initial data')
                    else:
                        exp._add_multifeature_to_data_hashes(feat_id, mode=data_type)
                        exp._add_multifeature_to_stats(feat_id)
                        pair_stats = DEFAULT_STATS.copy()

                current_data_hash = exp._data_hashes[data_type][feat_id][cell_id]

                if stats_not_empty(pair_stats, current_data_hash, stage=1):
                    precomputed_mask_stage1[i,j] = 0
                if stats_not_empty(pair_stats, current_data_hash, stage=2):
                    precomputed_mask_stage2[i,j] = 0

    combined_precomputed_mask = np.ones((n, f))
    if mode in ['stage2', 'two_stage']:
        combined_precomputed_mask[np.where((precomputed_mask_stage1 == 0) & (precomputed_mask_stage2 == 0))] = 0
    elif mode == 'stage1':
        combined_precomputed_mask[np.where(precomputed_mask_stage1 == 0)] = 0
    else:
        raise ValueError('Wrong mode!')

    computed_stats, computed_significance, info = compute_mi_stats(signals,
                                                                   feats,
                                                                   mode=mode,
                                                                   names1=cell_ids,
                                                                   names2=feat_ids,
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
                                                                   find_optimal_delays=find_optimal_delays,
                                                                   shift_window=shift_window*exp.fps,
                                                                   verbose=verbose)

    # add hash data and update Experiment saved statistics and significance if needed
    for i, cell_id in enumerate(cell_ids):
        for j, feat_id in enumerate(feat_ids):
            # TODO: add check for non-existing feature if use_precomputed_stats==False
            computed_stats[cell_id][feat_id]['data_hash'] = exp._data_hashes[data_type][feat_id][cell_id]

            mi_val = computed_stats[cell_id][feat_id].get('mi')
            if mi_val is not None:
                feat_entropy = exp.get_feature_entropy(feat_id, ds=ds)
                ca_entropy = exp.neurons[int(cell_id)].ca.get_entropy(ds=ds)
                computed_stats[cell_id][feat_id]['rel_mi_beh'] = mi_val / feat_entropy
                computed_stats[cell_id][feat_id]['rel_mi_ca'] = mi_val / ca_entropy

            if save_computed_stats:
                stage2_only = True if mode == 'stage2' else False
                if combined_precomputed_mask[i,j]:
                    exp.update_neuron_feature_pair_stats(computed_stats[cell_id][feat_id],
                                                         cell_id,
                                                         feat_id,
                                                         force_update=force_update,
                                                         stage2_only=stage2_only)

                    sig = computed_significance[cell_id][feat_id]
                    exp.update_neuron_feature_pair_significance(sig, cell_id, feat_id)

    return computed_stats, computed_significance, info
