from .stats import *
from .intense_base import compute_me_stats, IntenseResults
from ..information.info_base import TimeSeries, MultiTimeSeries


def compute_cell_feat_significance(exp,
                                  cell_bunch=None,
                                  feat_bunch=None,
                                  data_type='calcium',
                                  metric='mi',
                                  mode='two_stage',
                                  n_shuffles_stage1=100,
                                  n_shuffles_stage2=10000,
                                  joint_distr=False,
                                  allow_mixed_dimensions=False,
                                  metric_distr_type='gamma',
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
                                  skip_delays=[],
                                  shift_window=5,
                                  verbose=True,
                                  enable_parallelization=True,
                                  n_jobs=-1,
                                  seed=42):

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
        Data type used for INTENSE computations. Can be 'calcium' or 'spikes'

    metric: similarity metric between TimeSeries
        default: 'mi'

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
        if True, ALL features in feat_bunch will be treated as components of a single multifeature
        For example, 'x' and 'y' features will be put together into ('x','y') multifeature.
        default: False

    allow_mixed_dimensions: bool
        if True, both TimeSeries and MultiTimeSeries can be provided as signals.
        This parameter overrides "joint_distr"

    metric_distr_type: str
        Distribution type for shuffled metric distribution fit. Supported options are distributions from scipy.stats
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
        type of multiple comparison correction. Supported types are None (no correction),
        "bonferroni" and "holm".
        default: 'holm'

    pval_thr: float
        pvalue threshold. if multicomp_correction=None, this is a p-value for a single pair.
        Otherwise it is a FWER significance level.

    find_optimal_delays: bool
        Allows slight shifting (not more than +- shift_window) of time series,
        selects a shift with the highest MI as default.
        default: True

    skip_delays: list
        List of features for which delays are not applied (set to 0).
        Has no effect if find_optimal_delays = False

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
    feat_ids = exp._process_fbunch(feat_bunch, allow_multifeatures=True, mode=data_type)
    cells = [exp.neurons[cell_id] for cell_id in cell_ids]

    if data_type == 'calcium':
        signals = [cell.ca for cell in cells]
    elif data_type == 'spikes':
        signals = [cell.sp for cell in cells]
    else:
        raise ValueError('"data_type" can be either "calcium" or "spikes"')

    #min_shifts = [int(cell.get_t_off() * MIN_CA_SHIFT) for cell in cells]
    if not allow_mixed_dimensions:
        feats = [exp.dynamic_features[feat_id] for feat_id in feat_ids if hasattr(exp, feat_id)]
        if joint_distr:
            feat_ids = [tuple(sorted(feat_ids))]
    else:
        feats = []
        for feat_id in feat_ids:
            if isinstance(feat_id, str):
                ts = exp.dynamic_features[feat_id]
                feats.append(ts)
            elif isinstance(feat_id, tuple):
                parts = [exp.dynamic_features[f] for f in feat_id]
                mts = MultiTimeSeries(parts)
                feats.append(mts)
            else:
                raise ValueError('Unknown feature id type')

    n, t, f = len(cells), exp.n_frames, len(feats)

    precomputed_mask_stage1 = np.ones((n,f))
    precomputed_mask_stage2 = np.ones((n,f))

    if not exp.selectivity_tables_initialized:
        exp._set_selectivity_tables(data_type, cbunch=cell_ids, fbunch=feat_ids)

    if use_precomputed_stats:
        print('Retrieving saved stats data...')
        # 0 in mask values means precomputed results are found, calculation will be skipped.
        # 1 in mask values means precomputed results are not found or incomplete, calculation will proceed.

        for i, cell_id in enumerate(cell_ids):
            for j, feat_id in enumerate(feat_ids):
                try:
                    pair_stats = exp.get_neuron_feature_pair_stats(cell_id, feat_id, mode=data_type)
                except (ValueError, KeyError):
                    if isinstance(feat_id, str):
                        raise ValueError(f'Unknown single feature in feat_bunch: {feat_id}. Check initial data')
                    else:
                        exp._add_multifeature_to_data_hashes(feat_id, mode=data_type)
                        exp._add_multifeature_to_stats(feat_id, mode=data_type)
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

    computed_stats, computed_significance, info = compute_me_stats(signals,
                                                                   feats,
                                                                   mode=mode,
                                                                   names1=cell_ids,
                                                                   names2=feat_ids,
                                                                   metric=metric,
                                                                   precomputed_mask_stage1=precomputed_mask_stage1,
                                                                   precomputed_mask_stage2=precomputed_mask_stage2,
                                                                   n_shuffles_stage1=n_shuffles_stage1,
                                                                   n_shuffles_stage2=n_shuffles_stage2,
                                                                   joint_distr=joint_distr,
                                                                   allow_mixed_dimensions=allow_mixed_dimensions,
                                                                   metric_distr_type=metric_distr_type,
                                                                   noise_ampl=noise_ampl,
                                                                   ds=ds,
                                                                   topk1=topk1,
                                                                   topk2=topk2,
                                                                   multicomp_correction=multicomp_correction,
                                                                   pval_thr=pval_thr,
                                                                   find_optimal_delays=find_optimal_delays,
                                                                   skip_delays=[feat_ids.index(f) for f in skip_delays],
                                                                   shift_window=shift_window*exp.fps,
                                                                   verbose=verbose,
                                                                   enable_parallelization=enable_parallelization,
                                                                   n_jobs=n_jobs,
                                                                   seed=seed)

    exp.optimal_nf_delays = info['optimal_delays']
    # add hash data and update Experiment saved statistics and significance if needed
    for i, cell_id in enumerate(cell_ids):
        for j, feat_id in enumerate(feat_ids):
            # TODO: add check for non-existing feature if use_precomputed_stats==False
            computed_stats[cell_id][feat_id]['data_hash'] = exp._data_hashes[data_type][feat_id][cell_id]

            me_val = computed_stats[cell_id][feat_id].get('me')
            if me_val is not None and metric == 'mi':
                feat_entropy = exp.get_feature_entropy(feat_id, ds=ds)
                ca_entropy = exp.neurons[int(cell_id)].ca.get_entropy(ds=ds)
                computed_stats[cell_id][feat_id]['rel_me_beh'] = me_val / feat_entropy
                computed_stats[cell_id][feat_id]['rel_me_ca'] = me_val / ca_entropy

            if save_computed_stats:
                stage2_only = True if mode == 'stage2' else False
                if combined_precomputed_mask[i,j]:
                    exp.update_neuron_feature_pair_stats(computed_stats[cell_id][feat_id],
                                                         cell_id,
                                                         feat_id,
                                                         mode=data_type,
                                                         force_update=force_update,
                                                         stage2_only=stage2_only)

                    sig = computed_significance[cell_id][feat_id]
                    exp.update_neuron_feature_pair_significance(sig, cell_id, feat_id, mode=data_type)

    # save all results to a single object
    intense_params = {
        'neurons': {i: cell_ids[i] for i in range(len(cell_ids))},
        'feat_bunch': {i: feat_ids[i] for i in range(len(feat_ids))},
        'data_type': data_type,
        'mode': mode,
        'metric': metric,
        'n_shuffles_stage1': n_shuffles_stage1,
        'n_shuffles_stage2': n_shuffles_stage2,
        'joint_distr': joint_distr,
        'metric_distr_type': metric_distr_type,
        'noise_ampl': noise_ampl,
        'ds': ds,
        'topk1': topk1,
        'topk2': topk2,
        'multicomp_correction': multicomp_correction,
        'pval_thr': pval_thr,
        'find_optimal_delays': find_optimal_delays,
        'shift_window': shift_window
    }

    intense_res = IntenseResults()
    #intense_res.update('stats', computed_stats)
    #intense_res.update('significance', computed_significance)
    intense_res.update('info', info)
    intense_res.update('intense_params', intense_params)
    # Return multiple values for backward compatibility
    return computed_stats, computed_significance, info, intense_res

