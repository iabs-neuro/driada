import numpy as np
import tqdm
from joblib import Parallel, delayed
import multiprocessing

from .stats import *
from ..information.info_base import TimeSeries, get_1d_mi, get_multi_mi, get_mi, get_sim
from ..utils.data import write_dict_to_hdf5, nested_dict_to_seq_of_tables


def calculate_optimal_delays(ts_bunch1, ts_bunch2, metric,
                             shift_window, ds, verbose=True, enable_progressbar=True):
    if verbose:
        print('Calculating optimal delays:')

    optimal_delays = np.zeros((len(ts_bunch1), len(ts_bunch2)), dtype=int)
    shifts = np.arange(-shift_window, shift_window, ds) // ds

    for i, ts1 in tqdm.tqdm(enumerate(ts_bunch1), total=len(ts_bunch1), disable=not enable_progressbar):
        for j, ts2 in enumerate(ts_bunch2):
            shifted_me = []
            for shift in shifts:
                lag_me = get_sim(ts1, ts2, metric, ds=ds, shift=int(shift))
                shifted_me.append(lag_me)

            best_shift = shifts[np.argmax(shifted_me)]
            optimal_delays[i, j] = int(best_shift*ds)

    return optimal_delays


def calculate_optimal_delays_parallel(ts_bunch1, ts_bunch2, metric,
                                      shift_window, ds, verbose=True, n_jobs=-1):
    if verbose:
        print('Calculating optimal delays in parallel mode:')

    optimal_delays = np.zeros((len(ts_bunch1), len(ts_bunch2)), dtype=int)

    if n_jobs == -1:
        n_jobs = min(multiprocessing.cpu_count(), len(ts_bunch1))

    split_ts_bunch1_inds = np.array_split(np.arange(len(ts_bunch1)), n_jobs)
    split_ts_bunch1 = [np.array(ts_bunch1)[idxs] for idxs in split_ts_bunch1_inds]

    parallel_delays = Parallel(n_jobs=n_jobs, verbose=True)(
        delayed(calculate_optimal_delays)(small_ts_bunch,
                                          ts_bunch2,
                                          metric,
                                          shift_window,
                                          ds,
                                          verbose=False,
                                          enable_progressbar=False)
        for small_ts_bunch in split_ts_bunch1)

    for i, pd in enumerate(parallel_delays):
        inds_of_interest = split_ts_bunch1_inds[i]
        optimal_delays[inds_of_interest, :] = pd

    return optimal_delays


# TODO: add cbunch and fbunch logic
def get_calcium_feature_me_profile(exp, cell_id, feat_id, window=1000, ds=1):

    cell = exp.neurons[cell_id]
    ts1 = cell.ca
    shifted_mi = []

    if isinstance(feat_id, str):
        ts2 = exp.dynamic_features[feat_id]
        me0 = get_1d_mi(ts1, ts2, ds=ds)

        for shift in tqdm.tqdm(np.arange(-window, window, ds)//ds):
            lag_mi = get_1d_mi(ts1, ts2, ds=ds, shift=shift)
            shifted_mi.append(lag_mi)

    else:
        feats = [exp.dynamic_features[fid] for fid in feat_id]
        me0 = get_multi_mi(feats, ts1, ds=ds)

        for shift in tqdm.tqdm(np.arange(-window, window, ds)):
            lag_mi = get_multi_mi(feats, ts1, ds=ds, shift=shift)
            shifted_mi.append(lag_mi)

    return me0, shifted_mi


def scan_pairs(ts_bunch1,
               ts_bunch2,
               metric,
               nsh,
               optimal_delays,
               joint_distr=False,
               ds=1,
               mask=None,
               noise_const=1e-3,
               seed=None,
               enable_progressbar=True):

    """
    Calculates MI shuffles for 2 given sets of TimeSeries
    This function is generally assumed to be used internally,
    but can be also called manually to "look inside" high-level computation routines

    Parameters
    ----------
    ts_bunch1: list of TimeSeries objects

    ts_bunch2: list of TimeSeries objects
    
    metric: similarity metric between TimeSeries
    
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

    optimal_delays: np.array of shape (len(ts_bunch1), len(ts_bunch2)) or (len(ts_bunch), 1) if joint_distr=True
        best shifts from original time series alignment in terms of MI.

    seed: int
        Random seed for reproducibility

    Returns
    -------
    random_shifts: np.array of shape (len(ts_bunch1), len(ts_bunch2), nsh)
        signals shifts used for MI distribution computation

    me_total: np.array of shape (len(ts_bunch1), len(ts_bunch2)), nsh+1) or (len(ts_bunch1), 1, nsh+1) if joint_distr==True
        Aggregated array of true and shuffled MI values.
        True MI matrix can be obtained by me_total[:,:,0]
        Shuffled MI tensor of shape (len(ts_bunch1), len(ts_bunch2)), nsh) or (len(ts_bunch1), 1, nsh) if joint_distr==True
        can be obtained by me_total[:,:,1:]
    """

    if seed is None:
        seed = 0

    np.random.seed(seed)

    n1 = len(ts_bunch1)
    n2 = 1 if joint_distr else len(ts_bunch2)
    lengths1 = [len(ts.data) if isinstance(ts, TimeSeries) else ts.data.shape[1] for ts in ts_bunch1]
    lengths2 = [len(ts.data) if isinstance(ts, TimeSeries) else ts.data.shape[1] for ts in ts_bunch2]
    if len(set(lengths1)) == 1 and len(set(lengths2)) == 1 and set(lengths1) == set(lengths2):
        t = lengths1[0]  # full length is the same for all time series
    else:
        raise ValueError('Lenghts of TimeSeries do not match!')

    if mask is None:
        mask = np.ones((n1, n2))

    me_table = np.zeros((n1, n2))
    me_table_shuffles = np.zeros((n1, n2, nsh))
    random_shifts = np.zeros((n1, n2, nsh), dtype=int)

    # fill random shifts according to the allowed shuffles masks of both time series
    for i, ts1 in enumerate(ts_bunch1):
        if joint_distr:
            np.random.seed(seed)
            # TODO: add combination of ts shuffle masks for all ts from tsbunch2
            combined_shuffle_mask = ts1.shuffle_mask
            # move shuffle mask according to optimal shift
            combined_shuffle_mask = np.roll(combined_shuffle_mask, int(optimal_delays[i, 0]))
            indices_to_select = np.arange(t)[combined_shuffle_mask]
            random_shifts[i, 0, :] = np.random.choice(indices_to_select, size=nsh) // ds

        else:
            for j, ts2 in enumerate(ts_bunch2):
                np.random.seed(seed)
                combined_shuffle_mask = ts1.shuffle_mask & ts2.shuffle_mask
                # move shuffle mask according to optimal shift
                combined_shuffle_mask = np.roll(combined_shuffle_mask, int(optimal_delays[i, j]))
                indices_to_select = np.arange(t)[combined_shuffle_mask]
                random_shifts[i, j, :] = np.random.choice(indices_to_select, size=nsh)//ds

    # calculate similarity metric arrays
    for i, ts1 in tqdm.tqdm(enumerate(ts_bunch1),
                            total=len(ts_bunch1),
                            position=0,
                            leave=True,
                            disable=not enable_progressbar):

        np.random.seed(seed)

        # TODO: deprecate this branch, it is unnecessary with MultiTimeSeries
        if joint_distr:
            if metric != 'mi':
                raise ValueError("joint_distr mode works with metric = 'mi' only")
            if mask[i,0] == 1:
                # default metric without shuffling, minus due to different order
                me0 = get_multi_mi(ts_bunch2, ts1, ds=ds, shift=-optimal_delays[i, 0]//ds)
                me_table[i,0] = me0 + np.random.random()*noise_const  # add small noise for better fitting

                np.random.seed(seed)
                random_noise = np.random.random(size=len(random_shifts[i, 0, :])) * noise_const  # add small noise for better fitting
                for k, shift in enumerate(random_shifts[i, 0, :]):
                    mi = get_multi_mi(ts_bunch2, ts1, ds=ds, shift=shift)
                    me_table_shuffles[i,0,k] = mi + random_noise[k]

            else:
                me_table[i,0] = None
                me_table_shuffles[i,0,:] = np.full(shape=nsh, fill_value=None)

        else:
            for j, ts2 in enumerate(ts_bunch2):
                if mask[i,j] == 1:
                    me0 = get_sim(ts1,
                                  ts2,
                                  metric,
                                  ds=ds,
                                  shift=optimal_delays[i, j]//ds,
                                  check_for_coincidence=True)  # default metric without shuffling

                    np.random.seed(seed)
                    me_table[i,j] = me0 + np.random.random()*noise_const  # add small noise for better fitting

                    np.random.seed(seed)
                    random_noise = np.random.random(
                        size=len(random_shifts[i, j, :])) * noise_const  # add small noise for better fitting

                    for k, shift in enumerate(random_shifts[i,j,:]):
                        np.random.seed(seed)
                        #mi = get_1d_mi(ts1, ts2, shift=shift, ds=ds)
                        me = get_sim(ts1,
                                     ts2,
                                     metric,
                                     ds=ds,
                                     shift=shift)

                        me_table_shuffles[i,j,k] = me + random_noise[k]

                else:
                    me_table[i,j] = None
                    me_table_shuffles[i,j,:] = np.array([None for _ in range(nsh)])

    me_total = np.dstack((me_table, me_table_shuffles))

    return random_shifts, me_total


def scan_pairs_parallel(ts_bunch1,
                        ts_bunch2,
                        metric,
                        nsh,
                        optimal_delays,
                        joint_distr=False,
                        ds=1,
                        mask=None,
                        noise_const=1e-3,
                        seed=None,
                        n_jobs=-1):

    n1 = len(ts_bunch1)
    n2 = 1 if joint_distr else len(ts_bunch2)
    me_total = np.zeros((n1, n2, nsh+1))
    random_shifts = np.zeros((n1, n2, nsh), dtype=int)

    if n_jobs == -1:
        n_jobs = min(multiprocessing.cpu_count(), n1)

    split_ts_bunch1_inds = np.array_split(np.arange(len(ts_bunch1)), n_jobs)
    split_ts_bunch1 = [np.array(ts_bunch1)[idxs] for idxs in split_ts_bunch1_inds]
    split_optimal_delays = [optimal_delays[idxs] for idxs in split_ts_bunch1_inds]
    split_mask = [mask[idxs] for idxs in split_ts_bunch1_inds]

    parallel_result = Parallel(n_jobs=n_jobs, verbose=True)(
        delayed(scan_pairs)(small_ts_bunch,
                            ts_bunch2,
                            metric,
                            nsh,
                            split_optimal_delays[_],
                            joint_distr=joint_distr,
                            ds=ds,
                            mask=split_mask[_],
                            noise_const=noise_const,
                            seed=seed,
                            enable_progressbar=False)
        for _, small_ts_bunch in enumerate(split_ts_bunch1))

    for i in range(n_jobs):
        inds_of_interest = split_ts_bunch1_inds[i]
        random_shifts[inds_of_interest, :, :] = parallel_result[i][0][:, :, :]
        me_total[inds_of_interest, :, :] = parallel_result[i][1][:, :, :]

    return random_shifts, me_total


def scan_pairs_router(ts_bunch1,
                      ts_bunch2,
                      metric,
                      nsh,
                      optimal_delays,
                      joint_distr=False,
                      ds=1,
                      mask=None,
                      noise_const=1e-3,
                      seed=None,
                      enable_parallelization=True,
                      n_jobs=-1):

    if enable_parallelization:
        random_shifts, me_total = scan_pairs_parallel(ts_bunch1,
                                                      ts_bunch2,
                                                      metric,
                                                      nsh,
                                                      optimal_delays,
                                                      joint_distr=joint_distr,
                                                      ds=ds,
                                                      mask=mask,
                                                      noise_const=noise_const,
                                                      seed=seed,
                                                      n_jobs=n_jobs)

    else:
        random_shifts, me_total = scan_pairs(ts_bunch1,
                                             ts_bunch2,
                                             metric,
                                             nsh,
                                             optimal_delays,
                                             joint_distr=joint_distr,
                                             ds=ds,
                                             mask=mask,
                                             seed=seed,
                                             noise_const=noise_const)

    return random_shifts, me_total


class IntenseResults(object):
    '''
    A simple wrapper for convenient storage of INTENSE results
    '''
    def __init__(self):
        pass

    def update(self, property_name, data):
        setattr(self, property_name, data)

    def update_multiple(self, datadict):
        for dname, data in datadict.items:
            setattr(self, dname, data)

    def save_to_hdf5(self, fname):
        dict_repr = self.__dict__
        write_dict_to_hdf5(dict_repr, fname)


def compute_me_stats(ts_bunch1,
                     ts_bunch2,
                     names1=None,
                     names2=None,
                     mode='two_stage',
                     metric='mi',
                     precomputed_mask_stage1=None,
                     precomputed_mask_stage2=None,
                     n_shuffles_stage1=100,
                     n_shuffles_stage2=10000,
                     joint_distr=False,
                     allow_mixed_dimensions=False,
                     metric_distr_type='gamma',
                     noise_ampl=1e-3,
                     ds=1,
                     topk1=1,
                     topk2=5,
                     multicomp_correction='holm',
                     pval_thr=0.01,
                     find_optimal_delays=False,
                     skip_delays=[],
                     shift_window=100,
                     verbose=True,
                     seed=None,
                     enable_parallelization=True,
                     n_jobs=-1):

    """
    Calculates similarity metric statistics for TimeSeries or MultiTimeSeries pairs

    Parameters
    ----------
    ts_bunch1: list of TimeSeries objects

    ts_bunch2: list of TimeSeries objects

    names1: list of str
        names than will be given to time series from tsbunch1 in final results

    names2: list of str
        names than will be given to time series from tsbunch2 in final results

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
    
    metric: similarity metric between TimeSeries
        default: 'mi'
        
    precomputed_mask_stage1: np.array of shape (len(ts_bunch1), len(ts_bunch2)) or (len(ts_bunch), 1) if joint_distr=True
          precomputed mask for skipping some of possible pairs in stage 1.
          0 in mask values means calculation will be skipped.
          1 in mask values means calculation will proceed.

    precomputed_mask_stage2: np.array of shape (len(ts_bunch1), len(ts_bunch2)) or (len(ts_bunch), 1) if joint_distr=True
          precomputed mask for skipping some of possible pairs in stage 2.
          0 in mask values means calculation will be skipped.
          1 in mask values means calculation will proceed.

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

    allow_mixed_dimensions: bool
        if True, both TimeSeries and MultiTimeSeries can be provided as signals.
        This parameter overrides "joint_distr"
        default: False

    metric_distr_type: str
        Distribution type for shuffled metric distribution fit. Supported options are distributions from scipy.stats
        default: "gamma"

    noise_ampl: float
        Small noise amplitude, which is added to metrics to improve numerical fit
        default: 1e-3

    ds: int
        Downsampling constant. Every "ds" point will be taken from the data time series.
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

    find_optimal_delays: bool
        Allows slight shifting (not more than +- shift_window) of time series,
        selects a shift with the highest MI as default.
        default: True

    skip_delays: list
        List of indices from ts_bunch2 for which delays are not applied (set to 0).
        Has no effect if find_optimal_delays = False

    shift_window: int
        Window for optimal shift search (frames). Optimal shift will lie in the range
        -shift_window <= opt_shift <= shift_window

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

    accumulated_info: dict
        Data collected during computation.
    """

    # TODO: add automatic min_shifts from autocorrelation time

    accumulated_info = dict()

    n1 = len(ts_bunch1)
    n2 = len(ts_bunch2)
    if not allow_mixed_dimensions:
        n2 = 1 if joint_distr else len(ts_bunch2)

        tsbunch1_is_1d = np.all([isinstance(ts, TimeSeries) for ts in ts_bunch1])
        tsbunch2_is_1d = np.all([isinstance(ts, TimeSeries) for ts in ts_bunch2])
        if not (tsbunch1_is_1d and tsbunch2_is_1d):
            raise ValueError('Multiple time series types found, but allow_mixed_dimensions=False.'
                             'Consider setting it to True')

    if precomputed_mask_stage1 is None:
        precomputed_mask_stage1 = np.ones((n1, n2))
    if precomputed_mask_stage2 is None:
        precomputed_mask_stage2 = np.ones((n1, n2))

    # TODO: add a keyword argument for behavior on duplicate TS ('ignore/raise error')

    optimal_delays = np.zeros((n1, n2), dtype=int)
    ts_with_delays = [ts for _, ts in enumerate(ts_bunch2) if _ not in skip_delays]
    ts_with_delays_inds = np.array([_ for _, ts in enumerate(ts_bunch2) if _ not in skip_delays])

    if find_optimal_delays:
        if enable_parallelization:
            optimal_delays_res = calculate_optimal_delays_parallel(ts_bunch1,
                                                                   ts_with_delays,
                                                                   metric,
                                                                   shift_window,
                                                                   ds,
                                                                   verbose=verbose,
                                                                   n_jobs=n_jobs)
        else:
            optimal_delays_res = calculate_optimal_delays(ts_bunch1,
                                                          ts_with_delays,
                                                          metric,
                                                          shift_window,
                                                          ds,
                                                          verbose=verbose)

        optimal_delays[:, ts_with_delays_inds] = optimal_delays_res

    accumulated_info['optimal_delays'] = optimal_delays

    mask_from_stage1 = np.zeros((n1, n2))
    mask_from_stage2 = np.zeros((n1, n2))
    nhyp = n1*n2

    if mode in ['two_stage', 'stage1']:
        npairs_to_check1 = int(np.sum(precomputed_mask_stage1))
        if verbose:
            print(f'Starting stage 1 scanning for {npairs_to_check1}/{nhyp} possible pairs')

        # STAGE 1 - primary scanning
        random_shifts1, me_total1 = scan_pairs_router(ts_bunch1,
                                                      ts_bunch2,
                                                      metric,
                                                      n_shuffles_stage1,
                                                      optimal_delays,
                                                      joint_distr=joint_distr,
                                                      ds=ds,
                                                      mask=precomputed_mask_stage1,
                                                      noise_const=noise_ampl,
                                                      seed=seed,
                                                      enable_parallelization=enable_parallelization,
                                                      n_jobs=n_jobs)

        # turn computed data tables from stage 1 and precomputed data into dict of stats dicts
        stage_1_stats = get_table_of_stats(me_total1,
                                           optimal_delays,
                                           metric_distr_type=metric_distr_type,
                                           nsh=n_shuffles_stage1,
                                           precomputed_mask=precomputed_mask_stage1,
                                           stage=1)

        stage_1_stats_per_quantity = nested_dict_to_seq_of_tables(stage_1_stats,
                                                                  ordered_names1=range(n1),
                                                                  ordered_names2=range(n2))
        #print(stage_1_stats_per_quantity)

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

                sig1 = {'stage1': pair_passes_stage1}
                stage_1_significance[i][j].update(sig1)

        stage_1_significance_per_quantity = nested_dict_to_seq_of_tables(stage_1_significance,
                                                                          ordered_names1=range(n1),
                                                                          ordered_names2=range(n2))

        #print(stage_1_significance_per_quantity)
        accumulated_info.update(
            {
                'stage_1_significance': stage_1_significance_per_quantity,
                'stage_1_stats': stage_1_stats_per_quantity,
                'random_shifts1': random_shifts1,
                'me_total1': me_total1
            }
        )

        nhyp = int(np.sum(mask_from_stage1))  # number of hypotheses for further statistical testing
        if verbose:
            print('Stage 1 results:')
            print(f'{nhyp/n1/n2*100:.2f}% ({nhyp}/{n1*n2}) of possible pairs identified as candidates')

    if mode == 'stage1' or nhyp == 0:
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

        random_shifts2, me_total2 = scan_pairs_router(ts_bunch1,
                                                      ts_bunch2,
                                                      metric,
                                                      n_shuffles_stage2,
                                                      optimal_delays,
                                                      joint_distr=joint_distr,
                                                      ds=ds,
                                                      mask=combined_mask_for_stage_2,
                                                      noise_const=noise_ampl,
                                                      seed=seed,
                                                      enable_parallelization=enable_parallelization,
                                                      n_jobs=n_jobs)

        # turn data tables from stage 2 to array of stats dicts
        stage_2_stats = get_table_of_stats(me_total2,
                                           optimal_delays,
                                           metric_distr_type=metric_distr_type,
                                           nsh=n_shuffles_stage2,
                                           precomputed_mask=combined_mask_for_stage_2,
                                           stage=2)

        stage_2_stats_per_quantity = nested_dict_to_seq_of_tables(stage_2_stats,
                                                                  ordered_names1=range(n1),
                                                                  ordered_names2=range(n2))
        #print(stage_2_stats_per_quantity)

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

                sig2 = {'stage2': pair_passes_stage2}
                stage_2_significance[i][j] = sig2

        stage_2_significance_per_quantity = nested_dict_to_seq_of_tables(stage_2_significance,
                                                                          ordered_names1=range(n1),
                                                                          ordered_names2=range(n2))

        #print(stage_2_significance_per_quantity)
        accumulated_info.update(
            {
                'stage_2_significance': stage_2_significance_per_quantity,
                'stage_2_stats': stage_2_stats_per_quantity,
                'random_shifts2': random_shifts2,
                'me_total2': me_total2,
                'corrected_pval_thr': multicorr_thr,
                'group_pval_thr': pval_thr,
            }
        )

        num2 = int(np.sum(mask_from_stage2))
        if verbose:
            print('Stage 2 results:')
            print(f'{num2/n1/n2*100:.2f}% ({num2}/{n1*n2}) of possible pairs identified as significant')

        if mode == 'two_stage':
            merged_stats = merge_stage_stats(stage_1_stats, stage_2_stats)
            merged_significance = merge_stage_significance(stage_1_significance, stage_2_significance)
            final_stats = add_names_to_nested_dict(merged_stats, names1, names2)
            final_significance = add_names_to_nested_dict(merged_significance, names1, names2)
            return final_stats, final_significance, accumulated_info

        else:
            return stage_2_stats, stage_2_significance, accumulated_info


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
